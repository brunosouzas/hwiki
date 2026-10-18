"""The qmd backend keeps excluded folders out of qmd's own index.

These tests run the real ``qmd`` CLI (BM25 only, so no models are downloaded)
against a throwaway config dir and cache (``QMD_CONFIG_DIR``, ``XDG_CACHE_HOME``).
They are skipped when qmd is not installed; point ``HWIKI_QMD_BINARY`` at it
if it is not on ``PATH``.

"Never indexed" is checked on the index files themselves: the excluded text
must not appear anywhere in qmd's SQLite files, not just in search results.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import _hwiki_path  # noqa: F401
import pytest

from hwiki.search import QmdBackend, _glob_for_qmd, qmd_ignore_patterns
from hwiki.store import WikiStore

ALLOWED = "allowedwombat"
RAW = "rawpangolin"
PRIVATE = "privatenarwhal"
INDEX = "hwiki-test"


def qmd_binary() -> str:
    binary = os.environ.get("HWIKI_QMD_BINARY") or shutil.which("qmd") or ""
    if not binary or not Path(binary).exists() and not shutil.which(binary):
        pytest.skip("qmd is not installed (set HWIKI_QMD_BINARY)")
    return binary


@pytest.fixture
def qmd_env(tmp_path, monkeypatch):
    config_dir = tmp_path / "qmd-config"
    cache_dir = tmp_path / "qmd-cache"
    monkeypatch.setenv("QMD_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
    return {"config": config_dir, "cache": cache_dir, "binary": qmd_binary()}


@pytest.fixture
def wiki(tmp_path):
    root = tmp_path / "wiki"
    (root / "notes").mkdir(parents=True)
    (root / "raw").mkdir()
    (root / "private").mkdir()
    (root / "notes" / "a.md").write_text(f"# Allowed\n\nThe {ALLOWED} lives here.\n", encoding="utf-8")
    (root / "raw" / "b.md").write_text(f"# Raw\n\nThe {RAW} is third-party.\n", encoding="utf-8")
    (root / "private" / "c.md").write_text(f"# Private\n\nThe {PRIVATE} is private.\n", encoding="utf-8")
    return root


def index_bytes(cache_dir: Path) -> bytes:
    files = sorted((cache_dir / "qmd").glob(f"{INDEX}.sqlite*"))
    assert files, "qmd created no index files"
    return b"".join(f.read_bytes() for f in files)


def raw_qmd_search(binary: str, term: str) -> list:
    """Ask qmd directly, bypassing hwiki's query-time filter."""
    proc = subprocess.run([binary, "--index", INDEX, "search", term, "--json"],
                          capture_output=True, text=True, timeout=60, check=False)
    text = proc.stdout.strip()
    return json.loads(text[text.find("["):]) if "[" in text else []


def backend(store: WikiStore, binary: str) -> QmdBackend:
    return QmdBackend(store, collection="wiki", binary=binary, mode="search", timeout=30, index=INDEX)


def test_excluded_folders_are_never_indexed_by_qmd(qmd_env, wiki):
    store = WikiStore(wiki, untrusted_dirs=["raw"], exclude=["private/*"])
    backend(store, qmd_env["binary"]).sync()

    data = index_bytes(qmd_env["cache"])
    assert ALLOWED.encode() in data
    assert RAW.encode() not in data
    assert PRIVATE.encode() not in data
    assert raw_qmd_search(qmd_env["binary"], RAW) == []
    assert raw_qmd_search(qmd_env["binary"], PRIVATE) == []
    assert [h.path for h in backend(store, qmd_env["binary"]).search(ALLOWED)] == ["notes/a.md"]


def test_newly_excluded_folder_is_purged_from_the_index(qmd_env, wiki):
    before = WikiStore(wiki, untrusted_dirs=[], exclude=[])
    backend(before, qmd_env["binary"]).sync()
    assert PRIVATE.encode() in index_bytes(qmd_env["cache"])

    after = WikiStore(wiki, untrusted_dirs=["raw"], exclude=["private/*"])
    backend(after, qmd_env["binary"]).sync()

    data = index_bytes(qmd_env["cache"])
    assert PRIVATE.encode() not in data
    assert RAW.encode() not in data
    assert ALLOWED.encode() in data


def test_deleted_file_is_purged_from_the_index(qmd_env, wiki):
    store = WikiStore(wiki, untrusted_dirs=["raw"], exclude=["private/*"])
    backend(store, qmd_env["binary"]).sync()
    (wiki / "notes" / "a.md").unlink()
    backend(store, qmd_env["binary"]).sync()
    assert ALLOWED.encode() not in index_bytes(qmd_env["cache"])


def test_symlinks_into_an_excluded_folder_are_not_indexed(qmd_env, wiki):
    (wiki / "notes" / "dir-link").symlink_to(wiki / "private", target_is_directory=True)
    (wiki / "notes" / "file-link.md").symlink_to(wiki / "private" / "c.md")
    store = WikiStore(wiki, untrusted_dirs=["raw"], exclude=["private/*"])
    backend(store, qmd_env["binary"]).sync()

    assert PRIVATE.encode() not in index_bytes(qmd_env["cache"])
    assert raw_qmd_search(qmd_env["binary"], PRIVATE) == []


def test_default_qmd_index_is_never_touched(qmd_env, wiki):
    store = WikiStore(wiki, untrusted_dirs=["raw"], exclude=["private/*"])
    backend(store, qmd_env["binary"]).sync()
    assert (qmd_env["config"] / f"{INDEX}.yml").is_file()
    assert not (qmd_env["config"] / "index.yml").exists()
    assert not (qmd_env["cache"] / "qmd" / "index.sqlite").exists()


def test_project_local_qmd_config_is_not_adopted(qmd_env, wiki, tmp_path, monkeypatch):
    project = tmp_path / "some-project"
    (project / ".qmd").mkdir(parents=True)
    (project / ".qmd" / "index.yml").write_text(
        f"collections:\n  wiki:\n    path: {json.dumps(str(wiki))}\n    pattern: \"**/*.md\"\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    store = WikiStore(wiki, untrusted_dirs=["raw"], exclude=["private/*"])
    backend(store, qmd_env["binary"]).sync()

    assert not (project / ".qmd" / "index.sqlite").exists()
    assert PRIVATE.encode() not in index_bytes(qmd_env["cache"])


def test_index_config_keeps_other_keys(qmd_env, wiki):
    pytest.importorskip("yaml")
    import yaml

    qmd_env["config"].mkdir(parents=True)
    path = qmd_env["config"] / f"{INDEX}.yml"
    path.write_text("models:\n  embed: hf:example/model.gguf\ncollections:\n  stale:\n    path: /tmp\n",
                    encoding="utf-8")
    store = WikiStore(wiki, untrusted_dirs=["raw"], exclude=["private/*"])
    qmd = backend(store, qmd_env["binary"])
    assert qmd.ensure_collection()

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["models"] == {"embed": "hf:example/model.gguf"}
    assert list(data["collections"]) == ["wiki"]
    assert data["collections"]["wiki"]["path"] == str(wiki.resolve())
    assert "raw/**" in data["collections"]["wiki"]["ignore"]


@pytest.mark.parametrize("pattern, expected", [
    (".git/*", ".git/**"),
    ("*/.git/*", "**/.git/**"),
    ("*.tmp.md", "**/*.tmp.md"),
    ("drafts/*.md", "drafts/**/*.md"),
    ("90-archive/**", "90-archive/**"),
    ("99 - Local Private/*", "99 - Local Private/**"),
    ("log.md", "log.md"),
    ("/top/*", "top/**"),
])
def test_exclude_globs_translate_to_qmd_globs(pattern, expected):
    assert _glob_for_qmd(pattern) == expected


def test_ignore_patterns_cover_untrusted_and_default_excludes(tmp_path):
    store = WikiStore(tmp_path, untrusted_dirs=["raw", "vendor/docs"], exclude=["private/*"])
    patterns = qmd_ignore_patterns(store)
    for expected in ("raw/**", "vendor/docs/**", "private/**", ".obsidian/**", "**/.git/**"):
        assert expected in patterns
