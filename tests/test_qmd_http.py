"""qmd over its HTTP server (``qmd_url``), snippet cleaning and background qmd sync.

Parsing is tested against responses recorded from qmd 2.8.3 (``tests/fixtures``),
served by a stand-in HTTP server that also records each request. The end-to-end
test starts a real ``qmd mcp --http`` server on a throwaway index (BM25 only, so no
models are downloaded); it is skipped when qmd is not installed (set
``HWIKI_QMD_BINARY`` if it is not on ``PATH``).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import _hwiki_path  # noqa: F401
import pytest

from hwiki.search import HybridBackend, QmdBackend, clean_snippet, loopback_url
from hwiki.store import WikiStore

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# -- helpers ---------------------------------------------------------------

@pytest.fixture
def wiki(tmp_path):
    root = tmp_path / "wiki"
    (root / "notes").mkdir(parents=True)
    (root / "private").mkdir()
    (root / "notes" / "release.md").write_text(
        "# Release process\n\n## Tagging\n\nThe allowedwombat release is tagged on Fridays.\n",
        encoding="utf-8",
    )
    (root / "private" / "c.md").write_text("# Private\n\nThe privatenarwhal is private.\n",
                                           encoding="utf-8")
    return root


@pytest.fixture
def isolated_qmd(tmp_path, monkeypatch):
    monkeypatch.setenv("QMD_CONFIG_DIR", str(tmp_path / "qmd-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "qmd-cache"))


class StandIn:
    """Loopback HTTP server answering every POST with one recorded body."""

    def __init__(self, body: bytes, status: int = 200, delay: float = 0.0):
        self.requests: list = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append({"path": self.path, "body": json.loads(self.rfile.read(length))})
                time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def qmd_binary() -> str:
    binary = os.environ.get("HWIKI_QMD_BINARY") or shutil.which("qmd") or ""
    if not binary or not Path(binary).exists() and not shutil.which(binary):
        pytest.skip("qmd is not installed (set HWIKI_QMD_BINARY)")
    return binary


# -- recorded responses ----------------------------------------------------

def test_http_search_parses_recorded_response_and_sends_the_query(wiki, isolated_qmd):
    stand_in = StandIn((FIXTURES / "qmd_http_query_lex.json").read_bytes())
    try:
        store = WikiStore(wiki, untrusted_dirs=[], exclude=["private/*"])
        qmd = QmdBackend(store, collection="wiki", mode="search", url=stand_in.url, timeout=5)
        hits = qmd.search("allowedwombat release", limit=4)
    finally:
        stand_in.close()

    assert [h.path for h in hits] == ["notes/release.md"]
    assert hits[0].snippet == "The allowedwombat release is tagged on Fridays."
    assert hits[0].title == "Release process"
    [request] = stand_in.requests
    assert request["path"] == "/query"
    assert request["body"] == {
        "searches": [{"type": "lex", "query": "allowedwombat release"}],
        "collections": ["wiki"],
        "limit": 12,
        "rerank": False,
    }


@pytest.mark.parametrize("mode, kinds, rerank_setting, rerank_sent", [
    ("vsearch", ["vec"], True, False),
    ("query", ["lex", "vec"], False, False),
    ("query", ["lex", "vec"], True, True),
])
def test_http_request_shape_per_mode(wiki, isolated_qmd, mode, kinds, rerank_setting, rerank_sent):
    stand_in = StandIn(b'{"results": []}')
    try:
        store = WikiStore(wiki, untrusted_dirs=[])
        qmd = QmdBackend(store, collection="wiki", mode=mode, url=stand_in.url, rerank=rerank_setting)
        assert qmd.search("anything", limit=2) == []
    finally:
        stand_in.close()
    body = stand_in.requests[0]["body"]
    assert [s["type"] for s in body["searches"]] == kinds
    assert body["rerank"] is rerank_sent


def test_http_results_still_pass_the_trust_boundary(wiki, isolated_qmd):
    body = json.dumps({"results": [
        {"file": "qmd://wiki/private/c.md", "title": "Private", "snippet": "privatenarwhal"},
        {"file": "qmd://wiki/../../etc/passwd", "title": "x", "snippet": "x"},
        {"file": "qmd://wiki/notes/release.md", "title": "Release process", "snippet": "7: ok"},
    ]}).encode()
    stand_in = StandIn(body)
    try:
        store = WikiStore(wiki, untrusted_dirs=[], exclude=["private/*"])
        hits = QmdBackend(store, collection="wiki", url=stand_in.url).search("x", limit=5)
    finally:
        stand_in.close()
    assert [h.path for h in hits] == ["notes/release.md"]


def test_http_ignores_proxy_settings(wiki, isolated_qmd, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")  # nothing listens on port 9
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(urllib.request, "_opener", None)
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    stand_in = StandIn((FIXTURES / "qmd_http_query_lex.json").read_bytes())
    try:
        store = WikiStore(wiki, untrusted_dirs=[])
        hits = QmdBackend(store, collection="wiki", url=stand_in.url).search("allowedwombat")
    finally:
        stand_in.close()
    assert [h.path for h in hits] == ["notes/release.md"]


def test_http_failure_returns_nothing_within_the_timeout(wiki, isolated_qmd):
    store = WikiStore(wiki, untrusted_dirs=[])
    down = QmdBackend(store, collection="wiki", url=f"http://127.0.0.1:{free_port()}", timeout=2)
    assert down.search("allowedwombat") == []

    slow = StandIn(b'{"results": []}', delay=3)
    try:
        qmd = QmdBackend(store, collection="wiki", url=slow.url, timeout=0.5)
        started = time.perf_counter()
        assert qmd.search("allowedwombat") == []
        assert time.perf_counter() - started < 2
    finally:
        slow.close()

    broken = StandIn(b"not json", status=500)
    try:
        assert QmdBackend(store, collection="wiki", url=broken.url).search("allowedwombat") == []
    finally:
        broken.close()


@pytest.mark.parametrize("url, expected", [
    ("http://localhost:8181", "http://localhost:8181"),
    ("http://127.0.0.1:8181/", "http://127.0.0.1:8181"),
    ("http://[::1]:8181", "http://[::1]:8181"),
    ("https://localhost:8181", None),
    ("http://example.com:8181", None),
    ("http://192.168.1.10:8181", None),
    ("http://localhost.example.com", None),
    ("http://user:pw@localhost:8181", None),
    ("http://localhost:99999", None),
    ("localhost:8181", None),
    ("", None),
])
def test_only_loopback_http_urls_are_accepted(url, expected):
    assert loopback_url(url) == expected


def test_non_loopback_url_falls_back_to_the_cli(wiki, isolated_qmd):
    store = WikiStore(wiki, untrusted_dirs=[])
    assert QmdBackend(store, url="http://example.com:8181").url is None


def test_snippets_from_recorded_cli_and_http_output_are_cleaned():
    cli = json.loads((FIXTURES / "qmd_cli_search.json").read_text(encoding="utf-8"))
    http = json.loads((FIXTURES / "qmd_http_query_lex.json").read_text(encoding="utf-8"))
    expected = "The allowedwombat release is tagged on Fridays."
    assert clean_snippet(cli[0]["snippet"]) == expected
    assert clean_snippet(http["results"][0]["snippet"]) == expected
    assert clean_snippet("plain text, no markers") == "plain text, no markers"


def test_cli_result_with_index_suffix_maps_to_the_page(wiki):
    store = WikiStore(wiki, untrusted_dirs=[])
    qmd = QmdBackend(store, collection="wiki")
    hits = qmd._parse((FIXTURES / "qmd_cli_search.json").read_text(encoding="utf-8"), 5)
    assert [h.path for h in hits] == ["notes/release.md"]


# -- background qmd sync ---------------------------------------------------

class BlockingQmd:
    """Concurrency probe: its sync blocks until released."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def is_available(self):
        return True

    def sync(self, *, force=False):
        self.calls += 1
        self.started.set()
        self.release.wait(5)


class CountingFts:
    def sync(self, *, force=False):
        return {"pages": 1}


def test_background_qmd_sync_does_not_block_and_never_overlaps():
    qmd = BlockingQmd()
    hybrid = HybridBackend(CountingFts(), qmd, background_qmd_sync=True)
    started = time.perf_counter()
    assert hybrid.sync() == {"pages": 1}
    assert hybrid.sync() == {"pages": 1}  # skipped: one qmd sync at a time
    assert time.perf_counter() - started < 1
    assert qmd.started.wait(2)
    qmd.release.set()
    hybrid._qmd_sync_thread.join(5)
    assert qmd.calls == 1
    hybrid.sync()
    hybrid._qmd_sync_thread.join(5)
    assert qmd.calls == 2


def test_foreground_qmd_sync_waits_for_qmd():
    qmd = BlockingQmd()
    qmd.release.set()
    HybridBackend(CountingFts(), qmd).sync()
    assert qmd.calls == 1


# -- real qmd server -------------------------------------------------------

def test_real_qmd_http_server_end_to_end(wiki, isolated_qmd, tmp_path):
    binary = qmd_binary()
    store = WikiStore(wiki, untrusted_dirs=[], exclude=["private/*"])
    port = free_port()
    qmd = QmdBackend(store, collection="wiki", binary=binary, mode="search",
                     index="hwiki-http-test", url=f"http://localhost:{port}", timeout=10)
    qmd.sync()
    server = subprocess.Popen(
        [binary, "--index", "hwiki-http-test", "mcp", "--http", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.time() + 20
        while True:
            try:
                opener.open(f"http://localhost:{port}/health", timeout=1).read()
                break
            except Exception:
                if time.time() > deadline:
                    pytest.fail("qmd http server did not start")
                time.sleep(0.2)

        hits = qmd.search("allowedwombat release", limit=4)
        assert [h.path for h in hits] == ["notes/release.md"]
        assert "allowedwombat release is tagged" in hits[0].snippet
        assert not hits[0].snippet.startswith(("@@", "5:"))
        assert qmd.search("privatenarwhal") == []

        # a note added later is found by the running server after a CLI sync
        (wiki / "notes" / "later.md").write_text("# Later\n\nThe latecomerotter arrived.\n",
                                                encoding="utf-8")
        qmd.sync()
        assert [h.path for h in qmd.search("latecomerotter")] == ["notes/later.md"]
    finally:
        server.terminate()
        server.wait(10)
