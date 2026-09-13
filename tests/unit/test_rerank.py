import os
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ.pop("OLLAMA_HOST", None)

import rerank  # noqa: E402
import search  # noqa: E402
from config import RerankConfig  # noqa: E402
from rerank import RerankTimeoutError, rerank_candidates  # noqa: E402


def test_rerank_off_returns_input_order():
    candidates = [{"path": "a.md"}, {"path": "b.md"}]
    config = RerankConfig(mode="off", model="", top_n=24, keep_n=8, timeout_s=15, strict=False)

    assert rerank_candidates("query", candidates, config=config) == candidates


def test_lexical_rerank_promotes_heading_match():
    candidates = [
        {
            "path": "a.md",
            "heading": "雑則",
            "snippet": "一般",
            "rrf_score": 1.0,
            "source": "keyword",
        },
        {
            "path": "b.md",
            "heading": "申請手順",
            "snippet": "提出",
            "rrf_score": 0.1,
            "source": "embedding",
        },
    ]
    config = RerankConfig(mode="lexical", model="", top_n=24, keep_n=1, timeout_s=15, strict=False)

    reranked = rerank_candidates("申請手順", candidates, config=config)

    assert reranked[0]["path"] == "b.md"
    assert "rerank" in reranked[0]["source"]


def test_local_rerank_backend_reports_unconfigured_adapter():
    candidates = [{"path": "a.md"}, {"path": "b.md"}]
    config = RerankConfig(mode="local", model="m", top_n=24, keep_n=1, timeout_s=15, strict=False)

    with pytest.raises(RuntimeError, match="not configured"):
        rerank_candidates("query", candidates, config=config)


def test_local_rerank_calls_loopback_endpoint_and_restores_candidates(monkeypatch):
    candidates = [
        {"path": "a.md", "snippet": "一般", "source": "keyword"},
        {"path": "b.md", "snippet": "申請期限は3日前", "source": "embedding"},
    ]
    config = RerankConfig(
        mode="local",
        model="bge-reranker-v2-m3",
        top_n=24,
        keep_n=2,
        timeout_s=15,
        strict=False,
        endpoint="http://127.0.0.1:8012/v1/rerank",
    )
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            return json.dumps(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.95},
                        {"index": 0, "relevance_score": 0.10},
                    ]
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(rerank.urllib.request, "urlopen", fake_urlopen)

    result = rerank_candidates("申請期限", candidates, config=config)

    assert [item["path"] for item in result] == ["b.md", "a.md"]
    assert result[0]["rerank_score"] == 0.95
    assert "rerank-local" in result[0]["source"]
    assert captured["url"] == config.endpoint
    assert captured["body"]["documents"] == ["一般", "申請期限は3日前"]
    assert captured["timeout"] == 15


def test_local_rerank_rejects_duplicate_indices(monkeypatch):
    config = RerankConfig(
        mode="local",
        model="m",
        top_n=24,
        keep_n=2,
        timeout_s=15,
        strict=False,
        endpoint="http://127.0.0.1:8012/rerank",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            return b'{"results":[{"index":0,"score":1},{"index":0,"score":0.5}]}'

    monkeypatch.setattr(rerank.urllib.request, "urlopen", lambda request, timeout: FakeResponse())

    with pytest.raises(RuntimeError, match="duplicate index"):
        rerank_candidates("q", [{"snippet": "a"}], config=config)


def test_local_rerank_rejects_boolean_index(monkeypatch):
    config = RerankConfig(
        mode="local",
        model="m",
        top_n=24,
        keep_n=1,
        timeout_s=15,
        strict=True,
        endpoint="http://127.0.0.1:8012/rerank",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            return b'{"results":[{"index":true,"score":1}]}'

    monkeypatch.setattr(rerank.urllib.request, "urlopen", lambda request, timeout: FakeResponse())

    with pytest.raises(RuntimeError, match="invalid"):
        rerank_candidates("q", [{"snippet": "a"}], config=config)


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), float("-inf")])
def test_local_rerank_rejects_non_finite_or_boolean_score(monkeypatch, score):
    config = RerankConfig(
        mode="local",
        model="m",
        top_n=24,
        keep_n=1,
        timeout_s=15,
        strict=True,
        endpoint="http://127.0.0.1:8012/rerank",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            return json.dumps({"results": [{"index": 0, "score": score}]}).encode()

    monkeypatch.setattr(rerank.urllib.request, "urlopen", lambda request, timeout: FakeResponse())

    with pytest.raises(RuntimeError, match="non-finite"):
        rerank_candidates("q", [{"snippet": "a"}], config=config)


def test_local_rerank_oversized_response_uses_non_strict_fallback(monkeypatch, caplog):
    candidates = [{"path": "a.md", "snippet": "a"}]
    config = RerankConfig(
        mode="local",
        model="m",
        top_n=24,
        keep_n=1,
        timeout_s=15,
        strict=False,
        endpoint="http://127.0.0.1:8012/rerank",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            return b"x" * size

    monkeypatch.setattr(rerank.urllib.request, "urlopen", lambda request, timeout: FakeResponse())

    result = search._rerank_with_fallback("q", candidates, plan={}, config=config)

    assert result is candidates
    assert "size limit" in caplog.text


def test_local_rerank_loopback_http_contract_end_to_end():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length).decode()))
            response = json.dumps(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.8},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = RerankConfig(
            mode="local",
            model="test-model",
            top_n=24,
            keep_n=2,
            timeout_s=5,
            strict=True,
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/rerank",
        )

        result = rerank_candidates(
            "申請期限",
            [{"path": "a", "snippet": "一般"}, {"path": "b", "snippet": "期限"}],
            config=config,
        )

        assert [item["path"] for item in result] == ["b", "a"]
        assert requests[0]["model"] == "test-model"
        assert requests[0]["query"] == "申請期限"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_lexical_rerank_applies_min_score():
    candidates = [
        {
            "path": "a.md",
            "heading": "関係なし",
            "snippet": "",
            "rrf_score": 0.0,
            "source": "keyword",
        }
    ]
    config = RerankConfig(
        mode="lexical", model="", top_n=24, keep_n=8, timeout_s=15, strict=False, min_score=10.0
    )

    assert rerank_candidates("申請手順", candidates, config=config) == []


def test_lexical_rerank_calls_cancel_check():
    calls = {"n": 0}

    def cancel_check():
        calls["n"] += 1

    config = RerankConfig(mode="lexical", model="", top_n=24, keep_n=8, timeout_s=15, strict=False)
    rerank_candidates("query", [{"path": "a.md"}], config=config, cancel_check=cancel_check)

    assert calls["n"] == 1


def test_lexical_rerank_enforces_timeout(monkeypatch):
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(rerank.time, "monotonic", lambda: next(ticks))
    config = RerankConfig(mode="lexical", model="", top_n=24, keep_n=8, timeout_s=1, strict=False)

    with pytest.raises(RerankTimeoutError, match="timed out"):
        rerank_candidates("query", [{"path": "a.md"}], config=config)


def test_rerank_failure_falls_back_to_original_rrf_order(monkeypatch, caplog):
    candidates = [{"path": "a.md"}, {"path": "b.md"}]
    config = RerankConfig(mode="lexical", model="", top_n=24, keep_n=1, timeout_s=1, strict=False)

    def fail_rerank(*args, **kwargs):
        raise RerankTimeoutError("timeout")

    monkeypatch.setattr(search, "rerank_candidates", fail_rerank)
    result = search._rerank_with_fallback("query", candidates, plan={}, config=config)

    assert result is candidates
    assert result == [{"path": "a.md"}, {"path": "b.md"}]
    assert "using RRF order" in caplog.text


def test_rerank_failure_is_strict_when_requested(monkeypatch):
    config = RerankConfig(mode="lexical", model="", top_n=24, keep_n=8, timeout_s=1, strict=True)

    def fail_rerank(*args, **kwargs):
        raise RerankTimeoutError("timeout")

    monkeypatch.setattr(search, "rerank_candidates", fail_rerank)
    with pytest.raises(RerankTimeoutError):
        search._rerank_with_fallback("query", [{"path": "a.md"}], plan={}, config=config)


def test_rerank_does_not_swallow_cancellation():
    class TestCancelled(Exception):
        pass

    def cancel_check():
        raise TestCancelled("cancelled")

    config = RerankConfig(mode="lexical", model="", top_n=24, keep_n=8, timeout_s=15, strict=False)
    with pytest.raises(TestCancelled, match="cancelled"):
        search._rerank_with_fallback(
            "query",
            [{"path": "a.md"}],
            plan={},
            config=config,
            cancel_check=cancel_check,
        )
