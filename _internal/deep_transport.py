"""Per-request disposable worker. No files, logs, or shared-service termination."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


class DeepWorkerError(RuntimeError):
    pass


def run_worker(payload: dict, *, timeout: float, cancel_check=None):
    """Cancel even while connecting or waiting for HTTP headers; reap before return."""
    if cancel_check:
        cancel_check()
    if timeout <= 0:
        raise TimeoutError("deep worker deadline")
    deadline = time.monotonic() + timeout
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    try:
        while True:
            if cancel_check:
                cancel_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("deep worker deadline")
            try:
                output, _ = process.communicate(input=encoded, timeout=min(0.05, remaining))
                break
            except subprocess.TimeoutExpired:
                encoded = None
        if cancel_check:
            cancel_check()
        if time.monotonic() >= deadline:
            raise TimeoutError("deep worker deadline")
        if process.returncode != 0 or len(output) > 32_000_000:
            raise DeepWorkerError("deep worker failed")
        result = json.loads(output)
        if result.get("ok") is not True:
            raise DeepWorkerError("deep worker failed")
        return result["value"]
    finally:
        if process.poll() is None:
            process.kill()
        # communicate also joins the Windows pipe reader/writer threads.
        process.communicate()


def _execute(payload):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if payload["operation"] == "retrieve":
        import search

        chunks = payload["chunks"]
        model = search.detect_embed_model()
        cache = search.load_embed_cache() if model else {}
        if model and not search._cache_compatibility_matches(
            cache, model, search.get_embed_model_identity(model)
        ):
            model = None
        if model:
            # Stale/missing entries are never regenerated in a search worker.
            cache = dict(cache)
            cache["entries"] = {
                c["chunk_id"]: cache["entries"][c["chunk_id"]]
                for c in chunks
                if search._entry_matches_chunk(cache.get("entries", {}).get(c["chunk_id"]), c)
            }
        limit = max(1, len(chunks))
        keyword = search.keyword_search_chunks(
            payload["query"],
            search._split_terms(payload["query"]),
            top_k=limit,
            chunks=chunks,
        )
        embedding = (
            search.embedding_search(payload["query"], model, cache, top_k=limit) if model else []
        )
        return search.merge_results(keyword, embedding, max_results=limit)
    if payload["operation"] != "chat":
        raise ValueError("unsupported operation")
    import urllib.request
    from config import validate_ollama_host

    host = validate_ollama_host(payload["host"])
    request = urllib.request.Request(
        host + "/api/chat",
        data=json.dumps(payload["body"]).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts = []
    size = 0
    finished = False
    with urllib.request.urlopen(request, timeout=payload["timeout"]) as response:
        for raw in response:
            event = json.loads(raw)
            if event.get("error"):
                raise ValueError("model error")
            content = event.get("message", {}).get("content", "")
            size += len(content)
            if size > 65536:
                raise ValueError("model output limit")
            parts.append(content)
            if event.get("done") is True:
                if event.get("done_reason") == "length":
                    raise ValueError("model output incomplete")
                finished = True
                break
    if not finished:
        raise ValueError("incomplete stream")
    return "".join(parts).strip()


if __name__ == "__main__":
    try:
        value = _execute(json.loads(sys.stdin.buffer.read()))
        encoded = json.dumps({"ok": True, "value": value}, ensure_ascii=False).encode("utf-8")
        if len(encoded) > 32_000_000:
            raise ValueError("worker output limit")
        sys.stdout.buffer.write(encoded)
    except Exception:
        sys.stdout.buffer.write(b'{"ok":false}')
