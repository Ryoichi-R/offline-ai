"""Per-request disposable worker. No files, logs, or shared-service termination."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

WORKER_OUTPUT_MAX_BYTES = 32_000_000
CANDIDATE_TRANSFER_LIMIT_CODE = "candidate_transfer_limit"
WORKER_ERROR_CODES = frozenset({CANDIDATE_TRANSFER_LIMIT_CODE, "worker_error"})
# Operations whose oversize output is a candidate-transfer limit, not a generic failure.
RETRIEVE_OPERATIONS = frozenset({"retrieve", "retrieve_many"})


class CandidateTransferLimitError(RuntimeError):
    code = CANDIDATE_TRANSFER_LIMIT_CODE

class DeepWorkerError(RuntimeError):
    def __init__(self, message: str = "deep worker failed", *, code: str = "worker_error"):
        super().__init__(message)
        self.code = code


def _error_payload(code: str) -> bytes:
    if code not in WORKER_ERROR_CODES:
        code = "worker_error"
    return json.dumps({"ok": False, "code": code}, separators=(",", ":")).encode("ascii")


def _encode_success(payload: dict, value: object, *, max_bytes: int = WORKER_OUTPUT_MAX_BYTES) -> bytes:
    encoded = json.dumps(
        {"ok": True, "value": value},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        if payload.get("operation") in RETRIEVE_OPERATIONS:
            raise CandidateTransferLimitError("candidate transfer limit")
        raise ValueError("worker output limit")
    return encoded


def run_worker(payload: dict, *, timeout: float, cancel_check=None):
    """Cancel even while connecting or waiting for HTTP headers; reap before return."""
    if cancel_check:
        cancel_check()
    if timeout <= 0:
        raise TimeoutError("deep worker deadline")
    deadline = time.monotonic() + timeout
    worker_payload = dict(payload)
    worker_payload["_worker_deadline_monotonic"] = deadline
    encoded = json.dumps(worker_payload, ensure_ascii=False).encode("utf-8")
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
        if process.returncode != 0:
            raise DeepWorkerError()
        if len(output) > WORKER_OUTPUT_MAX_BYTES:
            code = (
                CANDIDATE_TRANSFER_LIMIT_CODE
                if payload.get("operation") in RETRIEVE_OPERATIONS
                else "worker_error"
            )
            raise DeepWorkerError(code=code)
        try:
            result = json.loads(output)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DeepWorkerError() from exc
        if not isinstance(result, dict) or result.get("ok") is not True:
            code = result.get("code") if isinstance(result, dict) else "worker_error"
            if code not in WORKER_ERROR_CODES:
                code = "worker_error"
            raise DeepWorkerError(code=code)
        if "value" not in result:
            raise DeepWorkerError()
        return result["value"]
    finally:
        if process.poll() is None:
            process.kill()
        # communicate also joins the Windows pipe reader/writer threads.
        process.communicate()


def _execute(payload):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if payload["operation"] in RETRIEVE_OPERATIONS:
        from deep_candidate_selection import (
            CANDIDATE_MANY_RESPONSE_SCHEMA_VERSION,
            select_candidates,
            strip_excluded_scores,
        )
        import search

        if payload["operation"] == "retrieve_many":
            queries = payload.get("queries")
            if (
                not isinstance(queries, list)
                or not queries
                or not all(isinstance(query, str) for query in queries)
            ):
                raise ValueError("retrieve_many requires queries")

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

        def rank(query):
            keyword = search.keyword_search_chunks(
                query,
                search._split_terms(query),
                top_k=limit,
                chunks=chunks,
            )
            embedding = (
                search.embedding_search(query, model, cache, top_k=limit) if model else []
            )
            ranked = search.merge_results(keyword, embedding, max_results=limit)
            return select_candidates(
                ranked,
                source_chunks=chunks,
                deadline=payload.get("_worker_deadline_monotonic"),
            )

        if payload["operation"] == "retrieve":
            return rank(payload["query"])
        # The embedding cache load and chunk transfer happen once for the
        # whole round instead of once per viewpoint.
        return {
            "schema_version": CANDIDATE_MANY_RESPONSE_SCHEMA_VERSION,
            "responses": [strip_excluded_scores(rank(query)) for query in queries],
        }
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
    payload = None
    try:
        payload = json.loads(sys.stdin.buffer.read())
        value = _execute(payload)
        encoded = _encode_success(payload, value)
        sys.stdout.buffer.write(encoded)
    except CandidateTransferLimitError:
        sys.stdout.buffer.write(_error_payload(CANDIDATE_TRANSFER_LIMIT_CODE))
    except Exception:
        sys.stdout.buffer.write(_error_payload("worker_error"))
