"""Per-request disposable worker. No files, logs, or shared-service termination."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError

WORKER_OUTPUT_MAX_BYTES = 32_000_000
CANDIDATE_TRANSFER_LIMIT_CODE = "candidate_transfer_limit"
MODEL_OUTPUT_INCOMPLETE_CODE = "model_output_incomplete"
WORKER_ERROR_CODES = frozenset({
    CANDIDATE_TRANSFER_LIMIT_CODE, MODEL_OUTPUT_INCOMPLETE_CODE,
    "model_http_error", "model_stream_error", "model_stream_incomplete", "timeout", "worker_error",
})
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


def run_worker(payload: dict, *, timeout: float, cancel_check=None, touch=None):
    """Cancel even while connecting or waiting for HTTP headers; reap before return."""
    if cancel_check:
        cancel_check()
    if timeout <= 0:
        raise TimeoutError("deep worker deadline")
    deadline = time.monotonic() + timeout
    worker_payload = dict(payload)
    if touch is not None:
        worker_payload['_worker_progress'] = True
    worker_payload["_worker_deadline_monotonic"] = deadline
    encoded = json.dumps(worker_payload, ensure_ascii=False).encode("utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if touch is not None else subprocess.DEVNULL,
        creationflags=flags,
    )
    finished = threading.Event()
    progress = threading.Event()
    progress_stream = process.stderr if touch is not None else None
    # communicate owns stdin/stdout; the progress reader alone owns stderr.
    if progress_stream is not None:
        process.stderr = None
    response = {}

    def receive_progress():
        try:
            while progress_stream.read(1):
                progress.set()
        finally:
            progress_stream.close()

    progress_thread = (threading.Thread(target=receive_progress, name="chat-worker-progress", daemon=True)
                       if progress_stream is not None else None)

    def communicate():
        try:
            response["output"], _ = process.communicate(input=encoded)
        except Exception as exc:
            response["error"] = exc
        finally:
            finished.set()

    # One uninterrupted communicate call must own stdin. Retrying with input=None
    # after a short timeout can leave a partially written pipe waiting forever.
    io_thread = threading.Thread(target=communicate, name="deep-worker-io", daemon=True)
    started = False
    try:
        io_thread.start()
        started = True
        if progress_thread is not None:
            progress_thread.start()
        while True:
            if touch is not None and progress.is_set():
                progress.clear()
                touch()
            if cancel_check:
                cancel_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("deep worker deadline")
            if finished.wait(timeout=min(0.05, remaining)):
                break
        if "error" in response:
            raise DeepWorkerError() from response["error"]
        output = response["output"]
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
        # Killing our disposable worker unblocks pipe I/O. Join its sole owner
        # before returning, including on cancellation and deadline expiry.
        if started:
            io_thread.join()
        else:
            process.communicate()
        if progress_thread is not None and progress_thread.ident is not None:
            progress_thread.join()
        elif progress_stream is not None:
            progress_stream.close()



def _load_worker_embedding_context(payload: dict, chunks: list[dict], search_module):
    """Resolve retrieval embeddings, with an explicit no-production-cache mode."""
    if "isolated_embedding" not in payload:
        model = search_module.detect_embed_model()
        cache = search_module.load_embed_cache() if model else {}
        if model and not search_module._cache_compatibility_matches(
            cache, model, search_module.get_embed_model_identity(model)
        ):
            return None, {}
        if model:
            cache = dict(cache)
            cache["entries"] = {
                chunk["chunk_id"]: cache.get("entries", {})[chunk["chunk_id"]]
                for chunk in chunks
                if search_module._entry_matches_chunk(
                    cache.get("entries", {}).get(chunk["chunk_id"]), chunk
                )
            }
        return model, cache

    context = payload.get("isolated_embedding")
    if not isinstance(context, dict) or set(context) != {"model", "cache_path", "cache_sha256"}:
        raise ValueError("invalid isolated embedding context")
    model = context["model"]
    cache_path = context["cache_path"]
    expected_digest = context["cache_sha256"]
    if model is None:
        if cache_path is not None or expected_digest is not None:
            raise ValueError("disabled isolated embedding context contains a cache")
        return None, {}
    if not isinstance(model, str) or not model.strip():
        raise ValueError("isolated embedding model is invalid")
    if not isinstance(cache_path, str) or not Path(cache_path).is_absolute():
        raise ValueError("isolated embedding cache path must be absolute")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("isolated embedding cache digest is invalid")
    path = Path(cache_path)
    if not path.is_file():
        raise ValueError("isolated embedding cache is unavailable")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    if hasher.hexdigest() != expected_digest:
        raise ValueError("isolated embedding cache changed")
    cache = search_module._load_embed_cache_file(path)
    identity = search_module.get_embed_model_identity(model)
    if not search_module._cache_compatibility_matches(cache, model, identity):
        raise ValueError("isolated embedding cache compatibility failed")
    entries = cache.get("entries")
    expected_ids = {str(chunk.get("chunk_id", "")) for chunk in chunks}
    if not isinstance(entries, dict) or set(entries) != expected_ids:
        raise ValueError("isolated embedding cache does not match the evaluation snapshot")
    for chunk in chunks:
        if not search_module._entry_matches_chunk(entries.get(chunk["chunk_id"]), chunk):
            raise ValueError("isolated embedding entry does not match the evaluation snapshot")
    return model, cache

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
        model, cache = _load_worker_embedding_context(payload, chunks, search)
        limit = max(1, len(chunks))
        # Answer policy v2: a validated, body-free intent lets the worker assign
        # candidate roles deterministically (no model call). Absent for legacy.
        role_of = None
        if "intent" in payload:
            from answer_policy import AnswerIntent, classify_role

            intent = AnswerIntent.from_dict(payload["intent"])
            role_cache: dict[str, str] = {}

            def role_of(chunk):
                key = str(chunk.get("chunk_id") or id(chunk))
                if key not in role_cache:
                    role_cache[key] = classify_role(
                        str(chunk.get("text") or ""),
                        str(chunk.get("heading") or ""),
                        intent=intent,
                    ).role
                return role_cache[key]

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
                role_of=role_of,
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
    import urllib.error
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
    try:
        response = urllib.request.urlopen(request, timeout=payload["timeout"])
    except urllib.error.HTTPError as exc:
        if not payload.get("retry_without_think") or "think" not in payload["body"] or exc.code < 400:
            raise
        body = dict(payload["body"])
        body.pop("think")
        request = urllib.request.Request(
            host + "/api/chat", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        response = urllib.request.urlopen(request, timeout=payload["timeout"])
    with response:
        for raw in response:
            event = json.loads(raw)
            if payload.get("_worker_progress"):
                # Only an activity marker. Never transfer/log thinking text.
                sys.stderr.buffer.write(b".")
                sys.stderr.buffer.flush()
            if event.get("error"):
                raise DeepWorkerError(code="model_stream_error")
            content = event.get("message", {}).get("content", "")
            size += len(content)
            if size > 65536:
                raise ValueError("model output limit")
            parts.append(content)
            if event.get("done") is True:
                if event.get("done_reason") == "length":
                    raise DeepWorkerError(code=MODEL_OUTPUT_INCOMPLETE_CODE)
                finished = True
                break
    if not finished:
        raise DeepWorkerError(code="model_stream_incomplete")
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
    except DeepWorkerError as exc:
        sys.stdout.buffer.write(_error_payload(exc.code))
    except TimeoutError:
        sys.stdout.buffer.write(_error_payload("timeout"))
    except HTTPError:
        sys.stdout.buffer.write(_error_payload("model_http_error"))
    except Exception:
        sys.stdout.buffer.write(_error_payload("worker_error"))
