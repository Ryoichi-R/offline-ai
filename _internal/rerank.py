"""Optional retrieval reranking helpers."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable

from config import RerankConfig


LOCAL_RERANK_MAX_RESPONSE_BYTES = 1_048_576


class RerankTimeoutError(TimeoutError):
    """Raised when a rerank backend exceeds its configured deadline."""


def _check_timeout(deadline: float, timeout_s: int) -> None:
    if time.monotonic() >= deadline:
        raise RerankTimeoutError(f"rerank timed out after {timeout_s}s")


def _terms(text: str) -> list[str]:
    result = []
    for term in re.split(r"[\s,、。・/\\]+", text.lower()):
        cleaned = term.strip()
        if len(cleaned) >= 2:
            result.append(cleaned)
    return list(dict.fromkeys(result))


def _lexical_score(query: str, candidate: dict, plan: dict | None = None) -> float:
    snippet = str(candidate.get("snippet", "") or "").lower()
    heading = str(candidate.get("heading", "") or "").lower()
    path_stem = Path(str(candidate.get("path", "") or "")).stem.lower()
    haystack = " ".join([snippet, heading, path_stem])
    score = float(candidate.get("rrf_score", candidate.get("score", 0)) or 0)
    query_terms = _terms(query)
    must_terms = [str(x).lower() for x in (plan or {}).get("must_find_terms", []) if str(x).strip()]
    for term in list(dict.fromkeys(query_terms + must_terms)):
        if term in haystack:
            score += 1.0
        if term in heading:
            score += 2.0
        if term in path_stem:
            score += 0.75
    layout_type = (candidate.get("layout_type") or candidate.get("metadata", {}).get("layout_type") or "").lower()
    if layout_type in {"title", "table"}:
        score += 0.25
    return score


def _local_http_rerank(
    query: str,
    candidates: list[dict],
    *,
    config: RerankConfig,
    cancel_check: Callable[[], None] | None = None,
) -> list[dict]:
    if not config.endpoint:
        raise RuntimeError("local reranker endpoint is not configured")
    if cancel_check:
        cancel_check()
    payload = {
        "model": config.model,
        "query": query,
        "top_n": min(config.keep_n, len(candidates)),
        "documents": [str(candidate.get("snippet") or candidate.get("text") or "") for candidate in candidates],
    }
    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            response_body = response.read(LOCAL_RERANK_MAX_RESPONSE_BYTES + 1)
            if len(response_body) > LOCAL_RERANK_MAX_RESPONSE_BYTES:
                raise RuntimeError("local reranker response exceeded the size limit")
            result = json.loads(response_body.decode("utf-8"))
    except TimeoutError as exc:
        raise RerankTimeoutError(f"rerank timed out after {config.timeout_s}s") from exc
    if cancel_check:
        cancel_check()

    rows = result.get("results") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("local reranker response must contain results[]")
    reranked = []
    seen_indices = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("local reranker result must be an object")
        index = row.get("index")
        score = row.get("relevance_score", row.get("score"))
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(candidates)
            or index in seen_indices
        ):
            raise RuntimeError("local reranker returned an invalid or duplicate index")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise RuntimeError("local reranker returned a non-finite numeric score")
        seen_indices.add(index)
        if config.min_score is not None and float(score) < config.min_score:
            continue
        item = dict(candidates[index])
        item["rerank_score"] = float(score)
        source = str(item.get("source", ""))
        item["source"] = "+".join(sorted(set(source.split("+")) | {"rerank-local"})).strip("+")
        reranked.append(item)
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[: config.keep_n]


def rerank_candidates(
    query: str,
    candidates: list[dict],
    *,
    plan: dict | None = None,
    config: RerankConfig,
    cancel_check: Callable[[], None] | None = None,
) -> list[dict]:
    """Rerank retrieval candidates.

    `off` returns the input unchanged. `lexical` is deterministic and has no
    external dependency. `local` calls a loopback-only Jina/llama.cpp-compatible
    rerank endpoint configured in `RerankConfig`.
    """
    if config.mode == "off":
        return candidates
    if config.mode == "local":
        return _local_http_rerank(
            query,
            candidates,
            config=config,
            cancel_check=cancel_check,
        )
    if config.mode != "lexical":
        return candidates

    deadline = time.monotonic() + config.timeout_s
    reranked = []
    for rank, candidate in enumerate(candidates):
        if cancel_check:
            cancel_check()
        _check_timeout(deadline, config.timeout_s)
        item = dict(candidate)
        item["rerank_score"] = round(_lexical_score(query, item, plan) + (1.0 / (rank + 100)), 6)
        _check_timeout(deadline, config.timeout_s)
        source = str(item.get("source", ""))
        item["source"] = "+".join(sorted(set(source.split("+")) | {"rerank"})).strip("+")
        reranked.append(item)
    _check_timeout(deadline, config.timeout_s)
    if config.min_score is not None:
        reranked = [item for item in reranked if float(item.get("rerank_score", 0)) >= config.min_score]
    reranked.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
    return reranked[: config.keep_n]
