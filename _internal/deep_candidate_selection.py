"""Bounded, body-free candidate transfer contract for deep retrieval."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from typing import Callable


CANDIDATE_RESPONSE_SCHEMA_VERSION = 2
# Envelope for one worker call that ranks every viewpoint of a round.
CANDIDATE_MANY_RESPONSE_SCHEMA_VERSION = 1
CANDIDATE_TRANSFER_LIMIT_CODE = "candidate_transfer_limit"
CANDIDATE_TRANSFER_LIMIT_MESSAGE = (
    "検索候補の内部転送容量を超えたため、調査を継続できませんでした"
)
DEEP_MAX_CANDIDATES_PER_FILE = 8
DEEP_MAX_CANDIDATES_TOTAL = 40

_REFERENCE_FIELDS = (
    "path",
    "chunk_id",
    "start_line",
    "end_line",
    "source_sha256",
)
_SCORE_FIELDS = ("score", "keyword_score", "embedding_score", "rrf_score")
_REFERENCE_KEYS = frozenset((*_REFERENCE_FIELDS, *_SCORE_FIELDS))
_RESPONSE_KEYS = frozenset(
    {"schema_version", "paths", "source_hashes", "selected", "excluded", "counts"}
)


def normalize_candidate_path(value: object) -> str:
    """Use the same slash normalization as the search merge boundary."""
    return str(value or "").replace("\\", "/")


def _required_line(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"invalid candidate {field}")
    return value


def _finite_number(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid candidate {field}")
    if not math.isfinite(value):
        raise ValueError(f"invalid candidate {field}")
    return value


def project_candidate(
    candidate: Mapping[str, object], *, source_chunk: Mapping | None = None
) -> dict:
    """Project a ranked match without transferring body, snippet, or vectors."""
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be an object")
    path = normalize_candidate_path(candidate.get("path"))
    chunk_id = str(candidate.get("chunk_id") or "").replace("\\", "/")
    # The legacy parent loop accepted an empty path and later recorded it as
    # source_missing. Keep that behavior; hydrate_selected intentionally does
    # not attach body text to such a reference.
    if not chunk_id:
        raise ValueError("candidate chunk_id is required")
    start_line = _required_line(candidate.get("start_line"), "start_line")
    end_line = _required_line(candidate.get("end_line"), "end_line")
    if end_line < start_line:
        raise ValueError("candidate line range is reversed")
    source_sha256 = str(
        candidate.get("source_sha256")
        or candidate.get("file_sha256")
        or (source_chunk or {}).get("file_sha256")
        or ""
    )
    result = {
        "path": path,
        "chunk_id": chunk_id,
        "start_line": start_line,
        "end_line": end_line,
        "source_sha256": source_sha256,
    }
    for field in _SCORE_FIELDS:
        if field in candidate and candidate[field] is not None:
            result[field] = _finite_number(candidate[field], field)
    return result


def select_candidates(
    ranked_candidates: Iterable[Mapping[str, object]],
    *,
    source_chunks: Iterable[Mapping[str, object]] | None = None,
    max_per_file: int = DEEP_MAX_CANDIDATES_PER_FILE,
    max_total: int = DEEP_MAX_CANDIDATES_TOTAL,
    cancel_check: Callable[[], None] | None = None,
    deadline: float | None = None,
) -> dict:
    """Select in ranked order and retain every omitted candidate as a reference."""
    ranked = []
    for candidate in ranked_candidates:
        _check_control(cancel_check, deadline)
        ranked.append(candidate)
    by_chunk_id = {
        str(chunk.get("chunk_id") or "").replace("\\", "/"): chunk
        for chunk in (source_chunks or [])
        if chunk.get("chunk_id")
    }
    counts: dict[str, int] = {}
    path_indexes: dict[str, int] = {}
    paths: list[str] = []
    source_hash_indexes: dict[str, int] = {}
    source_hashes: list[str] = []

    def path_index(path: str) -> int:
        if path not in path_indexes:
            path_indexes[path] = len(paths)
            paths.append(path)
        return path_indexes[path]

    def source_hash_index(source_sha256: str) -> int:
        if source_sha256 not in source_hash_indexes:
            source_hash_indexes[source_sha256] = len(source_hashes)
            source_hashes.append(source_sha256)
        return source_hash_indexes[source_sha256]

    selected: list[dict] = []
    excluded: list[dict] = []
    for candidate in ranked:
        _check_control(cancel_check, deadline)
        path = normalize_candidate_path(candidate.get("path"))
        source_chunk = by_chunk_id.get(
            str(candidate.get("chunk_id") or "").replace("\\", "/")
        )
        reference = project_candidate(candidate, source_chunk=source_chunk)
        if counts.get(path, 0) >= max_per_file or len(selected) >= max_total:
            reference["reason"] = "candidate_limit"
            reference.pop("path")
            reference["path_index"] = path_index(path)
            source_sha256 = str(reference.pop("source_sha256") or "")
            reference["source_hash_index"] = source_hash_index(source_sha256)
            excluded.append(reference)
            continue
        counts[path] = counts.get(path, 0) + 1
        selected.append(reference)
    return {
        "schema_version": CANDIDATE_RESPONSE_SCHEMA_VERSION,
        "paths": paths,
        "source_hashes": source_hashes,
        "selected": selected,
        "excluded": excluded,
        "counts": {
            "ranked": len(ranked),
            "selected": len(selected),
            "excluded": len(excluded),
        },
    }


def strip_excluded_scores(response: dict) -> dict:
    """Drop ranking scores from excluded references (the parent only records them as unread).

    A batched response carries one excluded list per viewpoint; without scores
    each reference is roughly half the size, keeping a whole round well inside
    the worker output limit. The response still satisfies validate_response.
    """
    excluded = [
        {key: value for key, value in reference.items() if key not in _SCORE_FIELDS}
        for reference in response["excluded"]
    ]
    return {**response, "excluded": excluded}


def validate_many_response(value: object, query_count: int) -> list[dict]:
    """Validate a batched retrieve response: one schema-2 response per query, in order."""
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "responses"}
        or value.get("schema_version") != CANDIDATE_MANY_RESPONSE_SCHEMA_VERSION
        or not isinstance(value.get("responses"), list)
        or len(value["responses"]) != query_count
    ):
        raise ValueError("invalid retrieve_many response")
    return [validate_response(response) for response in value["responses"]]


def _check_control(
    cancel_check: Callable[[], None] | None, deadline: float | None
) -> None:
    if cancel_check is not None:
        cancel_check()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("candidate selection deadline")


def validate_response(value: object) -> dict:
    """Validate the versioned response and its count conservation contract."""
    if not isinstance(value, dict) or set(value) != _RESPONSE_KEYS:
        raise ValueError("invalid retrieve response shape")
    if value.get("schema_version") != CANDIDATE_RESPONSE_SCHEMA_VERSION:
        raise ValueError("unsupported retrieve response version")
    selected = value.get("selected")
    excluded = value.get("excluded")
    paths = value.get("paths")
    source_hashes = value.get("source_hashes")
    counts = value.get("counts")
    if (
        not isinstance(selected, list)
        or not isinstance(excluded, list)
        or not isinstance(paths, list)
        or not isinstance(source_hashes, list)
        or not isinstance(counts, dict)
    ):
        raise ValueError("invalid retrieve response collections")
    if any(not isinstance(path, str) for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("invalid retrieve response path table")
    if any(not isinstance(value, str) for value in source_hashes) or len(
        set(source_hashes)
    ) != len(source_hashes):
        raise ValueError("invalid retrieve response source hash table")
    if set(counts) != {"ranked", "selected", "excluded"}:
        raise ValueError("invalid retrieve response counts")
    for name in ("ranked", "selected", "excluded"):
        count = counts[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid retrieve response count")
    if counts["selected"] != len(selected) or counts["excluded"] != len(excluded):
        raise ValueError("retrieve response count mismatch")
    if counts["ranked"] != counts["selected"] + counts["excluded"]:
        raise ValueError("retrieve response ranked count mismatch")
    per_file: dict[str, int] = {}
    for candidate in selected:
        _validate_reference(candidate, excluded=False)
        path = candidate["path"]
        per_file[path] = per_file.get(path, 0) + 1
    if len(selected) > DEEP_MAX_CANDIDATES_TOTAL or any(
        count > DEEP_MAX_CANDIDATES_PER_FILE for count in per_file.values()
    ):
        raise ValueError("retrieve response exceeds candidate limits")
    for candidate in excluded:
        _validate_compact_excluded(candidate, paths, source_hashes)
    return value


def _validate_compact_excluded(
    candidate: object, paths: list[str], source_hashes: list[str]
) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("invalid excluded candidate reference")
    if "path" in candidate or "source_sha256" in candidate:
        raise ValueError("expanded excluded candidate is not a worker response")
    allowed = (_REFERENCE_KEYS - {"path", "source_sha256"}) | {
        "path_index",
        "source_hash_index",
        "reason",
    }
    if set(candidate) - allowed:
        raise ValueError("invalid compact excluded candidate fields")
    path_index = candidate.get("path_index")
    source_hash_index = candidate.get("source_hash_index")
    if (
        isinstance(path_index, bool)
        or not isinstance(path_index, int)
        or path_index < 0
        or path_index >= len(paths)
        or isinstance(source_hash_index, bool)
        or not isinstance(source_hash_index, int)
        or source_hash_index < 0
        or source_hash_index >= len(source_hashes)
    ):
        raise ValueError("invalid excluded candidate reference indexes")
    expanded = dict(candidate)
    expanded.pop("path_index")
    expanded.pop("source_hash_index")
    expanded["path"] = paths[path_index]
    expanded["source_sha256"] = source_hashes[source_hash_index]
    _validate_reference(expanded, excluded=True)


def expand_response(value: object) -> dict:
    """Expand the strict worker reference tables for parent-side processing."""
    validate_response(value)
    paths = value["paths"]
    source_hashes = value["source_hashes"]
    excluded = []
    for candidate in value["excluded"]:
        path_index = candidate["path_index"]
        source_hash_index = candidate["source_hash_index"]
        expanded = dict(candidate)
        expanded.pop("path_index")
        expanded.pop("source_hash_index")
        expanded["path"] = paths[path_index]
        expanded["source_sha256"] = source_hashes[source_hash_index]
        excluded.append(expanded)
    return {**value, "excluded": excluded}


def _validate_reference(candidate: object, *, excluded: bool) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("invalid candidate reference")
    allowed = _REFERENCE_KEYS | ({"reason"} if excluded else set())
    if set(candidate) - allowed or set(_REFERENCE_FIELDS) - set(candidate):
        raise ValueError("invalid candidate reference fields")
    project_candidate(candidate)
    if excluded and candidate.get("reason") != "candidate_limit":
        raise ValueError("invalid excluded candidate reason")


def hydrate_selected(response: dict, source_chunks: Iterable[Mapping[str, object]]) -> dict:
    """Restore body-bearing candidates from the parent-owned snapshot."""
    normalized = expand_response(response)
    by_chunk_id = {
        str(chunk.get("chunk_id") or "").replace("\\", "/"): chunk
        for chunk in source_chunks
        if chunk.get("chunk_id")
    }
    hydrated = []
    for reference in normalized["selected"]:
        if not reference["path"]:
            hydrated.append(dict(reference))
            continue
        chunk = by_chunk_id.get(reference["chunk_id"])
        if chunk is None:
            raise ValueError("selected candidate is absent from the parent snapshot")
        if normalize_candidate_path(chunk.get("path")) != reference["path"]:
            raise ValueError("selected candidate path does not match the parent snapshot")
        item = dict(chunk)
        item.update(reference)
        item["file_sha256"] = chunk.get("file_sha256") or reference["source_sha256"]
        item["source_sha256"] = reference["source_sha256"] or item["file_sha256"]
        hydrated.append(item)
    return {**normalized, "selected": hydrated}
