"""Small shared configuration helpers for offline-ai Python entrypoints."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from urllib.parse import urlparse


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_ALLOWED_OLLAMA_HOSTS = {"localhost", "127.0.0.1", "::1"}
_ALLOWED_RERANK_PATHS = {"/rerank", "/reranking", "/v1/rerank", "/v1/reranking"}


def validate_ollama_host(value: str) -> str:
    """Validate and normalize the loopback-only Ollama API base URL."""
    normalized = value.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"OLLAMA_HOST: 許可されないスキーム: {parsed.scheme}")
    hostname = parsed.hostname or ""
    if hostname not in _ALLOWED_OLLAMA_HOSTS:
        raise ValueError(f"OLLAMA_HOST: 許可されないホスト: {hostname}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OLLAMA_HOST: ユーザー情報をURLへ含めることはできません")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("OLLAMA_HOST: パス・クエリ・フラグメントは指定できません")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"OLLAMA_HOST: ポート番号が不正です: {exc}") from exc
    return normalized.rstrip("/")


def get_ollama_host() -> str:
    """Return the shared, validated Ollama API base URL."""
    return validate_ollama_host(os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))


OLLAMA_HOST = get_ollama_host()


def validate_rerank_endpoint(value: str) -> str:
    """Validate a loopback-only llama.cpp/Jina-compatible rerank endpoint."""
    normalized = value.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"OFFLINE_AI_RERANK_ENDPOINT: 許可されないスキーム: {parsed.scheme}")
    hostname = parsed.hostname or ""
    if hostname not in _ALLOWED_OLLAMA_HOSTS:
        raise ValueError(f"OFFLINE_AI_RERANK_ENDPOINT: 許可されないホスト: {hostname}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OFFLINE_AI_RERANK_ENDPOINT: ユーザー情報をURLへ含めることはできません")
    if parsed.path not in _ALLOWED_RERANK_PATHS or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("OFFLINE_AI_RERANK_ENDPOINT: rerank endpointのpathだけを指定してください")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"OFFLINE_AI_RERANK_ENDPOINT: ポート番号が不正です: {exc}") from exc
    return normalized.rstrip("/")


def env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Read an integer environment variable with validation and fallback."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        print(
            f"[WARN] {name} must be an integer; using default {default}.",
            file=sys.stderr,
        )
        return default
    if min_value is not None and value < min_value:
        print(
            f"[WARN] {name} must be >= {min_value}; using default {default}.",
            file=sys.stderr,
        )
        return default
    if max_value is not None and value > max_value:
        print(
            f"[WARN] {name} must be <= {max_value}; using default {default}.",
            file=sys.stderr,
        )
        return default
    return value


def env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read a float environment variable with validation and fallback."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        print(
            f"[WARN] {name} must be a number; using default {default}.",
            file=sys.stderr,
        )
        return default
    if value != value or value in (float("inf"), float("-inf")):
        print(
            f"[WARN] {name} must be finite; using default {default}.",
            file=sys.stderr,
        )
        return default
    if min_value is not None and value < min_value:
        print(
            f"[WARN] {name} must be >= {min_value}; using default {default}.",
            file=sys.stderr,
        )
        return default
    if max_value is not None and value > max_value:
        print(
            f"[WARN] {name} must be <= {max_value}; using default {default}.",
            file=sys.stderr,
        )
        return default
    return value


def validate_timeout(
    name: str,
    value: float | int,
    *,
    min_value: float = 0.001,
    max_value: float = 86400.0,
) -> float:
    """Validate a finite, bounded timeout and return it as seconds.

    Timeout settings are safety boundaries rather than tuning hints.  Invalid
    values therefore fail fast instead of silently turning an accidental
    zero, negative value, or very large value into an unbounded worker.
    """
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: timeout must be numeric") from exc
    if not normalized == normalized or normalized in (float("inf"), float("-inf")):
        raise ValueError(f"{name}: timeout must be finite")
    if normalized < min_value or normalized > max_value:
        raise ValueError(
            f"{name}: timeout must be between {min_value:g} and {max_value:g} seconds"
        )
    return normalized


def env_timeout(
    name: str,
    default: float,
    *,
    min_value: float = 0.001,
    max_value: float = 86400.0,
) -> float:
    """Read a bounded timeout environment variable, failing fast on errors."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return validate_timeout(
            name, default, min_value=min_value, max_value=max_value
        )
    return validate_timeout(
        name, raw.strip(), min_value=min_value, max_value=max_value
    )


# 検索単位の全体timeout契約（Web UI入力 / `timeout_seconds` API パラメータ /
# `--search-timeout` / `OFFLINEAI_SEARCH_TIMEOUT` で共有する）。
# 300〜600秒の整数のみを受理し、丸め・clampは行わずfail-fastする。
# 下限が300秒なのは、num_ctx=32768 の推論あり生成が中央値100.3秒・
# 最大260.8秒（2026-09-08実測）であり、それより短い上限では実用にならないため。
SEARCH_TIMEOUT_MIN = 300
SEARCH_TIMEOUT_MAX = 600
SEARCH_TIMEOUT_DEFAULT = 300

_INTEGER_RE = re.compile(r"^[+-]?\d+$")


def validate_search_timeout_seconds(value: object) -> int:
    """Validate a per-search overall timeout: an integer in [300, 600] seconds.

    Rejects (rather than rounding or clamping) booleans, floats, non-numeric
    strings, and out-of-range integers so an accidental "300.5" or "1200"
    never silently becomes a different fail-safe budget.
    """
    if isinstance(value, bool):
        raise ValueError("search timeout must be an integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not _INTEGER_RE.match(stripped):
            raise ValueError("search timeout must be an integer")
        normalized = int(stripped)
    else:
        raise ValueError("search timeout must be an integer")
    if normalized < SEARCH_TIMEOUT_MIN or normalized > SEARCH_TIMEOUT_MAX:
        raise ValueError(
            f"search timeout must be between {SEARCH_TIMEOUT_MIN} and "
            f"{SEARCH_TIMEOUT_MAX} seconds"
        )
    return normalized


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with conservative parsing."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    print(
        f"[WARN] {name} must be boolean; using default {default}.",
        file=sys.stderr,
    )
    return default


def env_choice(name: str, default: str, choices: set[str]) -> str:
    """Read an enum-like environment variable with validation and fallback."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in choices:
        return value
    print(
        f"[WARN] {name} must be one of {sorted(choices)}; using default {default}.",
        file=sys.stderr,
    )
    return default


@dataclass(frozen=True)
class RerankConfig:
    mode: str
    model: str
    top_n: int
    keep_n: int
    timeout_s: int
    strict: bool
    endpoint: str = ""
    min_score: float | None = None


def get_rerank_config() -> RerankConfig:
    """Return optional retrieval reranker settings shared by CLI and Web."""
    mode = env_choice("OFFLINE_AI_RERANK_MODE", "off", {"off", "lexical", "local"})
    model = os.environ.get("OFFLINE_AI_RERANK_MODEL", "").strip()
    endpoint_raw = os.environ.get("OFFLINE_AI_RERANK_ENDPOINT", "").strip()
    endpoint = validate_rerank_endpoint(endpoint_raw) if endpoint_raw else ""
    strict = env_bool("OFFLINE_AI_RERANK_STRICT", False)
    if mode == "local" and (not model or not endpoint):
        missing = []
        if not model:
            missing.append("OFFLINE_AI_RERANK_MODEL")
        if not endpoint:
            missing.append("OFFLINE_AI_RERANK_ENDPOINT")
        message = f"{', '.join(missing)} is required when OFFLINE_AI_RERANK_MODE=local"
        if strict:
            raise ValueError(message)
        print(f"[WARN] {message}; rerank mode falls back to off.", file=sys.stderr)
        mode = "off"
    min_score_raw = os.environ.get("OFFLINE_AI_RERANK_MIN_SCORE", "").strip()
    min_score: float | None = None
    if min_score_raw:
        try:
            min_score = float(min_score_raw)
        except ValueError:
            print("[WARN] OFFLINE_AI_RERANK_MIN_SCORE must be numeric; ignoring.", file=sys.stderr)
    return RerankConfig(
        mode=mode,
        model=model,
        top_n=env_int("OFFLINE_AI_RERANK_TOP_N", 24, min_value=1),
        keep_n=env_int("OFFLINE_AI_RERANK_KEEP_N", 8, min_value=1),
        timeout_s=env_int("OFFLINE_AI_RERANK_TIMEOUT_S", 15, min_value=1),
        strict=strict,
        endpoint=endpoint,
        min_score=min_score,
    )
