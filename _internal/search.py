"""
オフライン資料調査メインCLI。
ローカル資料（skill-source）をスキャンし、Ollama でAI分析を実行する。

データフロー: search.bat → search.py → collect_skill_source.ps1 → Ollama API
See _internal/ARCHITECTURE.md for design details.

使用方法:
    python search.py "検索クエリ"
    python search.py --query-file path/to/query.txt

# 更新契機: データフロー・終了コード・セキュリティ設計変更時に本コメントも更新
"""

import argparse
import codecs
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import locale
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

# パス解決（自己完結パッケージ: offline-ai/ 内で完結）
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR))
from prompt_templates import (
    KEYWORD_EXTRACTION_SYSTEM,
    SEARCH_PLAN_SYSTEM,
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_keywords_prompt,
    search_plan_prompt,
)
from model_config import (
    DEFAULT_MODEL,
    LEGACY_MODELS as _LEGACY_MODELS,
    detect_model_config,
    migrate_model_file,
)

try:
    from config import (
        OLLAMA_HOST,
        RerankConfig,
        env_float,
        env_int,
        env_timeout,
        get_rerank_config,
    )
except ValueError as exc:
    if __name__ == "__main__":
        print(f"[設定エラー] {exc}", file=sys.stderr)
        print(
            "OLLAMA_HOST は localhost / 127.0.0.1 / ::1 のみ指定できます。",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    raise
from document_schema import load_metadata_sidecar, metadata_for_line_range
from rerank import rerank_candidates

# --- 定数 ---
COLLECT_SCRIPT = SCRIPT_DIR / "scripts" / "collect_skill_source.ps1"
SKILL_SOURCE_DIR = SCRIPT_DIR.parent / "skill-source"
MODEL_CONFIG = SCRIPT_DIR / ".model"

logger = logging.getLogger("offlineai.search")

COLLECT_SOURCES_TIMEOUT_S = 30  # PowerShell 実行タイムアウト（秒）

# 機能フラグ: 環境変数で新機能の有効/無効を制御
SEARCH_PHRASE_BONUS_ENABLED = os.environ.get(
    "SEARCH_PHRASE_BONUS_ENABLED", "true"
).lower() in ("true", "1", "yes")

try:
    SEARCH_MIN_RRF_SCORE = float(os.environ.get("SEARCH_MIN_RRF_SCORE", "0.015"))
except ValueError:
    SEARCH_MIN_RRF_SCORE = 0.015

API_TIMEOUT = 180  # 秒（大コンテキスト対応、後方互換のため残置。CLI実runtimeは GENERATION_STALL_TIMEOUT を使う）

# 生成フェーズの無通信（stall）上限。thinkingデルタも受信として数える。
# Ollama 0.31.2 実測の最大無通信間隔は 8.79 秒であり、60 秒は十分な安全余裕を持つ。
GENERATION_STALL_TIMEOUT = env_timeout(
    "OFFLINE_AI_GENERATION_STALL_TIMEOUT", 60, max_value=600
)

# Ollama へ渡す keep_alive。既定 30m でコールドロード（qwen3.5:9b 5.6s / bge-m3 6.3s）を避ける。
# "0" を指定すると従来どおり応答直後にアンロードされる。
OLLAMA_KEEP_ALIVE = os.environ.get("OFFLINE_AI_KEEP_ALIVE", "30m").strip() or "30m"

# --- Embedding 関連定数 ---
EMBED_MODEL_CONFIG = SCRIPT_DIR / ".model_embed"
EMBED_CACHE_PATH = SCRIPT_DIR / "embed_cache.json"
EMBED_CHECKPOINT_PATH = SCRIPT_DIR / "embed_cache.checkpoints"
EMBED_LOCK_PATH = SCRIPT_DIR / "embed_cache.lock"
EMBED_STATUS_PATH = SCRIPT_DIR / "embed_index_status.json"
# NOTE: EMBED_MODEL_CONFIG が未設定の場合は Embedding 無効（自動フォールバックはしない）。
# 有効化手順: echo bge-m3 > offline-ai\_internal\.model_embed
RECOMMENDED_EMBED_MODEL = "bge-m3"
EMBED_REQUEST_TIMEOUT = env_timeout(
    "OFFLINE_AI_EMBED_REQUEST_TIMEOUT", 120, max_value=600
)
# Backward-compatible name used by existing callers and tests.
EMBED_TIMEOUT = EMBED_REQUEST_TIMEOUT
EMBED_BATCH_SIZE = env_int(
    "OFFLINE_AI_EMBED_BATCH_SIZE", 16, min_value=1, max_value=64
)
EMBED_MAX_PAYLOAD_BYTES = env_int(
    "OFFLINE_AI_EMBED_MAX_PAYLOAD_BYTES", 1_000_000, min_value=4096, max_value=8_000_000
)
INDEX_MAX_SECONDS = env_timeout(
    "OFFLINE_AI_INDEX_MAX_SECONDS", 21600, max_value=86400
)
RETRIEVAL_TIMEOUT = env_timeout(
    "OFFLINE_AI_RETRIEVAL_TIMEOUT", 120, max_value=3600
)
KEYWORD_EXTRACT_TIMEOUT = 90
EMBED_SIM_THRESHOLD = 0.3
SEMANTIC_SUPPORT_THRESHOLD = 0.55
LEXICAL_ANCHOR_MIN_CHARS = 4

EMBED_PROGRESS_INTERVAL = env_int("OFFLINE_AI_EMBED_PROGRESS_INTERVAL", 25, min_value=1)
EMBED_PROGRESS_SECONDS = env_int("OFFLINE_AI_EMBED_PROGRESS_SECONDS", 15, min_value=1)
EMBED_CHECKPOINT_INTERVAL = env_int(
    "OFFLINE_AI_EMBED_CHECKPOINT_INTERVAL", 100, min_value=1
)

SOURCE_EXTENSIONS = (".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".html", ".htm")
EXCLUDED_SOURCE_DIRS = {"__pycache__", "node_modules"}
EXCLUDED_SOURCE_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "secrets.*",
    "*.secret.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    # metadata sidecar は原本の複製であり、独立した資料ではない。検索対象に含めると
    # 同一内容が二重に根拠提示され、根拠precisionを下げる。
    "*.metadata.json",
)

CHUNK_MAX_CHARS = env_int("OFFLINE_AI_CHUNK_MAX_CHARS", 2000, min_value=100)
CHUNK_OVERLAP_CHARS = env_int("OFFLINE_AI_CHUNK_OVERLAP_CHARS", 200, min_value=0)
MAX_QUERY_VARIANTS = env_int("OFFLINE_AI_MAX_QUERY_VARIANTS", 3, min_value=1)
MAX_RETRIEVAL_ATTEMPTS = env_int("OFFLINE_AI_MAX_RETRIEVAL_ATTEMPTS", 2, min_value=1)
RETRIEVAL_CANDIDATE_LIMIT = env_int(
    "OFFLINE_AI_RETRIEVAL_CANDIDATE_LIMIT", 8, min_value=1
)
RETRIEVAL_PROMPT_MATCH_LIMIT = env_int(
    "OFFLINE_AI_RETRIEVAL_PROMPT_MATCH_LIMIT", 5, min_value=1
)
MAX_CHUNKS_PER_FILE = env_int("OFFLINE_AI_MAX_CHUNKS_PER_FILE", 2, min_value=1)
PROMPT_EVIDENCE_CHAR_LIMIT = env_int(
    "OFFLINE_AI_PROMPT_EVIDENCE_CHAR_LIMIT", 9000, min_value=500
)
# 根拠 snippet 単体の文字上限。Phase 3 の予算再探索で調整する対象のため
# 直書きせず定数化する（削減は A-10 の precision 確認を必要とする）。
SNIPPET_CHAR_LIMIT = env_int("OFFLINE_AI_SNIPPET_CHAR_LIMIT", 1200, min_value=100)

# --- prompt 予算と context の結合（Phase 2 scaffold） ---
# prompt・thinking・本文は同じ context ウィンドウを共有する。予算が独立だと
# 推論ありのとき合計が num_ctx を超え、done_reason: length で本文が 0 文字になる。
# GENERATION_RESERVE_TOKENS > 0 のときだけ num_ctx から逆算した上限を併用する。
# 既定 0 は逆算を無効にし、現行挙動（PROMPT_EVIDENCE_CHAR_LIMIT のみ）を保つ。
# 既定値は think レベル・num_ctx との組で実測してから確定する（決定ゲート D-1）。
GENERATION_RESERVE_TOKENS = env_int(
    "OFFLINE_AI_GENERATION_RESERVE_TOKENS", 0, min_value=0
)
# 文字→token 換算係数。実測 6,180 文字 → 4,361 tokens（1.417 文字/token）に基づく
# 保守値。日本語資料と system prompt の混合比で変動するため実測で確定する。
PROMPT_CHARS_PER_TOKEN = env_float(
    "OFFLINE_AI_PROMPT_CHARS_PER_TOKEN", 1.4, min_value=0.1, max_value=10.0
)
# token 換算誤差と固定 metadata の余裕。
PROMPT_BUDGET_SAFETY_TOKENS = env_int(
    "OFFLINE_AI_PROMPT_BUDGET_SAFETY_TOKENS", 128, min_value=0
)
RERANK_CONFIG = get_rerank_config()


def build_chat_options(*, temperature: float | None = None) -> dict:
    """Return the shared Ollama chat context/load options."""
    options = {
        "num_ctx": NUM_CTX,
        "num_batch": NUM_BATCH,
        "num_gpu": NUM_GPU,
    }
    if temperature is not None:
        options["temperature"] = temperature
    return options


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _chunk_retrieval_enabled() -> bool:
    return _env_bool("OFFLINE_AI_CHUNK_RETRIEVAL", True)


def _agentic_lite_enabled() -> bool:
    return _env_bool("OFFLINE_AI_AGENTIC_LITE", True)


# load_embed_cache() の memo（mtime/size キー）を無効化する退避路。既定は有効。
EMBED_CACHE_MEMO_ENABLED = _env_bool("OFFLINE_AI_EMBED_CACHE_MEMO", True)

# build_source_chunks() / compute_embed_generation() の read 経路 memo を
# 無効化する退避路。既定は有効。
SOURCE_CHUNK_MEMO_ENABLED = _env_bool("OFFLINE_AI_SOURCE_CHUNK_MEMO", True)


# --- GPU 最適化パラメータ ---
# ローカルOllamaモデル向けの既定設定。環境変数で調整可能。
# num_ctx: コンテキストウィンドウ（デフォルト2048は小さすぎる）
# num_batch: バッチサイズ（推論スループット向上）
# num_gpu: GPUにオフロードするレイヤー数（-1=全レイヤー）
# 推論ありで回答本文が返る最小の context。8192 / 16384 では thinking が
# context の残りを使い切って done_reason: length で本文が 0 文字になる
# （2026-09-08 実測: 8192 で 3/3、16384 で 2/10 が空回答。32768 で 0/10）。
NUM_CTX = env_int("OLLAMA_NUM_CTX", 32768, min_value=512)
NUM_BATCH = env_int("OLLAMA_NUM_BATCH", 1024, min_value=1)
NUM_GPU = env_int("OLLAMA_NUM_GPU", -1, min_value=-1)

_VALID_REASONING = {"low", "medium", "high", "off"}


class ModelStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class RetrievalAttempt:
    query: str
    keywords: list[str]
    match_count: int
    confidence: float
    evidence_status: str
    reason: str = ""


@dataclass
class RetrievalResult:
    query: str
    attempts: list[RetrievalAttempt] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    evidence_status: str = "insufficient"
    user_prompt: str = ""
    plan: dict = field(default_factory=dict)
    route: str = "keyword"
    index_state: str = "missing"
    embedding_model: str | None = None
    route_reason: str = ""
    warnings: list[str] = field(default_factory=list)


def normalize_source_path(path: str | Path) -> str:
    """skill-source 相対パスを `/` 区切りの安定キーへ正規化する。"""
    text = str(path).replace("\\", "/").strip()
    marker = "skill-source/"
    lower = text.lower()
    marker_index = lower.rfind(marker)
    if marker_index >= 0:
        text = text[marker_index + len(marker) :]
    text = text.lstrip("/")
    parts = []
    for part in text.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            continue
        parts.append(part)
    return "/".join(parts)


def is_allowed_source_path(path: str | Path) -> bool:
    """検索・Embedding の共通スキャンポリシー。秘密値候補と隠しディレクトリを除外する。"""
    rel = normalize_source_path(path)
    if not rel:
        return False
    parts = rel.split("/")
    for directory in parts[:-1]:
        lower_dir = directory.lower()
        if directory.startswith(".") or lower_dir in EXCLUDED_SOURCE_DIRS:
            return False
    filename = parts[-1]
    lower_name = filename.lower()
    if any(
        fnmatch.fnmatch(lower_name, pattern)
        for pattern in EXCLUDED_SOURCE_FILE_PATTERNS
    ):
        return False
    return Path(lower_name).suffix in SOURCE_EXTENSIONS


def iter_source_files(source_root: Path | None = None) -> list[tuple[Path, str]]:
    """共通ポリシーに合う source ファイルを `(absolute_path, normalized_rel)` で返す。"""
    root = source_root or SKILL_SOURCE_DIR
    if not root.exists():
        return []
    files: list[tuple[Path, str]] = []
    for source_file in root.rglob("*"):
        if not source_file.is_file():
            continue
        rel = normalize_source_path(source_file.relative_to(root))
        if is_allowed_source_path(rel):
            files.append((source_file, rel))
    return files


def _modified_at(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return ""


def _migrate_model_file(model_path: Path, current_model: str) -> str:
    """旧モデル名が検出された場合、DEFAULT_MODEL に自動移行する。"""
    return migrate_model_file(model_path, current_model)


def _canonical_model_reference(model: str) -> str:
    """Ollama の省略タグを ``:latest`` として正規化する。"""
    normalized = model.strip()
    last_segment = normalized.rsplit("/", 1)[-1]
    if ":" not in last_segment and "@" not in last_segment:
        return f"{normalized}:latest"
    return normalized


def _is_model_available(model: str, timeout: float = 2.0) -> ModelStatus:
    """Ollama API でモデルの可用性を3状態で判定する。"""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            requested_model = _canonical_model_reference(model)
            models = {
                _canonical_model_reference(m["name"]) for m in data.get("models", [])
            }
            return (
                ModelStatus.AVAILABLE
                if requested_model in models
                else ModelStatus.UNAVAILABLE
            )
    except (urllib.error.URLError, TimeoutError, OSError):
        return ModelStatus.UNKNOWN
    except Exception:
        return ModelStatus.UNKNOWN


def _resolve_reasoning(cli_arg: Optional[str]) -> str:
    """reasoning 引数を優先順位に従って解決する。CLI > 環境変数 > デフォルト(low)。

    既定が low なのは 2026-09-08 の実測による。num_ctx=16384 で
    low 中央値 76.7秒 / medium 106.1秒 / high は 2/3 が空回答であり、
    low が最も thinking が短く速い。
    """
    raw = cli_arg or os.environ.get("OFFLINE_AI_REASONING", "low")
    normalized = raw.lower().strip()
    if normalized not in _VALID_REASONING:
        print(
            f"[WARN] 無効な reasoning 値 '{raw}' → 'low' にフォールバック",
            file=sys.stderr,
        )
        return "low"
    return normalized


def resolve_query(
    query_file: Optional[str] = None,
    query_positional: Optional[str] = None,
) -> str:
    """クエリソースを優先順に解決し、クエリ文字列を返す。

    Args:
        query_file: クエリファイルパス（省略可）
        query_positional: 位置引数のクエリ（省略可）

    Returns:
        解決されたクエリ文字列

    Raises:
        SystemExit(2): クエリが空・未指定・読み込み失敗の場合
    """
    if query_file:
        try:
            with open(query_file, "r", encoding="utf-8") as f:
                query = f.readline().strip()
        except UnicodeDecodeError:
            with open(query_file, "r", encoding=locale.getpreferredencoding()) as f:
                query = f.readline().strip()
        except OSError as e:
            print(f"[ERROR] Cannot read query file: {e}")
            sys.exit(2)
        if not query:
            print("[ERROR] Query file is empty.")
            sys.exit(2)
        return query

    if query_positional:
        return query_positional

    print("[ERROR] No query provided.")
    sys.exit(2)


def detect_model() -> str:
    """セットアップ時に記録されたモデル名を読み取る。旧モデルは自動移行する。"""
    return detect_model_config(MODEL_CONFIG)


def find_powershell() -> str:
    """利用可能な PowerShell コマンドを返す。"""
    for cmd in ("pwsh", "powershell.exe"):
        try:
            subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                timeout=10,
            )
            return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError("PowerShell が見つかりません。")


def _extract_matches(parsed: dict) -> list:
    """PowerShell 出力の matches/results キー互換アダプタ。"""
    matches = parsed.get("matches")
    if matches is not None:
        return matches
    results = parsed.get("results")
    if results is not None:
        return results
    return []


def _build_shell_cmd(
    ps_cmd: str,
    query: str,
    original_query: str = "",
    enable_phrase_bonus: bool = True,
    enable_dense_snippet: bool = True,
    legacy_only: bool = False,
) -> list:
    """PowerShell コマンドライン引数を構築する。"""
    cmd = [
        ps_cmd,
        "-ExecutionPolicy",
        "Bypass",
        "-NoProfile",
        "-File",
        str(COLLECT_SCRIPT),
        "-Query",
        query,
        "-SourceRoot",
        str(SKILL_SOURCE_DIR),
    ]
    if legacy_only:
        return cmd
    if original_query and enable_phrase_bonus:
        cmd.extend(["-OriginalQuery", original_query])
    if not enable_phrase_bonus:
        cmd.extend(["-EnablePhraseBonus:$false"])
    if not enable_dense_snippet:
        cmd.extend(["-EnableDenseSnippet:$false"])
    return cmd


def _decode_stdout(raw: bytes) -> str:
    """PowerShell stdout のデコード。"""
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw.decode("cp932", errors="replace").strip()


def _collect_sources_fallback(ps_cmd: str, query: str, reason: str = "unknown") -> dict:
    """フォールバック: 新機能を全て無効化して旧モードで再実行。"""
    shell_cmd = _build_shell_cmd(ps_cmd, query, legacy_only=True)
    try:
        result = subprocess.run(
            shell_cmd,
            capture_output=True,
            timeout=COLLECT_SOURCES_TIMEOUT_S,
            cwd=str(SCRIPT_DIR),
        )
        if result.returncode == 0:
            stdout = _decode_stdout(result.stdout or b"")
            if stdout:
                logger.info(
                    "collect_sources fallback succeeded (mode=legacy, reason=%s)",
                    reason,
                )
                parsed = json.loads(stdout)
                return {
                    "status": "fallback",
                    "matches": _extract_matches(parsed),
                    "scannedFileCount": parsed.get("scannedFileCount", 0),
                    "error_code": "primary_failed",
                    "fallback_mode": "legacy",
                    "fallback_reason": reason,
                }
    except Exception:
        pass
    logger.error(
        "collect_sources fallback also failed (mode=legacy, reason=%s)", reason
    )
    return {
        "status": "error",
        "matches": [],
        "scannedFileCount": 0,
        "error_code": "total_failure",
        "fallback_mode": "legacy",
        "fallback_reason": reason,
    }


def collect_sources(query: str, original_query: str = "") -> dict:
    """collect_skill_source.ps1 を実行してローカル資料を取得する。"""
    ps_cmd = find_powershell()
    use_new = SEARCH_PHRASE_BONUS_ENABLED
    shell_cmd = _build_shell_cmd(
        ps_cmd,
        query,
        original_query,
        enable_phrase_bonus=use_new,
        enable_dense_snippet=use_new,
    )

    try:
        result = subprocess.run(
            shell_cmd,
            capture_output=True,
            timeout=COLLECT_SOURCES_TIMEOUT_S,
            cwd=str(SCRIPT_DIR),
        )
        if result.returncode != 0:
            logger.warning(
                "collect_sources failed (exit=%d, mode=%s): %s",
                result.returncode,
                "new" if use_new else "legacy",
                _decode_stdout(result.stderr or b"")[:200],
            )
            return _collect_sources_fallback(ps_cmd, query, reason="nonzero_exit")

        stdout = _decode_stdout(result.stdout or b"")
        if not stdout:
            return _collect_sources_fallback(ps_cmd, query, reason="empty_output")

        parsed = json.loads(stdout)
        return {
            "status": parsed.get("status", "ok"),
            "matches": _extract_matches(parsed),
            "scannedFileCount": parsed.get("scannedFileCount", 0),
            "message": parsed.get("message", ""),
            "error_code": None,
            "fallback_mode": None,
            "fallback_reason": None,
        }
    except FileNotFoundError:
        if ps_cmd != "powershell.exe":
            # pwsh が見つからない場合 powershell.exe でフォールバック
            return _collect_sources_fallback(
                "powershell.exe", query, reason="pwsh_not_found"
            )
        raise
    except subprocess.TimeoutExpired:
        logger.error(
            "collect_sources timeout (%ds, mode=%s)",
            COLLECT_SOURCES_TIMEOUT_S,
            "new" if use_new else "legacy",
        )
        return _collect_sources_fallback(ps_cmd, query, reason="timeout")
    except json.JSONDecodeError:
        logger.error(
            "collect_sources JSON decode error (mode=%s)",
            "new" if use_new else "legacy",
        )
        return _collect_sources_fallback(ps_cmd, query, reason="json_decode")
    except Exception as e:
        logger.error(
            "collect_sources unexpected error (mode=%s): %s",
            "new" if use_new else "legacy",
            e,
        )
        return _collect_sources_fallback(ps_cmd, query, reason="unexpected")


def extract_matches_with_logging(response: dict) -> list:
    """collect_sources の共通戻り値契約を処理し、matches リストを返す。"""
    status = response.get("status", "ok")
    if status == "fallback":
        logger.warning(
            "collect_sources fallback (reason=%s, mode=%s)",
            response.get("fallback_reason"),
            response.get("fallback_mode"),
        )
    elif status == "error":
        logger.error(
            "collect_sources error (code=%s, reason=%s)",
            response.get("error_code"),
            response.get("fallback_reason"),
        )
    return response.get("matches", [])


# ---------------------------------------------------------------------------
# Embedding / キーワード抽出 関連関数
# ---------------------------------------------------------------------------


def extract_keywords_via_llm(query: str, model: str) -> str:
    """Ollama でキーワード抽出を行う。失敗時は元クエリをフォールバックとして返す。"""
    url = f"{OLLAMA_HOST}/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": KEYWORD_EXTRACTION_SYSTEM},
            {"role": "user", "content": extract_keywords_prompt(query)},
        ],
        "stream": False,
        "options": {
            "num_ctx": 2048,
            "num_batch": NUM_BATCH,
            "num_gpu": NUM_GPU,
        },
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=KEYWORD_EXTRACT_TIMEOUT) as resp:
            raw = resp.read()
            if not raw:
                print(
                    "[WARN] キーワード抽出: 空レスポンス。元クエリを使用します",
                    file=sys.stderr,
                )
                return query
            data = json.loads(raw.decode("utf-8"))
            text = data.get("message", {}).get("content", "").strip()
            if not text:
                print(
                    "[WARN] キーワード抽出: 空テキスト。元クエリを使用します",
                    file=sys.stderr,
                )
                return query
            print(f"検索キーワード: {text}")
            return text
    except TimeoutError:
        print(
            "[WARN] キーワード抽出: タイムアウト。元クエリを使用します", file=sys.stderr
        )
        return query
    except urllib.error.HTTPError as e:
        print(
            f"[WARN] キーワード抽出: HTTPエラー({e.code})。元クエリを使用します",
            file=sys.stderr,
        )
        return query
    except urllib.error.URLError as e:
        print(
            f"[WARN] キーワード抽出: 接続エラー({e})。元クエリを使用します",
            file=sys.stderr,
        )
        return query
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(
            f"[WARN] キーワード抽出: パースエラー({e})。元クエリを使用します",
            file=sys.stderr,
        )
        return query


def detect_embed_model() -> Optional[str]:
    """`.model_embed` ファイルからembeddingモデル名を読む。未設定なら None を返す。"""
    if EMBED_MODEL_CONFIG.exists():
        model = EMBED_MODEL_CONFIG.read_text(encoding="utf-8-sig").strip()
        if model:
            return model
    return None


class EmbeddingBatchError(RuntimeError):
    """/api/embed の応答を安全に解釈できない場合の安定エラー。"""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _validate_embedding_vector(vector: object) -> list[float]:
    if not isinstance(vector, list) or not vector:
        raise EmbeddingBatchError("EMBED_EMPTY_VECTOR", "embedding vector is empty")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in vector
    ):
        raise EmbeddingBatchError(
            "EMBED_INVALID_VECTOR", "embedding vector contains a non-finite value"
        )
    return [float(value) for value in vector]


def _get_embeddings(
    texts: list[str], embed_model: str, *, timeout: float | None = None
) -> list[list[float]]:
    """Ollama ``/api/embed`` を配列入力で呼び、順序とshapeをfail-closed検証する。"""
    if not texts:
        return []
    normalized = [str(text)[:2000] for text in texts]
    body = {
        "model": embed_model,
        "input": normalized,
        "truncate": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > EMBED_MAX_PAYLOAD_BYTES:
        raise EmbeddingBatchError(
            "EMBED_PAYLOAD_TOO_LARGE",
            "embedding request exceeds the configured payload limit",
            retryable=True,
        )
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or EMBED_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except TimeoutError as exc:
        raise EmbeddingBatchError(
            "EMBED_REQUEST_TIMEOUT", "embedding request timed out", retryable=True
        ) from exc
    except urllib.error.HTTPError as exc:
        retryable = exc.code in {413, 429, 500, 502, 503, 504}
        code = "EMBED_BATCH_UNSUPPORTED" if exc.code in {404, 405} else f"EMBED_HTTP_{exc.code}"
        raise EmbeddingBatchError(code, f"embedding HTTP error {exc.code}", retryable=retryable) from exc
    except urllib.error.URLError as exc:
        raise EmbeddingBatchError(
            "EMBED_TRANSPORT_ERROR", "embedding connection failed", retryable=True
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EmbeddingBatchError(
            "EMBED_MALFORMED_RESPONSE", "embedding response is not valid JSON"
        ) from exc
    except OSError as exc:
        raise EmbeddingBatchError("EMBED_IO_ERROR", "embedding I/O failed") from exc

    if not isinstance(data, dict) or not isinstance(data.get("embeddings"), list):
        raise EmbeddingBatchError(
            "EMBED_MALFORMED_RESPONSE", "response does not contain embeddings"
        )
    raw_vectors = data["embeddings"]
    if len(raw_vectors) != len(normalized):
        raise EmbeddingBatchError(
            "EMBED_COUNT_MISMATCH", "embedding count does not match input count"
        )
    vectors = [_validate_embedding_vector(vector) for vector in raw_vectors]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise EmbeddingBatchError(
            "EMBED_DIMENSION_MISMATCH", "embedding vectors have different dimensions"
        )
    return vectors


def _get_embedding(text: str, embed_model: str) -> Optional[list]:
    """Query embedding compatibility wrapper; even a single query uses ``/api/embed``."""
    try:
        return _get_embeddings([text], embed_model)[0]
    except EmbeddingBatchError as exc:
        print(f"[WARN] Embedding取得: {exc.code}", file=sys.stderr)
        return None


_ORIGINAL_GET_EMBEDDING = _get_embedding


def _get_index_embedding_batch(
    texts: list[str], embed_model: str, *, allow_split: bool = True
) -> list[list[float] | None]:
    """Generate one batch, splitting once for request-size/resource failures.

    The compatibility branch is only a test seam for legacy callers that
    replace ``_get_embedding``; production always uses the array API.
    """
    if _get_embedding is not _ORIGINAL_GET_EMBEDDING:
        return [_get_embedding(text, embed_model) for text in texts]
    try:
        return list(_get_embeddings(texts, embed_model))
    except EmbeddingBatchError as exc:
        if exc.retryable and allow_split and len(texts) > 1:
            midpoint = max(1, len(texts) // 2)
            left = _get_index_embedding_batch(
                texts[:midpoint], embed_model, allow_split=False
            )
            right = _get_index_embedding_batch(
                texts[midpoint:], embed_model, allow_split=False
            )
            return left + right
        raise


def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    """コサイン類似度を標準ライブラリ math のみで計算する。

    次元数が異なるベクトルは比較不能のため 0.0 を返す（zip による暗黙の
    切り詰めで誤った類似度を返すのを防ぐ二重防御）。
    """
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _file_sha256(path: Path) -> str:
    """ファイルの SHA-256 ハッシュ文字列を返す。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _chunk_id(path: str, index: int) -> str:
    return f"{normalize_source_path(path)}#{index:04d}"


def _apply_metadata_fields(item: dict, metadata: dict | None) -> dict:
    if not metadata:
        return item
    clean_metadata = {
        key: value for key, value in metadata.items() if value is not None
    }
    if clean_metadata:
        item["metadata"] = clean_metadata
    for key in (
        "parser",
        "page",
        "bbox",
        "layout_type",
        "source_title",
        "confidence",
        "block_id",
        "parser_warnings",
    ):
        if key in clean_metadata and clean_metadata[key] is not None:
            item[key] = clean_metadata[key]
    return item


_CACHE_METADATA_KEYS = {
    "metadata",
    "parser",
    "page",
    "bbox",
    "layout_type",
    "source_title",
    "confidence",
    "block_id",
    "parser_warnings",
}


def _strip_cache_metadata(entry: dict) -> dict:
    """Return an embedding cache entry without volatile sidecar metadata."""
    return {
        key: value for key, value in entry.items() if key not in _CACHE_METADATA_KEYS
    }


def _current_sidecar_metadata_for_entry(
    entry: dict, sidecar_cache: dict[str, dict | None] | None = None
) -> dict:
    rel_path = normalize_source_path(
        entry.get("path") or str(entry.get("chunk_id", "")).split("#", 1)[0]
    )
    if not rel_path:
        return {}
    source_file = SKILL_SOURCE_DIR / rel_path
    if sidecar_cache is None:
        metadata = load_metadata_sidecar(source_file)
    else:
        if rel_path not in sidecar_cache:
            sidecar_cache[rel_path] = load_metadata_sidecar(source_file)
        metadata = sidecar_cache[rel_path]
    return metadata_for_line_range(
        metadata,
        entry.get("start_line"),
        entry.get("end_line"),
    )


def _line_chunks(
    path: str,
    lines: list[str],
    *,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[dict]:
    chunks: list[dict] = []
    buffer: list[str] = []
    start_line = 1
    heading = ""
    current_heading = ""

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line, heading
        text = "\n".join(buffer).strip()
        if not text:
            buffer = []
            start_line = end_line + 1
            return
        idx = len(chunks) + 1
        chunks.append(
            {
                "path": normalize_source_path(path),
                "chunk_id": _chunk_id(path, idx),
                "heading": heading,
                "start_line": start_line,
                "end_line": end_line,
                "text": text,
                "text_sha256": _text_sha256(text),
            }
        )
        if CHUNK_OVERLAP_CHARS > 0 and len(text) > CHUNK_OVERLAP_CHARS:
            overlap = text[-CHUNK_OVERLAP_CHARS:]
            buffer = [overlap]
            start_line = end_line
            heading = current_heading
        else:
            buffer = []
            start_line = end_line + 1
            heading = current_heading

    for line_no, line in enumerate(lines, 1):
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if match:
            current_heading = match.group(2).strip()
            if buffer:
                flush(line_no - 1)
            heading = current_heading
            start_line = line_no

        if not buffer:
            start_line = line_no
            heading = current_heading
        buffer.append(line)
        if sum(len(x) + 1 for x in buffer) >= max_chars:
            flush(line_no)

    if buffer:
        flush(len(lines))
    return chunks


def chunk_source_file(
    path: Path, rel_path: str, *, max_chars: int = CHUNK_MAX_CHARS
) -> list[dict]:
    """source ファイルを検索・Embedding 用チャンクへ分割する。"""
    # 本文とhashを同じ読出しbytesから作り、途中の更新で別世代を混ぜない。
    # 検索後に更新された資料はビューア側のhash照合で拒否される。
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    chunks = _line_chunks(rel_path, lines, max_chars=max_chars)
    modified = _modified_at(path)
    file_sha = hashlib.sha256(data).hexdigest()
    sidecar_metadata = load_metadata_sidecar(path)
    for chunk in chunks:
        chunk["modifiedAt"] = modified
        chunk["file_sha256"] = file_sha
        _apply_metadata_fields(
            chunk,
            metadata_for_line_range(
                sidecar_metadata,
                chunk.get("start_line"),
                chunk.get("end_line"),
            ),
        )
    return chunks


def build_source_chunks(source_root: Path | None = None) -> list[dict]:
    chunks: list[dict] = []
    for path, rel in iter_source_files(source_root):
        try:
            chunks.extend(chunk_source_file(path, rel))
        except OSError:
            continue
    return chunks


# build_source_chunks() / compute_embed_generation() の read 経路専用 memo。
# キーは skill-source の (rel, st_mtime_ns, st_size) 集合であり、追加・更新・
# 削除のいずれでもキーが変わるため自己無効化する。index build の writer 経路は
# build_source_chunks() を直接呼び、この memo を通らない。
_SOURCE_CHUNK_MEMO_KEY: tuple | None = None
_SOURCE_CHUNK_MEMO_VALUE: list[dict] | None = None
_EMBED_GENERATION_MEMO_KEY: tuple | None = None
_EMBED_GENERATION_MEMO_VALUE: str | None = None


def _invalidate_source_chunk_memo() -> None:
    global _SOURCE_CHUNK_MEMO_KEY, _SOURCE_CHUNK_MEMO_VALUE
    global _EMBED_GENERATION_MEMO_KEY, _EMBED_GENERATION_MEMO_VALUE
    _SOURCE_CHUNK_MEMO_KEY = None
    _SOURCE_CHUNK_MEMO_VALUE = None
    _EMBED_GENERATION_MEMO_KEY = None
    _EMBED_GENERATION_MEMO_VALUE = None


def _source_chunk_memo_key(source_root: Path | None = None) -> tuple | None:
    """skill-source の identity キーを返す。stat に失敗したら None（memo 無効）。"""
    entries = []
    for path, rel in iter_source_files(source_root):
        try:
            stat = path.stat()
        except OSError:
            return None
        entries.append((rel, stat.st_mtime_ns, stat.st_size))
    entries.sort()
    return tuple(entries)


def load_source_chunks_cached(source_root: Path | None = None) -> tuple[list[dict], tuple | None]:
    """read 経路向けに chunk 集合を memo 付きで返す。

    戻り値は ``(chunks, memo_key)``。``memo_key`` は generation memo の
    合成キーに使う。戻る list は memo 有効時に共有参照となるため、
    呼び出し元は内容を変更してはならない（read-only 契約）。
    """
    global _SOURCE_CHUNK_MEMO_KEY, _SOURCE_CHUNK_MEMO_VALUE
    if not SOURCE_CHUNK_MEMO_ENABLED:
        return build_source_chunks(source_root), None
    key = _source_chunk_memo_key(source_root)
    if key is not None and key == _SOURCE_CHUNK_MEMO_KEY and _SOURCE_CHUNK_MEMO_VALUE is not None:
        return _SOURCE_CHUNK_MEMO_VALUE, key
    chunks = build_source_chunks(source_root)
    if key is not None:
        _SOURCE_CHUNK_MEMO_KEY = key
        _SOURCE_CHUNK_MEMO_VALUE = chunks
    return chunks, key


def compute_embed_generation_cached(
    embed_model: str, chunks: list[dict], memo_key: tuple | None
) -> str:
    """generation digest を chunk memo キーと model の組で memo する。"""
    global _EMBED_GENERATION_MEMO_KEY, _EMBED_GENERATION_MEMO_VALUE
    if not SOURCE_CHUNK_MEMO_ENABLED or memo_key is None:
        return compute_embed_generation(embed_model, chunks)
    key = (memo_key, _canonical_embed_model(embed_model))
    if key == _EMBED_GENERATION_MEMO_KEY and _EMBED_GENERATION_MEMO_VALUE is not None:
        return _EMBED_GENERATION_MEMO_VALUE
    generation = compute_embed_generation(embed_model, chunks)
    _EMBED_GENERATION_MEMO_KEY = key
    _EMBED_GENERATION_MEMO_VALUE = generation
    return generation


EMBED_CACHE_VERSION = 4
EMBED_CHECKPOINT_SCHEMA_VERSION = 2
_EMBED_PROCESS_LOCK = threading.Lock()
_CHECKPOINT_BATCH_RE = re.compile(r"^batch-(\d{8})\.json$")
_CHECKPOINT_BATCH_TMP_RE = re.compile(
    r"^batch-(\d{8})\.json\.tmp(?:\.[0-9a-f]{32})?$"
)
_CHECKPOINT_STATE_TMP_RE = re.compile(r"^state\.json\.tmp(?:\.[0-9a-f]{32})?$")

# status JSONはEmbedding writer lockと別のlockを持つ。Embedding build中に
# statusを書き込むため、同じlockを再取得するとdeadlockになるからである。
_INDEX_STATUS_PROCESS_LOCK = threading.Lock()


class IndexStatusConflictError(RuntimeError):
    """古いjob/generationまたは不正な状態遷移のstatus保存を拒否した。"""


@contextmanager
def _index_status_lock():
    """status JSONのread/validate/replaceをprocess間で直列化する。"""
    _INDEX_STATUS_PROCESS_LOCK.acquire()
    handle = None
    acquired = False
    lock_path = EMBED_STATUS_PATH.with_name(EMBED_STATUS_PATH.name + ".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            _safe_fsync(handle)
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        acquired = True
        yield True
    except (OSError, IOError):
        yield False
    finally:
        if acquired and handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        if handle is not None:
            handle.close()
        _INDEX_STATUS_PROCESS_LOCK.release()


class EmbedIndexBusyError(RuntimeError):
    """別プロセスまたは別threadがEmbedding cacheを構築中。"""


class EmbedCachePersistenceError(RuntimeError):
    """最終cacheまたはcheckpointの永続化に失敗した。"""


class EmbedBuildError(RuntimeError):
    """Embedding index cannot be promoted to ready."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def compute_embed_generation(embed_model: str, chunks: list[dict]) -> str:
    """source/model/chunk契約から決定的なgeneration digestを作る。"""
    identities = [
        {
            "chunk_id": str(chunk.get("chunk_id", "")),
            "file_sha256": str(chunk.get("file_sha256", "")),
            "text_sha256": str(chunk.get("text_sha256", "")),
        }
        for chunk in chunks
    ]
    identities.sort(key=lambda item: item["chunk_id"])
    contract = {
        "embed_model": _canonical_embed_model(embed_model),
        "cache_version": EMBED_CACHE_VERSION,
        "chunking": {
            "max_chars": CHUNK_MAX_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
            "extensions": SOURCE_EXTENSIONS,
            "excluded_dirs": sorted(EXCLUDED_SOURCE_DIRS),
            "excluded_patterns": EXCLUDED_SOURCE_FILE_PATTERNS,
        },
        "chunks": identities,
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_embed_model(model: str | None) -> str:
    return _canonical_model_reference(str(model or "")) if model else ""


def _safe_fsync(handle) -> None:
    try:
        handle.flush()
        os.fsync(handle.fileno())
    except (OSError, AttributeError):
        # Filesystems without fsync support still get the atomic rename contract.
        pass


def _atomic_write_json(path: Path, payload: dict) -> bool:
    """同一directoryへのwrite、flush/fsync、replaceを一体化する。"""
    if path.parent == EMBED_CHECKPOINT_PATH:
        _assert_runtime_directory_safe(EMBED_CHECKPOINT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Windowsでは固定名tmpを複数writerが共有すると、片方のopen/replace/
    # cleanupが他方の中間ファイルを壊し、WinError 32/5になり得る。tmpは
    # writerごとに一意化し、失敗時にはこの呼出しが作ったものだけを消す。
    tmp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            _safe_fsync(handle)
        os.replace(tmp_path, path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("atomic JSON save failed for %s: %s", path.name, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


def _checkpoint_batch_files() -> list[Path]:
    if not EMBED_CHECKPOINT_PATH.exists():
        return []
    try:
        return sorted(
            path
            for path in EMBED_CHECKPOINT_PATH.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and _CHECKPOINT_BATCH_RE.match(path.name)
        )
    except OSError:
        return []


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except OSError:
        return False


def _assert_runtime_directory_safe(path: Path) -> None:
    if path.is_symlink() or (path.exists() and _is_reparse_point(path)):
        raise EmbedCachePersistenceError(
            f"runtime生成物directoryがreparse pointです: {path.name}"
        )


def _checkpoint_tmp_files() -> list[Path]:
    if not EMBED_CHECKPOINT_PATH.exists():
        return []
    try:
        return sorted(
            path
            for path in EMBED_CHECKPOINT_PATH.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and (
                _CHECKPOINT_BATCH_TMP_RE.match(path.name)
                or _CHECKPOINT_STATE_TMP_RE.match(path.name)
            )
        )
    except OSError:
        return []


def _remove_checkpoint_files(*, include_state: bool = False) -> None:
    """生成物として認識できる固定名だけを削除する。"""
    candidates = _checkpoint_batch_files() + _checkpoint_tmp_files()
    if include_state:
        state_path = EMBED_CHECKPOINT_PATH / "state.json"
        if state_path.is_file() and not state_path.is_symlink():
            candidates.append(state_path)
    for path in candidates:
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("checkpoint cleanup failed for %s: %s", path.name, exc)


def _remove_checkpoint_tmp_files() -> None:
    """resume対象外の中間fileだけを整理する。確定batchは保持する。"""
    for path in _checkpoint_tmp_files():
        try:
            path.unlink()
        except OSError as exc:
            logger.warning(
                "checkpoint temporary cleanup failed for %s: %s", path.name, exc
            )


def _new_checkpoint_state(embed_model: str, generation: str | None = None) -> dict:
    now = _utc_now()
    return {
        "schema_version": EMBED_CHECKPOINT_SCHEMA_VERSION,
        "cache_version": EMBED_CACHE_VERSION,
        "embed_model": _canonical_embed_model(embed_model),
        "chunking": {
            "max_chars": CHUNK_MAX_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
        },
        "generation": generation or os.urandom(16).hex(),
        "next_batch": 1,
        "created_at": now,
        "updated_at": now,
    }


def _checkpoint_header_matches(
    state: dict, embed_model: str, generation: str | None = None
) -> bool:
    chunking = state.get("chunking") if isinstance(state.get("chunking"), dict) else {}
    return (
        state.get("schema_version") == EMBED_CHECKPOINT_SCHEMA_VERSION
        and state.get("cache_version") == EMBED_CACHE_VERSION
        and _canonical_embed_model(state.get("embed_model"))
        == _canonical_embed_model(embed_model)
        and chunking.get("max_chars") == CHUNK_MAX_CHARS
        and chunking.get("overlap_chars") == CHUNK_OVERLAP_CHARS
        and isinstance(state.get("generation"), str)
        and bool(state.get("generation"))
        and (generation is None or state.get("generation") == generation)
        and isinstance(state.get("next_batch"), int)
        and state.get("next_batch") >= 1
    )


def _valid_embedding_entry(entry: object, chunk_id: str | None = None) -> bool:
    if not isinstance(entry, dict):
        return False
    if chunk_id and entry.get("chunk_id") != chunk_id:
        return False
    vector = entry.get("embedding")
    return (
        isinstance(vector, list)
        and bool(vector)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in vector
        )
        and bool(entry.get("file_sha256"))
        and bool(entry.get("text_sha256"))
        and isinstance(entry.get("text"), str)
    )


def _cache_header_matches(
    cache: dict, embed_model: str, generation: str | None = None
) -> bool:
    chunking = cache.get("chunking") if isinstance(cache.get("chunking"), dict) else {}
    return (
        _canonical_embed_model(cache.get("embed_model"))
        == _canonical_embed_model(embed_model)
        and cache.get("version") == EMBED_CACHE_VERSION
        and chunking.get("max_chars") == CHUNK_MAX_CHARS
        and chunking.get("overlap_chars") == CHUNK_OVERLAP_CHARS
        and (generation is None or cache.get("generation") == generation)
    )


def _load_checkpoint(
    embed_model: str, generation: str | None = None
) -> tuple[dict, dict[str, dict]]:
    """checkpointを読み込み、不一致generationはbatchだけ整理して新規化する。"""
    _assert_runtime_directory_safe(EMBED_CHECKPOINT_PATH)
    state_path = EMBED_CHECKPOINT_PATH / "state.json"
    state: dict | None = None
    if state_path.is_file() and not state_path.is_symlink():
        try:
            candidate = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                state = candidate
        except (OSError, json.JSONDecodeError):
            logger.warning("checkpoint state is unreadable; starting a new generation")

    if state is None or not _checkpoint_header_matches(state, embed_model, generation):
        _remove_checkpoint_files(include_state=True)
        state = _new_checkpoint_state(embed_model, generation)
        if not _atomic_write_json(state_path, state):
            raise EmbedCachePersistenceError("checkpoint stateの初期化に失敗しました")
        return state, {}

    _remove_checkpoint_tmp_files()
    entries: dict[str, dict] = {}
    highest_sequence = 0
    for batch_path in _checkpoint_batch_files():
        match = _CHECKPOINT_BATCH_RE.match(batch_path.name)
        if not match:
            continue
        sequence = int(match.group(1))
        highest_sequence = max(highest_sequence, sequence)
        try:
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            if (
                not isinstance(batch, dict)
                or batch.get("schema_version") != EMBED_CHECKPOINT_SCHEMA_VERSION
                or batch.get("generation") != state.get("generation")
                or batch.get("sequence") != sequence
                or not isinstance(batch.get("entries"), list)
            ):
                continue
            for entry in batch["entries"]:
                chunk_id = entry.get("chunk_id") if isinstance(entry, dict) else None
                if _valid_embedding_entry(entry, chunk_id):
                    entries[chunk_id] = _strip_cache_metadata(entry)
        except (OSError, json.JSONDecodeError):
            logger.warning("checkpoint batchを読み込めません: %s", batch_path.name)
    if int(state.get("next_batch", 1)) <= highest_sequence:
        state["next_batch"] = highest_sequence + 1
        state["updated_at"] = _utc_now()
        if not _atomic_write_json(state_path, state):
            raise EmbedCachePersistenceError("checkpoint stateの修復に失敗しました")
    return state, entries


def _persist_checkpoint_batch(state: dict, entries: list[dict]) -> bool:
    if not entries:
        return True
    sequence = int(state.get("next_batch", 1))
    generation = state.get("generation")
    payload = {
        "schema_version": EMBED_CHECKPOINT_SCHEMA_VERSION,
        "generation": generation,
        "sequence": sequence,
        "entries": [_strip_cache_metadata(entry) for entry in entries],
    }
    batch_path = EMBED_CHECKPOINT_PATH / f"batch-{sequence:08d}.json"
    if not _atomic_write_json(batch_path, payload):
        return False
    state["next_batch"] = sequence + 1
    state["updated_at"] = _utc_now()
    return _atomic_write_json(EMBED_CHECKPOINT_PATH / "state.json", state)


def _format_embed_progress(event: dict) -> str:
    total = int(event.get("total", 0))
    processed = int(event.get("processed", 0))
    percent = 100.0 if total == 0 else min(100.0, processed * 100.0 / total)
    details = (
        f"新規 {int(event.get('generated', 0)):,}、"
        f"再利用 {int(event.get('reused', 0)):,}、"
        f"失敗 {int(event.get('failed', 0)):,}、"
        f"checkpoint {int(event.get('checkpointed', 0)):,}"
    )
    phase = event.get("phase")
    if phase == "checkpoint":
        return f"Embedding checkpoint保存: {int(event.get('checkpointed', 0)):,} / {total:,}、{details}"
    if phase == "resumed":
        return f"Embeddingインデックスを再開: {processed:,} / {total:,} ({percent:.1f}%)、{details}"
    if phase == "completed":
        return f"Embeddingインデックス構築完了: {processed:,} / {total:,} ({percent:.1f}%)、{details}"
    if phase == "interrupted":
        return f"Embeddingインデックスを中断: {processed:,} / {total:,} ({percent:.1f}%)、{details}"
    return (
        f"Embeddingインデックス: {processed:,} / {total:,} ({percent:.1f}%)、{details}"
    )


def _emit_embed_progress(emit_progress, **values) -> None:
    if emit_progress:
        emit_progress(dict(values))


@contextmanager
def _embed_writer_lock():
    """thread/process間でEmbedding writerを1本に制限する。"""
    if not _EMBED_PROCESS_LOCK.acquire(blocking=False):
        yield False
        return
    handle = None
    acquired = False
    try:
        if EMBED_LOCK_PATH.is_symlink() or (
            EMBED_LOCK_PATH.exists() and _is_reparse_point(EMBED_LOCK_PATH)
        ):
            raise EmbedCachePersistenceError("runtime lock fileがreparse pointです")
        EMBED_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        handle = open(EMBED_LOCK_PATH, "a+b")
        handle.seek(0)
        if handle.tell() == 0:
            handle.write(b"0")
            _safe_fsync(handle)
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield True
    except (OSError, IOError):
        yield False
    finally:
        if acquired and handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        if handle is not None:
            handle.close()
        _EMBED_PROCESS_LOCK.release()


def _empty_embed_cache() -> dict:
    """空の Embedding キャッシュ構造を返す（chunk embedding / version 3）。"""
    return {
        "version": EMBED_CACHE_VERSION,
        "embed_model": None,
        "generation": None,
        "chunking": {
            "max_chars": CHUNK_MAX_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
        },
        "entries": {},
    }


# load_embed_cache() の read 経路専用 memo。キーは (絶対path, st_mtime_ns, st_size)。
# writer 経路（build_or_update_embed_index 内の3箇所）は use_memo=False で必ず
# fresh load し、save_embed_cache 成功時に破棄する。返るdictは共有参照であり、
# read 経路（retrieval / status）は変更してはならない（read-only 契約）。
_EMBED_CACHE_MEMO_KEY: tuple | None = None
_EMBED_CACHE_MEMO_VALUE: dict | None = None


def _invalidate_embed_cache_memo() -> None:
    global _EMBED_CACHE_MEMO_KEY, _EMBED_CACHE_MEMO_VALUE
    _EMBED_CACHE_MEMO_KEY = None
    _EMBED_CACHE_MEMO_VALUE = None


# embed_cache.json の逐次読込で一度に decode する bytes 量。
# 1 entry は約 24 KB（1024 次元の embedding が支配的）なので、1 MB あれば
# 常に 1 件以上を含み、buffer は数 MB を超えない。
_EMBED_CACHE_STREAM_CHUNK_BYTES = 1 << 20


class _JsonMemberReader:
    """UTF-8 の JSON object を、全文を 1 つの str にせずメンバ単位で読む。

    441 MB 級の ``embed_cache.json`` を ``read_text`` で読むと、内容に BMP 外の
    文字が 1 つでもあれば str 全体が UCS-4 になり、441 MB のファイルが 1.7 GB の
    str になる。text mode の読み込み自体も chunk の結合で一時的に倍近く確保する
    ため、peak working set が 3.8 GB へ達していた（A-6 の 2.5 GB 未満を超過）。
    値の parse そのものは標準 ``json`` の C スキャナへ渡し、buffer だけを
    slide させることで peak を parse 済み構造＋数 MB に抑える。
    """

    _WHITESPACE = " \t\r\n"

    def __init__(self, handle):
        self._handle = handle
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._scanner = json.JSONDecoder()
        self._buf = ""
        self._pos = 0
        self._eof = False

    def _fill(self) -> bool:
        """次の chunk を buffer へ足す。EOF まで読み切っていれば False。"""
        if self._eof:
            return False
        chunk = self._handle.read(_EMBED_CACHE_STREAM_CHUNK_BYTES)
        if not chunk:
            self._buf += self._decoder.decode(b"", True)
            self._eof = True
            return False
        # 消費済みの前半を捨ててから足す（buffer を成長させない）。
        if self._pos:
            self._buf = self._buf[self._pos :]
            self._pos = 0
        self._buf += self._decoder.decode(chunk)
        return True

    def peek(self) -> str:
        """次の非空白文字を返す。無ければ空文字。"""
        while True:
            while self._pos < len(self._buf) and self._buf[self._pos] in self._WHITESPACE:
                self._pos += 1
            if self._pos < len(self._buf):
                return self._buf[self._pos]
            if not self._fill():
                return ""

    def consume(self, char: str) -> None:
        if self.peek() != char:
            raise ValueError(f"expected {char!r} in embed cache")
        self._pos += 1

    def read_value(self):
        """次の JSON 値を 1 つ読む。

        buffer 末尾で切れた数値・文字列を途中まで受け取らないよう、値の直後が
        buffer 末尾のままなら EOF になるまで読み足してから確定する。
        """
        self.peek()
        while True:
            try:
                value, end = self._scanner.raw_decode(self._buf, self._pos)
            except json.JSONDecodeError:
                if not self._fill():
                    raise
                continue
            if end >= len(self._buf) and self._fill():
                continue
            self._pos = end
            return value


def _load_embed_cache_streaming(path: Path) -> Optional[dict]:
    """entries を 1 件ずつ parse して cache を読む。構造が想定外なら None。

    None を返した場合、呼び出し元は従来どおり全文 parse へ fallback する。
    返る dict は全文 parse と同じ構造・同じ順序であり、cache 形式は変えない。
    """
    try:
        with path.open("rb") as handle:
            reader = _JsonMemberReader(handle)
            if reader.peek() != "{":
                return None
            reader.consume("{")
            data: dict = {}
            if reader.peek() == "}":
                reader.consume("}")
                return data
            while True:
                key = reader.read_value()
                if not isinstance(key, str):
                    return None
                reader.consume(":")
                if key == "entries":
                    if reader.peek() != "{":
                        return None
                    reader.consume("{")
                    entries: dict = {}
                    if reader.peek() == "}":
                        reader.consume("}")
                    else:
                        while True:
                            entry_key = reader.read_value()
                            if not isinstance(entry_key, str):
                                return None
                            reader.consume(":")
                            entries[entry_key] = reader.read_value()
                            nxt = reader.peek()
                            if nxt == ",":
                                reader.consume(",")
                                continue
                            if nxt == "}":
                                reader.consume("}")
                                break
                            return None
                    data["entries"] = entries
                else:
                    data[key] = reader.read_value()
                nxt = reader.peek()
                if nxt == ",":
                    reader.consume(",")
                    continue
                if nxt == "}":
                    reader.consume("}")
                    break
                return None
            return data
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError, OSError, RecursionError):
        return None


def _normalize_embed_cache(data) -> dict:
    if not isinstance(data, dict) or "entries" not in data:
        return _empty_embed_cache()
    data.setdefault("version", 1)
    data.setdefault("embed_model", None)
    data.setdefault("chunking", {})
    return data


def _load_embed_cache_from_disk() -> dict:
    streamed = _load_embed_cache_streaming(EMBED_CACHE_PATH)
    if streamed is not None:
        return _normalize_embed_cache(streamed)
    try:
        return _normalize_embed_cache(json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return _empty_embed_cache()


def _load_embed_cache_impl(*, use_memo: bool) -> dict:
    global _EMBED_CACHE_MEMO_KEY, _EMBED_CACHE_MEMO_VALUE
    if not EMBED_CACHE_PATH.exists():
        if use_memo:
            _invalidate_embed_cache_memo()
        return _empty_embed_cache()
    memo_enabled = use_memo and EMBED_CACHE_MEMO_ENABLED
    key = None
    if memo_enabled:
        try:
            stat = EMBED_CACHE_PATH.stat()
            key = (str(EMBED_CACHE_PATH.resolve()), stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        if key is not None and key == _EMBED_CACHE_MEMO_KEY:
            return _EMBED_CACHE_MEMO_VALUE
    data = _load_embed_cache_from_disk()
    if memo_enabled and key is not None:
        _EMBED_CACHE_MEMO_KEY = key
        _EMBED_CACHE_MEMO_VALUE = data
    return data


def load_embed_cache() -> dict:
    """embed_cache.json を読み込む。破損・未存在時は空キャッシュで継続。

    version 1/2 は後段で version 不一致として安全に再構築される。
    read専用の呼び出し元向けに mtime/size をキーとした memo を持つ
    （``OFFLINE_AI_EMBED_CACHE_MEMO=0`` で無効化可能）。戻り値は memo 有効時
    共有参照になるため、呼び出し元は内容を変更してはならない。
    """
    return _load_embed_cache_impl(use_memo=True)


def save_embed_cache(cache: dict) -> bool:
    """embed_cache.json をアトミックに保存する（tmp 書き込み → os.replace）。

    書き込み途中の中断（Ctrl-C / 電源断 / ディスクフル）で既存キャッシュが
    破損するのを防ぐ。tmp は同一ディレクトリに作成し、失敗時は除去する。
    破損すると全 embedding の再計算が発生するため、本変更（モデル変更時の
    全再構築）と組み合わせて永続化の堅牢性を担保する。
    """
    if _atomic_write_json(EMBED_CACHE_PATH, cache):
        _invalidate_embed_cache_memo()
        return True
    print(
        "[WARN] embed_cache.json の保存に失敗しました。既存cacheとcheckpointを保持します。",
        file=sys.stderr,
    )
    return False


@dataclass(frozen=True)
class _EmbedIndex:
    """ノルム事前計算済みの read-only Embedding index。

    ``embedding_search_multi`` 専用の並走構造であり、``load_embed_cache`` が
    返す dict 契約は変更しない（N-EMBED-INDEX-TYPE 対応: 型はdictのまま据え置き、
    ノルムは別構造として保持する）。呼び出し元はいずれのフィールドも変更しないこと。
    """

    chunk_ids: list[str]
    entries: list[dict]
    vectors: list[list[float]]
    norms: list[float]


def _build_embed_index(cache: dict) -> _EmbedIndex:
    """cache の entries から一度だけノルムを計算し、多クエリ一括採点用の索引を作る。"""
    entries_map = cache.get("entries") if isinstance(cache.get("entries"), dict) else {}
    chunk_ids: list[str] = []
    entries: list[dict] = []
    vectors: list[list[float]] = []
    norms: list[float] = []
    for chunk_id, entry in entries_map.items():
        emb = entry.get("embedding") if isinstance(entry, dict) else None
        if not emb:
            continue
        chunk_ids.append(chunk_id)
        entries.append(entry)
        vectors.append(emb)
        norms.append(math.sqrt(sum(v * v for v in emb)))
    return _EmbedIndex(chunk_ids=chunk_ids, entries=entries, vectors=vectors, norms=norms)


def embedding_search_multi(
    queries: list[str], embed_model: str, index: _EmbedIndex, top_k: int = 8,
    *, raise_on_error: bool = False,
) -> dict[str, list]:
    """複数クエリのembeddingを1バッチで取得し、entriesを1パス走査して採点する。

    戻り値は ``{query: [match, ...]}``。各クエリの結果は ``embedding_search`` を
    個別に呼んだ場合と ``round(sim, 4)`` の桁で一致する（次元不一致entryのスキップ、
    閾値境界、top_k切り詰めを含む）。
    """
    unique_queries = list(dict.fromkeys(queries))
    if not unique_queries or not index.chunk_ids:
        return {query: [] for query in queries}
    try:
        query_vectors = _get_embeddings(unique_queries, embed_model)
    except EmbeddingBatchError as exc:
        if raise_on_error:
            raise
        print(f"[WARN] Embedding取得: {exc.code}", file=sys.stderr)
        return {query: [] for query in queries}

    query_norms = [math.sqrt(sum(v * v for v in vec)) for vec in query_vectors]
    scored_by_query: dict[str, list] = {query: [] for query in unique_queries}
    sidecar_cache: dict[str, dict | None] = {}

    for entry_index, chunk_id in enumerate(index.chunk_ids):
        emb = index.vectors[entry_index]
        entry_norm = index.norms[entry_index]
        entry = index.entries[entry_index]
        current_metadata = None
        metadata_loaded = False
        for query_index, query_vec in enumerate(query_vectors):
            if len(emb) != len(query_vec):
                continue
            query_norm = query_norms[query_index]
            if query_norm == 0.0 or entry_norm == 0.0:
                continue
            dot = sum(a * b for a, b in zip(query_vec, emb))
            sim = dot / (query_norm * entry_norm)
            if sim < EMBED_SIM_THRESHOLD:
                continue
            if not metadata_loaded:
                current_metadata = _current_sidecar_metadata_for_entry(entry, sidecar_cache)
                metadata_loaded = True
            text = entry.get("text", "")
            item = {
                "path": normalize_source_path(entry.get("path", chunk_id.split("#", 1)[0])),
                "chunk_id": entry.get("chunk_id", chunk_id),
                "heading": entry.get("heading", ""),
                "start_line": entry.get("start_line"),
                "end_line": entry.get("end_line"),
                "score": round(sim, 4),
                "embedding_score": round(sim, 4),
                "snippet": text[:SNIPPET_CHAR_LIMIT],
                "modifiedAt": entry.get("modifiedAt", ""),
                "source_sha256": entry.get("file_sha256", ""),
                "source": "embedding",
            }
            scored_by_query[unique_queries[query_index]].append(
                _apply_metadata_fields(item, current_metadata)
            )

    for query in unique_queries:
        scored_by_query[query].sort(key=lambda x: x["score"], reverse=True)
        scored_by_query[query] = scored_by_query[query][:top_k]
    return {query: scored_by_query[query] for query in queries}


INDEX_STATES = {
    "missing",
    "stale",
    "building",
    "paused",
    "ready",
    "cancelled",
    "failed",
    "cancelling",
}

_INDEX_STATUS_ACTIVE_STATES = {"building", "cancelling"}
_INDEX_STATUS_TERMINAL_STATES = {"ready", "cancelled", "failed", "stale", "missing"}


def _index_status_transition_allowed(
    current: dict,
    candidate: dict,
    *,
    allow_restart: bool,
    allow_generation_change: bool,
) -> bool:
    """古いworkerが新しい状態を巻き戻さないための保存前検証。"""
    if not current:
        return True

    current_state = current.get("state")
    candidate_state = candidate.get("state")
    current_active = current_state in _INDEX_STATUS_ACTIVE_STATES

    # restartは、現在のjobがactiveでない場合だけjob/generationを切り替えられる。
    # activeな別workerを、開始直後のstatusで上書きすることは許可しない。
    if allow_restart and not current_active:
        return candidate_state == "building"

    current_job = current.get("job_id")
    candidate_job = candidate.get("job_id")
    if current_job and candidate_job and current_job != candidate_job:
        return False
    current_generation = current.get("generation")
    candidate_generation = candidate.get("generation")
    if (
        current_generation
        and candidate_generation
        and current_generation != candidate_generation
        and not allow_generation_change
    ):
        return False

    if current_state == "cancelling" and candidate_state not in {
        "cancelling",
        "cancelled",
    }:
        return False
    if current_state in _INDEX_STATUS_TERMINAL_STATES and candidate_state != current_state:
        return False
    if current_state == "building" and candidate_state == "building":
        old_processed = current.get("processed")
        new_processed = candidate.get("processed")
        if isinstance(old_processed, int) and isinstance(new_processed, int):
            if new_processed < old_processed:
                return False
    return True


def load_index_status() -> dict:
    """Read the redacted, user-local index status snapshot."""
    if not EMBED_STATUS_PATH.exists() or EMBED_STATUS_PATH.is_symlink():
        return {}
    try:
        data = json.loads(EMBED_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Status is deliberately an allowlist: no query, source body, vector, or path.
    allowed = {
        "state",
        "job_id",
        "generation",
        "embed_model",
        "total",
        "processed",
        "generated",
        "reused",
        "failed",
        "checkpointed",
        "elapsed_seconds",
        "rate_per_second",
        "eta_seconds",
        "error_code",
        "cancel_requested",
        "updated_at",
    }
    return {key: value for key, value in data.items() if key in allowed}


def save_index_status(status: dict) -> bool:
    """Atomically persist a redacted index status snapshot.

    ``_status_guard`` is an internal, non-persisted field used by
    ``IndexCoordinator``.  Direct callers without that field retain the old
    write contract; coordinated callers get a read/validate/replace critical
    section protected across processes.
    """
    guard = status.get("_status_guard")
    safe = {
        key: value
        for key, value in status.items()
        if key
        in {
            "state",
            "job_id",
            "generation",
            "embed_model",
            "total",
            "processed",
            "generated",
            "reused",
            "failed",
            "checkpointed",
            "elapsed_seconds",
            "rate_per_second",
            "eta_seconds",
            "error_code",
            "cancel_requested",
            "updated_at",
        }
    }
    state = safe.get("state")
    if state not in INDEX_STATES:
        raise ValueError(f"invalid index state: {state}")
    if not isinstance(guard, dict):
        return _atomic_write_json(EMBED_STATUS_PATH, safe)

    with _index_status_lock() as locked:
        if not locked:
            return False
        current = load_index_status()
        if not _index_status_transition_allowed(
            current,
            safe,
            allow_restart=bool(guard.get("allow_restart")),
            allow_generation_change=bool(guard.get("allow_generation_change")),
        ):
            raise IndexStatusConflictError("stale index status transition")
        return _atomic_write_json(EMBED_STATUS_PATH, safe)


def get_embed_index_status(
    embed_model: str | None = None,
    chunks: list[dict] | None = None,
    *,
    cache: dict | None = None,
    validate_entries: bool = True,
) -> dict:
    """Return ready/stale/building state without starting an index build.

    ``cache`` が渡された場合は ``load_embed_cache()`` を呼ばない（呼び出し元が
    既に1回ロード済みのキャッシュを渡す1-a最適化）。``validate_entries=False``
    は header 一致＋key集合一致までで ready 判定し、18,284件規模の全件検証
    （``_valid_embedding_entry``）を省略する。build完了直後の検証には
    必ず ``validate_entries=True``（既定）を使うこと。
    """
    model = embed_model or detect_embed_model()
    persisted = load_index_status()
    if not model:
        return {"state": "missing", "total": 0, "processed": 0, "generation": None}
    if chunks is not None:
        source_chunks, chunk_memo_key = chunks, None
    else:
        # status は UI から繰り返し呼ばれる read 経路であり、毎回の
        # 482 ファイル走査（26 MB の読み込みと sha256）が p95 を支配する。
        source_chunks, chunk_memo_key = load_source_chunks_cached(SKILL_SOURCE_DIR)
    generation = compute_embed_generation_cached(model, source_chunks, chunk_memo_key)
    resolved_cache = cache if cache is not None else load_embed_cache()
    entries = (
        resolved_cache.get("entries") if isinstance(resolved_cache.get("entries"), dict) else {}
    )
    cache_ready = _cache_header_matches(resolved_cache, model, generation) and set(entries) == {
        chunk.get("chunk_id") for chunk in source_chunks
    }
    entries_valid = (
        all(
            _valid_embedding_entry(entries.get(chunk.get("chunk_id")), chunk.get("chunk_id"))
            for chunk in source_chunks
        )
        if validate_entries
        else cache_ready
    )
    if cache_ready and entries_valid:
        return {
            "state": "ready",
            "embed_model": _canonical_embed_model(model),
            "generation": generation,
            "total": len(source_chunks),
            "processed": len(source_chunks),
            "generated": len(source_chunks),
            "reused": 0,
            "failed": 0,
            "checkpointed": 0,
        }
    if persisted.get("generation") == generation and persisted.get("state") in {
        "building",
        "cancelling",
        "paused",
        "cancelled",
        "failed",
    }:
        return {
            **persisted,
            "embed_model": _canonical_embed_model(model),
            "generation": generation,
            "total": len(source_chunks),
        }
    state = "stale" if EMBED_CACHE_PATH.exists() else "missing"
    return {
        "state": state,
        "embed_model": _canonical_embed_model(model),
        "generation": generation,
        "total": len(source_chunks),
        "processed": 0,
        "generated": 0,
        "reused": 0,
        "failed": 0,
        "checkpointed": 0,
    }


def build_or_update_embed_index(
    embed_model: str,
    chunks: list[dict] | None = None,
    *,
    emit_progress=None,
    cancel_check=None,
) -> dict:
    """skill-source 配下をチャンク走査し、変更分のみembeddingを再計算してキャッシュ返却。

    キャッシュ生成時の embed_model と現在のモデルが異なる場合は、ベクトル次元の
    不整合（例: nomic 768次元 → bge-m3 1024次元）を防ぐため全エントリを破棄して
    再構築する。version 2 以前のファイル単位キャッシュも再構築する。
    """
    source_chunks = (
        chunks if chunks is not None else build_source_chunks(SKILL_SOURCE_DIR)
    )
    total = len(source_chunks)
    generation = compute_embed_generation(embed_model, source_chunks)

    def check_cancel() -> None:
        if cancel_check:
            cancel_check()

    with _embed_writer_lock() as lock_acquired:
        if not lock_acquired:
            existing = _load_embed_cache_impl(use_memo=False)
            if EMBED_CACHE_PATH.exists() and _cache_header_matches(
                existing, embed_model, generation
            ):
                _emit_embed_progress(
                    emit_progress,
                    phase="progress",
                    processed=0,
                    total=total,
                    generated=0,
                    reused=0,
                    failed=0,
                    checkpointed=0,
                )
                return existing
            raise EmbedIndexBusyError("Embeddingインデックスを更新中です")

        check_cancel()
        cache = _load_embed_cache_impl(use_memo=False)
        cached_model = cache.get("embed_model")
        cached_version = cache.get("version")
        cache_header_matches = _cache_header_matches(cache, embed_model, generation)
        if not cache_header_matches:
            if cached_model is not None:
                print(
                    "[INFO] Embeddingキャッシュの再構築が必要です "
                    f"(version {cached_version} → {EMBED_CACHE_VERSION}, "
                    f"model '{cached_model}' → '{embed_model}')。",
                    file=sys.stderr,
                )
            cache = _empty_embed_cache()
            cache["embed_model"] = embed_model
        else:
            cache["embed_model"] = embed_model
        cache["generation"] = generation

        state, checkpoint_entries = _load_checkpoint(embed_model, generation)
        final_entries = (
            cache.get("entries", {}) if isinstance(cache.get("entries"), dict) else {}
        )
        current_ids = {chunk["chunk_id"] for chunk in source_chunks}
        reusable: dict[str, dict] = {}
        valid_checkpoint_count = 0
        for chunk in source_chunks:
            chunk_id = chunk["chunk_id"]
            candidates = [checkpoint_entries.get(chunk_id), final_entries.get(chunk_id)]
            for candidate in candidates:
                if (
                    _valid_embedding_entry(candidate, chunk_id)
                    and candidate.get("file_sha256") == chunk.get("file_sha256")
                    and candidate.get("text_sha256") == chunk.get("text_sha256")
                ):
                    reusable[chunk_id] = _strip_cache_metadata(candidate)
                    if candidate is checkpoint_entries.get(chunk_id):
                        valid_checkpoint_count += 1
                    break

        entries: dict[str, dict] = dict(reusable)
        processed = len(reusable)
        generated = 0
        failed = 0
        checkpointed = valid_checkpoint_count
        last_progress_processed = processed
        last_progress_time = time.monotonic()
        pending_batch: list[dict] = []

        _emit_embed_progress(
            emit_progress,
            phase="resumed" if valid_checkpoint_count else "start",
            processed=processed,
            total=total,
            generated=generated,
            reused=processed,
            failed=failed,
            checkpointed=checkpointed,
        )

        def emit_progress_if_due(force: bool = False) -> None:
            nonlocal last_progress_processed, last_progress_time
            now = time.monotonic()
            if force or (
                processed - last_progress_processed >= EMBED_PROGRESS_INTERVAL
                or now - last_progress_time >= EMBED_PROGRESS_SECONDS
            ):
                _emit_embed_progress(
                    emit_progress,
                    phase="progress",
                    processed=processed,
                    total=total,
                    generated=generated,
                    reused=len(reusable),
                    failed=failed,
                    checkpointed=checkpointed,
                )
                last_progress_processed = processed
                last_progress_time = now

        def flush_pending(*, force: bool = False) -> None:
            nonlocal checkpointed, pending_batch
            if not pending_batch or (
                not force and len(pending_batch) < EMBED_CHECKPOINT_INTERVAL
            ):
                return
            batch = list(pending_batch)
            if not _persist_checkpoint_batch(state, batch):
                print(
                    "[WARN] Embedding checkpointの保存に失敗しました。",
                    file=sys.stderr,
                )
                return
            checkpointed += len(batch)
            pending_batch = pending_batch[len(batch) :]
            _emit_embed_progress(
                emit_progress,
                phase="checkpoint",
                processed=processed,
                total=total,
                generated=generated,
                reused=len(reusable),
                failed=failed,
                checkpointed=checkpointed,
            )

        try:
            pending_chunks: list[dict] = []
            # Legacy monkeypatches in downstream tests retain the old single
            # item seam.  Normal runtime always takes the array API path.
            effective_batch_size = (
                1 if _get_embedding is not _ORIGINAL_GET_EMBEDDING else EMBED_BATCH_SIZE
            )

            def process_batch(batch_chunks: list[dict]) -> None:
                nonlocal processed, generated, failed
                if not batch_chunks:
                    return
                check_cancel()
                vectors = _get_index_embedding_batch(
                    [chunk["text"] for chunk in batch_chunks], embed_model
                )
                check_cancel()
                for chunk, embedding in zip(batch_chunks, vectors):
                    chunk_id = chunk["chunk_id"]
                    processed += 1
                    if embedding is not None and _valid_embedding_entry(
                        {**_strip_cache_metadata(chunk), "embedding": embedding}, chunk_id
                    ):
                        entry = {**_strip_cache_metadata(chunk), "embedding": embedding}
                        entries[chunk_id] = entry
                        pending_batch.append(entry)
                        generated += 1
                    else:
                        entries.pop(chunk_id, None)
                        failed += 1
                    flush_pending()
                    emit_progress_if_due()

            for chunk in source_chunks:
                if chunk["chunk_id"] in reusable:
                    continue
                pending_chunks.append(chunk)
                if len(pending_chunks) >= effective_batch_size:
                    process_batch(pending_chunks)
                    pending_chunks = []
            process_batch(pending_chunks)

            # 正常完了時も端数batchをdurableにしてから最終cacheを確定する。
            flush_pending(force=True)
            if failed:
                raise EmbedBuildError(
                    "EMBED_BATCH_FAILED",
                    f"{failed} embedding item(s) failed; final cache was not promoted",
                )
            for rel in list(entries.keys()):
                if rel not in current_ids:
                    del entries[rel]
            cache["entries"] = entries
            cache["version"] = EMBED_CACHE_VERSION
            cache["embed_model"] = embed_model
            cache["generation"] = generation
            cache["chunking"] = {
                "max_chars": CHUNK_MAX_CHARS,
                "overlap_chars": CHUNK_OVERLAP_CHARS,
            }
            if not save_embed_cache(cache):
                raise EmbedCachePersistenceError(
                    "最終Embedding cacheの保存に失敗しました。checkpointを保持します。"
                )
            verified = _load_embed_cache_impl(use_memo=False)
            if (
                _canonical_embed_model(verified.get("embed_model"))
                != _canonical_embed_model(embed_model)
                or verified.get("version") != EMBED_CACHE_VERSION
                or verified.get("generation") != generation
                or verified.get("chunking")
                != {
                    "max_chars": CHUNK_MAX_CHARS,
                    "overlap_chars": CHUNK_OVERLAP_CHARS,
                }
                or set(verified.get("entries", {})) != set(entries)
                or any(
                    not _valid_embedding_entry(verified["entries"].get(chunk_id), chunk_id)
                    or verified["entries"][chunk_id].get("file_sha256")
                    != entries[chunk_id].get("file_sha256")
                    or verified["entries"][chunk_id].get("text_sha256")
                    != entries[chunk_id].get("text_sha256")
                    for chunk_id in entries
                )
            ):
                raise EmbedCachePersistenceError(
                    "最終Embedding cacheの検証に失敗しました。checkpointを保持します。"
                )
            _remove_checkpoint_files(include_state=True)
            _emit_embed_progress(
                emit_progress,
                phase="completed",
                processed=processed,
                total=total,
                generated=generated,
                reused=len(reusable),
                failed=failed,
                checkpointed=checkpointed,
            )
            return cache
        except BaseException:
            # Ctrl+C、Web cancel、timeout、予期しない例外のいずれでも、
            # 直前の成功entryを可能な範囲で確定してから元の例外を再送出する。
            try:
                flush_pending(force=True)
            finally:
                _emit_embed_progress(
                    emit_progress,
                    phase="interrupted",
                    processed=processed,
                    total=total,
                    generated=generated,
                    reused=len(reusable),
                    failed=failed,
                    checkpointed=checkpointed,
                )
            raise


def embedding_search(query: str, embed_model: str, cache: dict, top_k: int = 8) -> list:
    """クエリのembeddingとキャッシュを比較し、類似度上位を返す。"""
    query_vec = _get_embedding(query, embed_model)
    if query_vec is None:
        return []

    entries = cache.get("entries", {})
    scored = []
    sidecar_cache: dict[str, dict | None] = {}
    for chunk_id, entry in entries.items():
        emb = entry.get("embedding")
        if not emb:
            continue
        if len(emb) != len(query_vec):
            # 次元不一致（モデル変更の過渡期・破損）は誤スコアを避けてスキップ
            continue
        sim = _cosine_similarity(query_vec, emb)
        if sim >= EMBED_SIM_THRESHOLD:
            text = entry.get("text", "")
            item = {
                "path": normalize_source_path(
                    entry.get("path", chunk_id.split("#", 1)[0])
                ),
                "chunk_id": entry.get("chunk_id", chunk_id),
                "heading": entry.get("heading", ""),
                "start_line": entry.get("start_line"),
                "end_line": entry.get("end_line"),
                "score": round(sim, 4),
                "embedding_score": round(sim, 4),
                "snippet": text[:SNIPPET_CHAR_LIMIT],
                "modifiedAt": entry.get("modifiedAt", ""),
                "source_sha256": entry.get("file_sha256", ""),
                "source": "embedding",
            }
            scored.append(
                _apply_metadata_fields(
                    item, _current_sidecar_metadata_for_entry(entry, sidecar_cache)
                )
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _split_terms(text: str) -> list[str]:
    terms = []
    for term in re.split(r"[\s,、。・/\\]+", text):
        cleaned = term.strip().lower()
        if len(cleaned) >= 2:
            terms.append(cleaned)
    return list(dict.fromkeys(terms))


def _query_ngrams(query: str) -> list[str]:
    compact = re.sub(r"\s+", "", query)
    if len(compact) < 3:
        return []
    grams = []
    for size in range(3, min(8, len(compact)) + 1):
        for idx in range(0, len(compact) - size + 1):
            grams.append(compact[idx : idx + size].lower())
    return list(dict.fromkeys(grams))


def keyword_search_chunks(
    query: str,
    keywords: str | list[str] | None = None,
    top_k: int = 8,
    *,
    chunks: list[dict] | None = None,
) -> list:
    """Python 標準ライブラリだけでチャンク単位キーワード検索を行う。"""
    if isinstance(keywords, str):
        terms = _split_terms(keywords)
    elif keywords:
        terms = [str(x).strip().lower() for x in keywords if str(x).strip()]
    else:
        terms = []
    terms = list(dict.fromkeys(terms + _split_terms(query)))
    if not terms:
        terms = [query.strip().lower()] if query.strip() else []
    phrase_terms = _query_ngrams(query)

    source_chunks = (
        chunks if chunks is not None else build_source_chunks(SKILL_SOURCE_DIR)
    )
    if not source_chunks:
        return []

    df: dict[str, int] = {term: 0 for term in terms}
    for chunk in source_chunks:
        haystack = " ".join(
            [
                chunk.get("path", ""),
                chunk.get("heading", ""),
                chunk.get("text", ""),
            ]
        ).lower()
        for term in terms:
            if term in haystack:
                df[term] += 1

    scored = []
    total = max(len(source_chunks), 1)
    for chunk in source_chunks:
        text = chunk.get("text", "")
        haystack = " ".join(
            [chunk.get("path", ""), chunk.get("heading", ""), text]
        ).lower()
        score = 0.0
        hit_terms = []
        for term in terms:
            count = haystack.count(term)
            if count <= 0:
                continue
            hit_terms.append(term)
            idf = math.log((total + 1) / (df.get(term, 0) + 1)) + 1.0
            score += count * idf
            if term in chunk.get("heading", "").lower():
                score += 3.0
            if term in Path(chunk.get("path", "")).stem.lower():
                score += 2.0
        for phrase in phrase_terms:
            if phrase in haystack:
                score += 0.4
        if score <= 0:
            continue
        snippet = text[:SNIPPET_CHAR_LIMIT]
        scored.append(
            {
                "path": chunk["path"],
                "chunk_id": chunk["chunk_id"],
                "heading": chunk.get("heading", ""),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "score": round(score, 4),
                "keyword_score": round(score, 4),
                "snippet": snippet,
                "modifiedAt": chunk.get("modifiedAt", ""),
                "hitKeywords": hit_terms,
                "source": "keyword",
            }
            | {
                key: value
                for key, value in {
                    "metadata": chunk.get("metadata"),
                    "parser": chunk.get("parser"),
                    "page": chunk.get("page"),
                    "bbox": chunk.get("bbox"),
                    "layout_type": chunk.get("layout_type"),
                    "source_title": chunk.get("source_title"),
                    "confidence": chunk.get("confidence"),
                    "block_id": chunk.get("block_id"),
                    "parser_warnings": chunk.get("parser_warnings"),
                    "source_sha256": chunk.get("file_sha256", ""),
                }.items()
                if value is not None
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def merge_results(
    keyword_matches: list, embed_matches: list, max_results: int = 8
) -> list:
    """RRF (Reciprocal Rank Fusion) で2つのリストを統合する。定数 k=60。"""
    rrf_scores: dict[str, float] = {}
    path_to_item: dict[str, dict] = {}

    def add(rank: int, item: dict) -> None:
        normalized = dict(item)
        path = normalize_source_path(normalized.get("path", ""))
        if not path:
            return
        normalized["path"] = path
        key = normalized.get("chunk_id") or path
        if normalized.get("chunk_id"):
            normalized["chunk_id"] = str(normalized["chunk_id"]).replace("\\", "/")
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 60)
        existing = path_to_item.get(key)
        if existing is None:
            path_to_item[key] = normalized
            return
        if not existing.get("snippet") and normalized.get("snippet"):
            existing["snippet"] = normalized["snippet"]
        if not existing.get("modifiedAt") and normalized.get("modifiedAt"):
            existing["modifiedAt"] = normalized["modifiedAt"]
        for metadata_key in (
            "metadata",
            "parser",
            "page",
            "bbox",
            "layout_type",
            "source_title",
            "confidence",
            "block_id",
            "parser_warnings",
            "source_sha256",
            "keyword_score",
            "embedding_score",
        ):
            value = normalized.get(metadata_key)
            if value is None:
                continue
            if metadata_key in {"keyword_score", "embedding_score"}:
                existing[metadata_key] = max(
                    float(existing.get(metadata_key, 0)), float(value)
                )
            elif existing.get(metadata_key) is None:
                existing[metadata_key] = value
        existing["source"] = "+".join(
            sorted(
                set(str(existing.get("source", "")).split("+"))
                | {str(normalized.get("source", ""))}
            )
        ).strip("+")

    for rank, item in enumerate(keyword_matches):
        add(rank, item)

    for rank, item in enumerate(embed_matches):
        add(rank, item)

    sorted_paths = sorted(rrf_scores.keys(), key=lambda p: rrf_scores[p], reverse=True)
    results = []
    for p in sorted_paths[:max_results]:
        item = path_to_item[p]
        item["rrf_score"] = rrf_scores[p]
        results.append(item)
    return results


def filter_by_rrf_score(matches: list, min_score: float) -> list:
    """RRF統合スコアで低関連度の結果を除外する。

    min_score が 0 以下の場合はフィルタを無効化し全件返却する。
    これにより環境変数 SEARCH_MIN_RRF_SCORE=0 でロールバック可能。
    """
    if min_score <= 0:
        return matches
    return [m for m in matches if m.get("rrf_score", 0) >= min_score]


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def create_search_plan(query: str, model: str) -> dict:
    """検索計画JSONをLLMに要求する。失敗時は従来キーワード抽出へ戻す。"""
    if not _agentic_lite_enabled():
        keywords = extract_keywords_via_llm(query, model)
        return {
            "keywords": _split_terms(keywords),
            "search_queries": [query],
            "must_find_terms": [],
            "query_type": "standard",
            "answer_should_compare": False,
            "fallback": "agentic_disabled",
        }

    url = f"{OLLAMA_HOST}/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SEARCH_PLAN_SYSTEM},
            {"role": "user", "content": search_plan_prompt(query)},
        ],
        "stream": False,
        # 検索計画は短いJSONを返す補助処理であり、thinking tokenは不要。
        # Qwen3.5等で既定thinkingが有効だと30秒timeout内にモデル出力へ
        # 到達できないため、明示的に無効化する。
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        # 補助chatも回答chatと同じcontext/load契約を明示する。Ollamaが
        # 4096等のモデル既定へ戻ると、keep_aliveが有効でもchat間で再loadが
        # 発生し得るため、load_durationとVRAMを同一条件で比較できるようにする。
        "options": build_chat_options(temperature=0),
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=KEYWORD_EXTRACT_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parsed = _parse_json_object(data.get("message", {}).get("content", ""))
        if parsed:
            keywords = parsed.get("keywords", [])
            if isinstance(keywords, str):
                keywords = _split_terms(keywords)
            queries = parsed.get("search_queries", [])
            if isinstance(queries, str):
                queries = [queries]
            parsed["keywords"] = [str(x) for x in keywords][:8]
            parsed["search_queries"] = [str(x) for x in queries if str(x).strip()][
                :MAX_QUERY_VARIANTS
            ]
            if not parsed["search_queries"]:
                parsed["search_queries"] = [query]
            return parsed
    except Exception:
        # search plan が timeout / transport error になった直後に同じ chat model へ
        # keyword抽出を再送すると、失敗時の待ち時間が最大2倍になる。元クエリを使う
        # 決定的fallbackへ直行し、bounded retrieval の上限を維持する。
        return {
            "keywords": [query],
            "search_queries": [query],
            "must_find_terms": [],
            "query_type": "standard",
            "answer_should_compare": False,
            "fallback": "plan_request_failed",
        }

    return {
        "keywords": [query],
        "search_queries": [query],
        "must_find_terms": [],
        "query_type": "standard",
        "answer_should_compare": False,
        "fallback": "plan_parse_failed",
    }


def _limit_chunks_per_file(
    matches: list[dict], max_per_file: int = MAX_CHUNKS_PER_FILE
) -> list[dict]:
    counts: dict[str, int] = {}
    limited = []
    for match in matches:
        path = normalize_source_path(match.get("path", ""))
        count = counts.get(path, 0)
        if count >= max_per_file:
            continue
        counts[path] = count + 1
        limited.append(match)
    return limited


class PromptBudgetError(RuntimeError):
    """prompt 予算が生成分を確保できず、本文生成前に fail closed した。

    根拠を強制挿入すると reserve を侵食し、done_reason: length で本文が
    0 文字になるため、無言で劣化させず構成エラーとして扱う。
    """


def _estimate_prompt_shell_chars(
    query: str, *, attempts: list[dict], evidence_status: str, confidence: float
) -> int:
    """根拠を除いた prompt の文字数（system prompt を含む）を見積もる。

    予算逆算が無効（reserve 0）なら呼ばれない。テンプレート生成に失敗した
    場合は質問文の長さへ退避し、逆算そのものは継続する。
    """
    try:
        shell = build_user_prompt(
            query,
            [],
            attempts=attempts,
            evidence_status=evidence_status,
            confidence=confidence,
        )
    except Exception:
        shell = query
    return len(SYSTEM_PROMPT) + len(shell)


def compute_evidence_char_limit(
    prompt_shell_chars: int,
    *,
    num_ctx: int | None = None,
    reserve_tokens: int | None = None,
    chars_per_token: float | None = None,
    safety_tokens: int | None = None,
    char_limit: int | None = None,
) -> int:
    """根拠 snippet に使える実効文字上限を返す。

    ``reserve_tokens`` が 0 のときは逆算を行わず ``char_limit`` をそのまま
    返す（現行挙動）。0 より大きいときは context ウィンドウから生成分の
    予備と prompt shell のオーバーヘッドを差し引き、``char_limit`` との
    小さい方を採る。差し引いた残量が 0 以下なら ``PromptBudgetError``。

    ``prompt_shell_chars`` は根拠を除いた prompt（system prompt、指示
    テンプレート、質問文、区切り）の文字数である。
    """
    limit = PROMPT_EVIDENCE_CHAR_LIMIT if char_limit is None else char_limit
    reserve = GENERATION_RESERVE_TOKENS if reserve_tokens is None else reserve_tokens
    if reserve <= 0:
        return limit

    ctx = NUM_CTX if num_ctx is None else num_ctx
    per_token = PROMPT_CHARS_PER_TOKEN if chars_per_token is None else chars_per_token
    safety = PROMPT_BUDGET_SAFETY_TOKENS if safety_tokens is None else safety_tokens
    overhead_tokens = math.ceil(max(0, prompt_shell_chars) / per_token) + safety
    evidence_token_budget = ctx - reserve - overhead_tokens
    if evidence_token_budget <= 0:
        raise PromptBudgetError(
            f"prompt 予算が不足しています（num_ctx={ctx}, 生成予備={reserve} tokens, "
            f"prompt overhead={overhead_tokens} tokens）。"
            "OLLAMA_NUM_CTX を大きくするか OFFLINE_AI_GENERATION_RESERVE_TOKENS を"
            "小さくしてください。"
        )
    return min(limit, math.floor(evidence_token_budget * per_token))


def _fit_prompt_budget(
    matches: list[dict], char_limit: int = PROMPT_EVIDENCE_CHAR_LIMIT
) -> list[dict]:
    used = 0
    fitted = []
    for match in matches:
        item = dict(match)
        snippet = item.get("snippet", "") or ""
        remaining = char_limit - used
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining]
            item["snippet"] = snippet
        used += len(snippet)
        fitted.append(item)
    return fitted


COVERAGE_NGRAM_SIZE = 2
_CJK_PATTERN = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")
_QUERY_BOILERPLATE = (
    "教えてください",
    "教えて下さい",
    "何ですか",
    "いくらですか",
    "どうすればよいですか",
)
_EXPLICIT_CONFLICT_MARKERS = (
    "対象外",
    "適用しない",
    "適用されない",
    "含まない",
    "定めない",
    "支給しない",
    "記載がない",
    "記載なし",
)


def _compact_relevance_text_raw(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower(), flags=re.UNICODE)


def _compact_relevance_text(text: str) -> str:
    """具体語の連続一致に使う、句読点と質問定型句を除いた文字列を返す。"""
    compact = _compact_relevance_text_raw(text)
    for phrase in _QUERY_BOILERPLATE:
        compact = compact.replace(_compact_relevance_text_raw(phrase), "")
    return compact


def _lexical_anchor_length(query: str, match: dict, max_chars: int = 12) -> int:
    """質問と候補が共有する最長の具体的な連続文字列長を返す。"""
    needle = _compact_relevance_text(query)
    haystack = _compact_relevance_text(
        " ".join(
            str(match.get(key, "") or "") for key in ("path", "heading", "snippet")
        )
    )
    upper = min(max_chars, len(needle))
    for size in range(upper, LEXICAL_ANCHOR_MIN_CHARS - 1, -1):
        if any(
            needle[index : index + size] in haystack
            for index in range(len(needle) - size + 1)
        ):
            return size
    return 0


def _has_relevance_support(query: str, match: dict) -> bool:
    """弱いEmbedding近接だけの候補を除外し、同義語候補は強い類似度で残す。"""
    if _lexical_anchor_length(query, match) >= LEXICAL_ANCHOR_MIN_CHARS:
        return True
    embedding_score = float(match.get("embedding_score", 0) or 0)
    if not embedding_score and str(match.get("source", "")) == "embedding":
        embedding_score = float(match.get("score", 0) or 0)
    return embedding_score >= SEMANTIC_SUPPORT_THRESHOLD


def filter_by_relevance_support(query: str, matches: list[dict]) -> list[dict]:
    """質問固有の語または強い意味類似性を持つ候補だけを残す。"""
    return [match for match in matches if _has_relevance_support(query, match)]


def _has_explicit_constraint_conflict(query: str, matches: list[dict]) -> bool:
    """質問の限定条件が根拠中で明示的に否定されているかを判定する。"""
    compact_query = _compact_relevance_text(query)
    if len(compact_query) < LEXICAL_ANCHOR_MIN_CHARS:
        return False
    query_anchors = {
        compact_query[index : index + LEXICAL_ANCHOR_MIN_CHARS]
        for index in range(len(compact_query) - LEXICAL_ANCHOR_MIN_CHARS + 1)
    }
    for match in matches:
        evidence = " ".join(
            str(match.get(key, "") or "") for key in ("heading", "snippet")
        )
        for sentence in re.split(r"[。！？\n]+", evidence):
            if not any(marker in sentence for marker in _EXPLICIT_CONFLICT_MARKERS):
                continue
            compact_sentence = _compact_relevance_text_raw(sentence)
            if any(anchor in compact_sentence for anchor in query_anchors):
                return True
    return False


def coverage_terms(query: str, must_find_terms: list[str] | None = None) -> list[str]:
    """confidence の coverage 判定に使う語を返す。

    `_split_terms` は空白・句読点でしか分割しないため、日本語のように語間へ区切りを
    置かない言語では query 全体が1語となり、本文へ逐語一致することがまずない。その
    結果 coverage が常に 0 となり、confidence が構造的信号だけで決まっていた。
    CJK を含む語は `COVERAGE_NGRAM_SIZE` 文字の n-gram へ展開して部分一致を測る。

    `must_find_terms` は検索計画が抽出した精密語であり、展開せず逐語で評価する。
    """
    terms: list[str] = []
    for term in _split_terms(query):
        compact = re.sub(r"\s+", "", term)
        if _CJK_PATTERN.search(compact) and len(compact) > COVERAGE_NGRAM_SIZE:
            terms.extend(
                compact[i : i + COVERAGE_NGRAM_SIZE]
                for i in range(len(compact) - COVERAGE_NGRAM_SIZE + 1)
            )
        else:
            terms.append(term)
    terms.extend(str(x).lower() for x in (must_find_terms or []) if str(x).strip())
    return list(dict.fromkeys(terms))


def _calculate_confidence(
    query: str, matches: list[dict], must_find_terms: list[str] | None = None
) -> tuple[float, str]:
    if not matches:
        return 0.0, "insufficient"
    text = " ".join(
        (m.get("snippet", "") or "") + " " + m.get("heading", "") for m in matches
    ).lower()
    terms = coverage_terms(query, must_find_terms)
    coverage = 0.0
    if terms:
        coverage = sum(1 for term in terms if term in text) / len(terms)
    non_empty = sum(1 for m in matches if m.get("snippet")) / max(len(matches), 1)
    top_score = float(matches[0].get("rrf_score", matches[0].get("score", 0)) or 0)
    file_count = len({normalize_source_path(m.get("path", "")) for m in matches})
    confidence = min(
        1.0,
        (coverage * 0.45)
        + (non_empty * 0.25)
        + (min(top_score / 0.035, 1.0) * 0.2)
        + (min(file_count / 2, 1.0) * 0.1),
    )
    # 語の重なりだけでは、質問の限定条件（例: 「海外」出張）を満たさない資料を
    # 十分な根拠として区別できない。keyword一致のみの根拠でsufficientを宣言すると、
    # promptの根拠不足警告が外れ、該当情報なしを維持できなくなる。sufficientは
    # Embedding由来の根拠を伴う場合に限り、それ以外はpartial止まりとする。
    has_semantic_evidence = any(
        "embedding" in str(m.get("source", "")) for m in matches
    )
    if len(matches) >= 2 and confidence >= 0.55 and has_semantic_evidence:
        return round(confidence, 3), "sufficient"
    if confidence >= 0.28:
        return round(confidence, 3), "partial"
    return round(confidence, 3), "insufficient"


def _legacy_keyword_matches(query: str, original_query: str) -> list:
    source_data = collect_sources(query, original_query=original_query)
    matches = extract_matches_with_logging(source_data)
    for match in matches:
        if "idfScore" in match:
            match["score"] = match["idfScore"]
    return matches


def _run_single_retrieval(
    query: str,
    keywords: list[str],
    embed_model: str | None,
    embed_cache: dict | None = None,
    source_chunks: list[dict] | None = None,
    *,
    precomputed_embed_matches: list[dict] | None = None,
) -> list[dict]:
    """1クエリ分の keyword+embedding 検索を実行する。

    ``precomputed_embed_matches`` が渡された場合（多クエリ一括採点の結果）は
    ``embedding_search`` を呼ばずそれを使う。渡されなければ従来どおり単一
    クエリの ``embedding_search`` を呼ぶ（CLI / 既存テスト互換の経路）。
    """
    candidate_limit = (
        RERANK_CONFIG.top_n
        if RERANK_CONFIG.mode != "off"
        else RETRIEVAL_CANDIDATE_LIMIT
    )
    keyword_text = " ".join(keywords) if keywords else query
    if _chunk_retrieval_enabled():
        keyword_matches = keyword_search_chunks(
            query,
            keyword_text,
            top_k=candidate_limit,
            chunks=source_chunks,
        )
    else:
        keyword_matches = _legacy_keyword_matches(keyword_text, query)

    embed_matches: list[dict] = []
    if precomputed_embed_matches is not None:
        embed_matches = precomputed_embed_matches
    elif embed_model:
        try:
            cache = (
                embed_cache
                if embed_cache is not None
                else build_or_update_embed_index(embed_model, source_chunks)
            )
            embed_matches = embedding_search(
                query, embed_model, cache, top_k=candidate_limit
            )
        except Exception as e:
            logger.warning("Embedding search failed: %s", e)
    return merge_results(keyword_matches, embed_matches, max_results=candidate_limit)


class _RerankCancellation(Exception):
    def __init__(self, cause: Exception):
        super().__init__(str(cause))
        self.cause = cause


def _rerank_with_fallback(
    query: str,
    candidates: list[dict],
    *,
    plan: dict,
    config: RerankConfig,
    cancel_check=None,
) -> list[dict]:
    """Apply rerank without swallowing cancellation or losing the RRF order."""

    def guarded_cancel_check() -> None:
        try:
            cancel_check()
        except Exception as exc:
            raise _RerankCancellation(exc) from exc

    try:
        return rerank_candidates(
            query,
            candidates,
            plan=plan,
            config=config,
            cancel_check=guarded_cancel_check if cancel_check else None,
        )
    except _RerankCancellation as exc:
        raise exc.cause from exc
    except Exception as exc:
        logger.warning("Rerank failed; using RRF order: %s", exc)
        if config.strict:
            raise
        return candidates


def _retry_queries(query: str, plan: dict, matches: list[dict]) -> list[str]:
    seen_text = " ".join(m.get("snippet", "") for m in matches).lower()
    missing = [
        str(term)
        for term in plan.get("must_find_terms", [])
        if str(term).strip() and str(term).lower() not in seen_text
    ]
    if missing:
        return [f"{query} {' '.join(missing[:4])}"]
    keywords = plan.get("keywords", [])
    if keywords:
        return [f"{query} {' '.join(str(x) for x in keywords[:4])}"]
    return [query]


def merge_and_filter_matches(
    previous: list[dict],
    new_matches: list[dict],
    *,
    candidate_limit: int,
    min_rrf_score: float = SEARCH_MIN_RRF_SCORE,
) -> list[dict]:
    """RRF統合と低スコア除外までの決定的な前段。CLI/Web/評価harnessで共有する。"""
    merged = merge_results(previous, new_matches, max_results=candidate_limit * 2)
    return filter_by_rrf_score(merged, min_rrf_score)


def finalize_ranked_matches(
    query: str,
    matches: list[dict],
    *,
    must_find_terms: list[str] | None = None,
    max_per_file: int = MAX_CHUNKS_PER_FILE,
    relative_score_floor: float = 0.3,
) -> tuple[list[dict], float, str]:
    """file別上限・相対スコア足切り・confidence算出までの決定的な後段。

    rerank後の順位付き候補を受け取り、`(matches, confidence, evidence_status)` を返す。
    """
    supported = filter_by_relevance_support(query, matches)
    constraint_conflict = _has_explicit_constraint_conflict(query, supported)
    limited = _limit_chunks_per_file(supported, max_per_file)
    if len(limited) > 1:
        top_score = limited[0].get("rrf_score", 0)
        if top_score > 0:
            limited = [
                m
                for m in limited
                if m.get("rrf_score", 0) >= top_score * relative_score_floor
            ]
    confidence, status = _calculate_confidence(query, limited, must_find_terms)
    if status == "sufficient" and constraint_conflict:
        status = "partial"
    if constraint_conflict and limited:
        limited[0] = {**limited[0], "constraint_conflict": True}
    return limited, confidence, status


def run_retrieval_pipeline(
    query: str,
    *,
    model: str,
    reasoning: str | None = None,
    emit_status=None,
    cancel_check=None,
    mode: str = "answer",
) -> RetrievalResult:
    """CLI/Web 共通の bounded retrieval pipeline。

    ``mode=search`` は検索計画用・回答用の chat を一切呼ばず、決定的な
    query だけで一回検索する。既定の ``answer`` は従来の経路を維持する。
    """

    if mode not in {"answer", "search"}:
        raise ValueError("invalid retrieval mode")
    search_only = mode == "search"

    def status(text: str) -> None:
        if emit_status:
            emit_status(text)

    def check_cancel() -> None:
        if cancel_check:
            cancel_check()

    check_cancel()
    if search_only:
        # 検索専用は chat model・agentic-lite・キーワード抽出用 chat を
        # 呼ばず、既存の決定的な語分割を使う。
        plan = {
            "keywords": _split_terms(query),
            "search_queries": [query],
            "must_find_terms": [],
            "query_type": "search_only",
            "answer_should_compare": False,
            "fallback": "search_only",
        }
    else:
        status("検索計画を作成しています...")
        plan = create_search_plan(query, model)
    keywords = [str(x) for x in plan.get("keywords", []) if str(x).strip()]
    queries = [str(x) for x in plan.get("search_queries", []) if str(x).strip()]
    if query not in queries:
        queries.insert(0, query)
    queries = list(dict.fromkeys(queries))[:MAX_QUERY_VARIANTS]
    if search_only:
        queries = [query]

    embed_model = detect_embed_model()
    source_chunks = (
        build_source_chunks(SKILL_SOURCE_DIR) if _chunk_retrieval_enabled() else None
    )
    embed_cache = None
    embed_index = None
    route = "keyword"
    index_state = "missing"
    route_reason = ""
    warnings: list[str] = []

    if search_only and not embed_model:
        index_state = "unknown"
        route_reason = "Embeddingモデル未設定のためキーワード検索のみ"
        warnings.append(route_reason)
    elif search_only:
        # 検索専用では接続不能をモデル不存在と混同せず、index state は
        # ローカル cache/status から独立に判定する。
        candidate_cache = load_embed_cache()
        index_info = get_embed_index_status(
            embed_model, source_chunks, cache=candidate_cache
        )
        index_state = str(index_info.get("state") or "unknown")
        availability = _is_model_available(embed_model)
        if availability != ModelStatus.AVAILABLE:
            embed_model = None
            route_reason = (
                "Embeddingの利用可否を確認できないためキーワード検索のみ"
                if availability == ModelStatus.UNKNOWN
                else "Embeddingモデルがないためキーワード検索のみ"
            )
            warnings.append(route_reason)
        elif index_state == "ready":
            embed_cache = candidate_cache
            embed_index = _build_embed_index(embed_cache)
            route = "hybrid"
        else:
            embed_model = None
            route_reason = f"Embeddingインデックスは{index_state}のためキーワード検索のみ"
            warnings.append(route_reason)

    if embed_model and not search_only:
        status("Embeddingインデックスの状態を確認しています...")
        # 1-a: get_embed_index_status に渡すことで内部の二重ロードを避ける。
        candidate_cache = load_embed_cache()
        index_info = get_embed_index_status(
            embed_model, source_chunks, cache=candidate_cache
        )
        index_state = str(index_info.get("state") or "missing")
        if index_state == "ready":
            embed_cache = candidate_cache
            embed_index = _build_embed_index(embed_cache)
            route = "hybrid"
        else:
            embed_model = None
            status(
                f"Embeddingインデックスは{index_state}です。キーワード検索のみで続行します。"
            )
    attempts: list[RetrievalAttempt] = []
    combined: list[dict] = []
    final_confidence = 0.0
    final_status = "insufficient"
    max_attempts = 1 if search_only else max(1, min(MAX_RETRIEVAL_ATTEMPTS, 2))
    rerank_enabled = RERANK_CONFIG.mode != "off"
    candidate_limit = (
        RERANK_CONFIG.top_n if rerank_enabled else RETRIEVAL_CANDIDATE_LIMIT
    )

    for attempt_index in range(max_attempts):
        check_cancel()
        status(
            "資料を検索しています..."
            if attempt_index == 0
            else "根拠不足のため再検索しています..."
        )
        # 1-c/1-d: クエリ変体の埋め込みを1バッチで取得し、1パスで全クエリ採点する。
        precomputed_by_query: dict[str, list] = {}
        if embed_model and embed_index is not None:
            try:
                precomputed_by_query = embedding_search_multi(
                    queries,
                    embed_model,
                    embed_index,
                    top_k=candidate_limit,
                    raise_on_error=search_only,
                )
            except Exception as e:
                if search_only:
                    # この要求内の embedding 候補を破棄し、単件再試行や
                    # index buildを行わず keyword だけで確定する。
                    embed_model = None
                    embed_index = None
                    embed_cache = None
                    route = "keyword"
                    route_reason = "Embedding検索に失敗したためキーワード検索のみ"
                    warnings.append(route_reason)
                logger.warning("Embedding search (multi) failed: %s", e)
                precomputed_by_query = {}
        attempt_matches: list[dict] = []
        for search_query in queries:
            check_cancel()
            # embed_model が無効な既存経路（keyword-only fallback）では
            # precomputed_embed_matches を渡さず、従来のシグネチャのまま呼ぶ
            # （_run_single_retrieval をモックする既存テストとの互換のため）。
            extra: dict = {}
            if search_query in precomputed_by_query:
                extra["precomputed_embed_matches"] = precomputed_by_query[search_query]
            attempt_matches.extend(
                _run_single_retrieval(
                    search_query,
                    keywords,
                    embed_model,
                    embed_cache,
                    source_chunks,
                    **extra,
                )
            )
        combined = merge_and_filter_matches(
            combined,
            attempt_matches,
            candidate_limit=candidate_limit,
        )
        if rerank_enabled:
            check_cancel()
            status("検索結果を再順位付けしています...")
            combined = _rerank_with_fallback(
                query,
                combined,
                plan=plan,
                config=RERANK_CONFIG,
                cancel_check=check_cancel,
            )
        combined, final_confidence, final_status = finalize_ranked_matches(
            query,
            combined,
            must_find_terms=plan.get("must_find_terms", []),
        )
        attempts.append(
            RetrievalAttempt(
                query=" / ".join(queries),
                keywords=keywords,
                match_count=len(combined),
                confidence=final_confidence,
                evidence_status=final_status,
                reason=plan.get("fallback", ""),
            )
        )
        if (
            final_status == "sufficient"
            or attempt_index >= max_attempts - 1
            or search_only
            or not _agentic_lite_enabled()
        ):
            break
        queries = _retry_queries(query, plan, combined)[:MAX_QUERY_VARIANTS]

    attempt_dicts = [attempt.__dict__ for attempt in attempts]
    if search_only:
        # 回答用 prompt 予算・prompt builder を通さず、SSE防御上限と
        # 既存の件数設定だけを適用する（既定5、最大8）。
        matches = combined[: min(RETRIEVAL_PROMPT_MATCH_LIMIT, 8)]
        return RetrievalResult(
            query=query,
            attempts=attempts,
            matches=matches,
            confidence=final_confidence,
            evidence_status=final_status,
            user_prompt="",
            plan=plan,
            route=route,
            index_state=index_state,
            embedding_model=embed_model if route == "hybrid" else None,
            route_reason=route_reason,
            warnings=list(dict.fromkeys(warnings)),
        )

    # 予算逆算が無効なら shell 見積り（build_user_prompt の追加呼び出し）を行わない。
    shell_chars = (
        _estimate_prompt_shell_chars(
            query,
            attempts=attempt_dicts,
            evidence_status=final_status,
            confidence=final_confidence,
        )
        if GENERATION_RESERVE_TOKENS > 0
        else 0
    )
    evidence_char_limit = compute_evidence_char_limit(shell_chars)
    matches = _fit_prompt_budget(
        combined[:RETRIEVAL_PROMPT_MATCH_LIMIT], evidence_char_limit
    )
    user_prompt = build_user_prompt(
        query,
        matches,
        attempts=attempt_dicts,
        evidence_status=final_status,
        confidence=final_confidence,
    )
    return RetrievalResult(
        query=query,
        attempts=attempts,
        matches=matches,
        confidence=final_confidence,
        evidence_status=final_status,
        user_prompt=user_prompt,
        plan=plan,
        route=route,
        index_state=index_state,
        embedding_model=embed_model if route == "hybrid" else None,
        route_reason=route_reason,
        warnings=list(dict.fromkeys(warnings)),
    )


# ---------------------------------------------------------------------------


def build_chat_payload(
    model: str, system_prompt: str, user_prompt: str, reasoning: Optional[str] = None
) -> dict:
    """Ollama Chat API 用のリクエストボディを構築する。

    think は常に明示送信する契約とする。reasoning が None / "" / "off" なら
    通常はFalse、内部推論を無効化できないgpt-ossは最小のlowを送る。
    low/medium/highの明示指定はそのまま送る（F-7: 未送信による既定値復帰を防ぐ）。
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": build_chat_options(),
    }
    if reasoning in ("low", "medium", "high"):
        body["think"] = reasoning
    elif model.rsplit("/", 1)[-1].split(":", 1)[0].lower() == "gpt-oss":
        # GPT-OSS cannot disable internal reasoning; use its supported minimum.
        # The Web service independently hides thinking events for reasoning=off.
        body["think"] = "low"
    else:
        body["think"] = False
    return body


def build_evidence_summary(
    matches: list[dict],
    *,
    evidence_status: str,
    confidence: float,
) -> str:
    """CLI向けに採用根拠を決定論的な一覧へ整形する。"""
    lines = [f"根拠一覧（状態: {evidence_status}、信頼度: {confidence:.2f}）"]
    if not matches:
        lines.append("  根拠チャンクはありません。")
        return "\n".join(lines)

    warnings: list[str] = []
    for index, match in enumerate(matches, 1):
        title = str(match.get("source_title") or match.get("path") or "不明な資料")
        location = []
        if match.get("page") is not None:
            location.append(f"ページ {match['page']}")
        if match.get("heading"):
            location.append(f"見出し {match['heading']}")
        start_line = match.get("start_line")
        end_line = match.get("end_line")
        if start_line is not None and end_line is not None:
            location.append(f"行 {start_line}-{end_line}")
        if match.get("layout_type"):
            location.append(f"種別 {match['layout_type']}")
        if match.get("parser"):
            location.append(f"parser {match['parser']}")
        if match.get("source"):
            location.append(f"検索 {match['source']}")
        suffix = " / ".join(location) if location else "位置情報なし"
        lines.append(f"  {index}. {title} — {suffix}")
        for warning in match.get("parser_warnings") or []:
            normalized = str(warning).strip()
            if normalized and normalized not in warnings:
                warnings.append(normalized)

    if warnings:
        lines.append("Parser警告:")
        lines.extend(f"  - {warning}" for warning in warnings[:10])
    return "\n".join(lines)


def answer_empty_message(done_reason: str) -> str:
    """本文 0 文字で終了したときに利用者へ示す理由別メッセージを返す。

    ``length`` は context 枯渇による打ち切りであり対処が明確なため専用文言を
    返す。それ以外は原因を ``stop`` と断定せず、再試行を促すに留める。
    """
    if done_reason == "length":
        return (
            "[警告] モデルの出力上限に達したため回答本文を生成できませんでした。"
            "「推論: なし」で再試行するか、OLLAMA_NUM_CTX を大きくしてください。"
        )
    return "[警告] 回答本文が生成されませんでした。再試行してください。"


def iter_stream_events(resp):
    """Ollama ストリーミングレスポンスを逐次パースし、

    ``("thinking" | "content", text)`` のタプルを yield する。
    ストリーム終端では ``("done", done_reason)`` を最後に 1 回だけ yield する
    （``done_reason`` は ``"stop"`` / ``"length"`` 等。欠落時は ``"stop"``）。
    ``length`` は context を使い切って打ち切られたことを示し、呼び出し元が
    「本文が生成されなかった」ことを利用者へ明示するために使う。
    Web UI (SSE) と CLI の両方から利用可能なジェネレータ。
    副作用（print 等）を持たないため、呼び出し元で表示方法を制御できる。
    """
    incomplete_line = ""

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        if incomplete_line:
            line = incomplete_line + line
            incomplete_line = ""

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            incomplete_line = line
            continue

        if data.get("done", False):
            yield ("done", str(data.get("done_reason") or "stop"))
            break

        message = data.get("message")
        if not isinstance(message, dict):
            continue
        thinking = message.get("thinking")
        if thinking:
            yield ("thinking", thinking)
        content = message.get("content")
        if content:
            yield ("content", content)


def iter_stream_chunks(resp):
    """Ollama ストリーミングレスポンスを逐次パースし、content チャンク文字列を yield する。

    ``iter_stream_events`` の薄いラッパ。thinking は破棄する
    （既存 CLI / テスト互換のため維持）。
    """
    for kind, text in iter_stream_events(resp):
        if kind == "content":
            yield text


def _iter_stream(resp, *, show_thinking: bool = False) -> list[str]:
    """Ollama ストリーミングレスポンスを逐次パースし表示する。"""
    full_response = []
    thinking_open = False
    done_reason = "stop"
    for kind, text in iter_stream_events(resp):
        if kind == "thinking":
            if show_thinking:
                if not thinking_open:
                    print("[thinking] ", end="", flush=True)
                    thinking_open = True
                print(text, end="", flush=True)
            continue
        if kind == "done":
            done_reason = text
            continue
        if thinking_open:
            print(flush=True)
            thinking_open = False
        print(text, end="", flush=True)
        full_response.append(text)
    if thinking_open:
        print(flush=True)
    if not full_response:
        print(answer_empty_message(done_reason), flush=True)
    return full_response


def stream_ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning: Optional[str] = None,
    *,
    show_thinking: bool = False,
) -> str:
    """Ollama Chat API にストリーミングリクエストを送信し、応答を逐次表示する。

    reasoning が指定されている場合、think パラメータを付与する。
    think パラメータでエラーが発生した場合は除去して1回だけ再送を試行する
    （送信した think の値に関わらず、body に think キーが含まれていれば対象）。
    ソケットの読取タイムアウトは生成フェーズの stall 上限と揃える。
    """
    url = f"{OLLAMA_HOST}/api/chat"
    body = build_chat_payload(model, system_prompt, user_prompt, reasoning)
    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full_response = []

    try:
        with urllib.request.urlopen(req, timeout=GENERATION_STALL_TIMEOUT) as resp:
            full_response = _iter_stream(resp, show_thinking=show_thinking)
    except urllib.error.HTTPError as e:
        if "think" in body and e.code >= 400:
            # think パラメータを除去して再試行
            print(
                f"[WARN] think パラメータでエラー({e.code})。通常モードで再試行します",
                file=sys.stderr,
            )
            body.pop("think", None)
            retry_payload = json.dumps(body).encode("utf-8")
            retry_req = urllib.request.Request(
                url,
                data=retry_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(retry_req, timeout=GENERATION_STALL_TIMEOUT) as resp:
                full_response = _iter_stream(resp, show_thinking=show_thinking)
        else:
            raise
    except urllib.error.URLError as e:
        msg = f"\n\n[接続エラー] Ollama への接続が中断されました: {e}"
        print(msg, flush=True)
        full_response.append(msg)
    except TimeoutError:
        msg = "\n\n[タイムアウト] Ollama からの応答がタイムアウトしました。"
        print(msg, flush=True)
        full_response.append(msg)

    print()  # 改行
    return "".join(full_response)


def main():
    parser = argparse.ArgumentParser(description="Offline AI Search CLI")
    parser.add_argument(
        "query", nargs="?", default=None, help="Search query (positional)"
    )
    parser.add_argument(
        "--query-file", type=str, default=None, help="Path to file containing query"
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default=None,
        help="Reasoning effort: low/medium/high/off (off disables thinking)",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="thinkingデルタをコンソールへ表示する（既定: 非表示）",
    )
    args = parser.parse_args()

    query = resolve_query(query_file=args.query_file, query_positional=args.query)
    model = detect_model()
    reasoning = _resolve_reasoning(args.reasoning)

    # モデル可用性チェック
    model_status = _is_model_available(model)
    if model_status == ModelStatus.UNAVAILABLE:
        print(
            f"[ERROR] モデル '{model}' が見つかりません。オフラインパッケージの install-offline.bat を再実行してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    elif model_status == ModelStatus.UNKNOWN:
        print(
            "[WARN] Ollama への接続に失敗しました。モデル可用性を検証できません。",
            file=sys.stderr,
        )

    print(f"モデル: {model}")
    print(f"クエリ: {query}")
    print(f"リーズニング: {reasoning}")
    print(
        f"コンテキスト: {NUM_CTX} tokens, バッチ: {NUM_BATCH}, GPU layers: {'all' if NUM_GPU == -1 else NUM_GPU}"
    )
    print()

    # Embedding モデルの検出
    embed_model = detect_embed_model()
    if embed_model:
        embed_status = _is_model_available(embed_model)
        embed_available = embed_status == ModelStatus.AVAILABLE
        if not embed_available:
            print(
                f"[WARN] Embeddingモデル '{embed_model}' が利用不可。キーワード検索のみで続行します。",
                file=sys.stderr,
            )
    else:
        embed_available = False
        print(
            "[INFO] .model_embed 未設定。Embedding検索は無効です。"
            "有効化(日本語推奨): echo bge-m3 > offline-ai\\_internal\\.model_embed",
            file=sys.stderr,
        )

    try:
        retrieval = run_retrieval_pipeline(
            query,
            model=model,
            reasoning=reasoning,
            emit_status=lambda text: print(text),
        )
    except KeyboardInterrupt:
        print(
            "中断しました。Embedding checkpointがあれば次回検索時に再開します。",
            file=sys.stderr,
        )
        raise SystemExit(130) from None

    for i, attempt in enumerate(retrieval.attempts, 1):
        print(
            f"検索試行{i}: マッチ {attempt.match_count} 件、"
            f"根拠ステータス {attempt.evidence_status}、信頼度 {attempt.confidence}"
        )

    print(f"最終マッチ: {len(retrieval.matches)} 件")
    print(
        build_evidence_summary(
            retrieval.matches,
            evidence_status=retrieval.evidence_status,
            confidence=retrieval.confidence,
        )
    )
    print()

    # 2. プロンプト構築
    user_prompt = retrieval.user_prompt

    # 3. Ollama API で回答生成
    print("=" * 50)
    print("AI 回答:")
    print("=" * 50)
    print()

    stream_ollama_chat(
        model, SYSTEM_PROMPT, user_prompt, reasoning=reasoning, show_thinking=args.show_thinking
    )


if __name__ == "__main__":
    main()
