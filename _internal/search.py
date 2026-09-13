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
import source_structure as _source_structure

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


def get_embed_model_identity(
    model: str, *, timeout: float = 2.0
) -> dict[str, str] | None:
    """Ollamaのmodel digestを取得する。取得不能なら安全側に ``None``。"""
    requested = _canonical_model_reference(model)
    try:
        request = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return None
    for item in payload["models"]:
        if not isinstance(item, dict):
            continue
        if _canonical_model_reference(str(item.get("name", ""))) != requested:
            continue
        digest = str(item.get("digest", "")).strip()
        if digest:
            return {"name": requested, "digest": digest}
        return None
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


class SourceSnapshotError(RuntimeError):
    """資料snapshotを完全には取得できなかった。"""

    code = "SOURCE_SNAPSHOT_FAILED"

    def __init__(self, failed_paths: list[str]):
        super().__init__(
            f"{len(failed_paths)} source file(s) could not be read or parsed: "
            + ", ".join(failed_paths[:5])
        )
        self.failure_count = len(failed_paths)
        self.failed_paths = list(failed_paths)


class SourceChunkList(list):
    """chunk list と、同じ走査で得た file manifest の組。

    list の部分型なので既存 caller はそのまま使える。``source_complete`` が
    False の場合は読み取り・解析できないファイルがあり、manifest は資料全体を
    表さない（ready 判定に使ってはならない）。
    """

    source_manifest: dict[str, dict] | None = None
    source_complete: bool = True


def _chunk_source_bytes(
    data: bytes,
    path: Path,
    rel_path: str,
    *,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[dict]:
    """同じbytesから本文・hash・chunkを作り、世代混在を防ぐ。"""
    # 不正バイトは U+FFFD として保持する。検索とindexで同じ本文・chunk_idに
    # なるよう、decode規則を1つに固定する（EMBED_SOURCE_PARSER_VERSIONの一部）。
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


def chunk_source_file(
    path: Path, rel_path: str, *, max_chars: int = CHUNK_MAX_CHARS
) -> list[dict]:
    """source ファイルを検索・Embedding 用チャンクへ分割する。"""
    return _chunk_source_bytes(path.read_bytes(), path, rel_path, max_chars=max_chars)


def _read_source_snapshot(
    source_root: Path | None = None,
) -> tuple[SourceChunkList, dict[str, dict], list[str]]:
    """資料を1回走査し、chunk・file manifest・失敗pathを返す。

    manifestはchunkが0件のファイルも保持する。読み取り・解析に失敗した
    ファイルは manifest に含めず、空ファイルや削除とは区別して返す。
    """
    chunks = SourceChunkList()
    files: dict[str, dict] = {}
    failed: list[str] = []
    for path, rel in iter_source_files(source_root):
        try:
            data = path.read_bytes()
            file_chunks = _chunk_source_bytes(data, path, rel)
        except (OSError, ValueError):
            failed.append(rel)
            continue
        files[rel] = {
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "chunk_ids": [str(chunk["chunk_id"]) for chunk in file_chunks],
            "chunk_count": len(file_chunks),
        }
        chunks.extend(file_chunks)
    chunks.source_manifest = files
    chunks.source_complete = not failed
    return chunks, files, failed


def build_source_snapshot(
    source_root: Path | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """index writer用の完全snapshot。1件でも読めなければ更新を保留する。"""
    chunks, files, failed = _read_source_snapshot(source_root)
    if failed:
        raise SourceSnapshotError(failed)
    return chunks, files


def _source_manifest_from_chunks(chunks: list[dict]) -> dict[str, dict]:
    """テスト・既存caller向けにchunk集合からmanifestを再構成する。"""
    files: dict[str, dict] = {}
    for chunk in chunks:
        rel = normalize_source_path(str(chunk.get("path", "")))
        if not rel:
            continue
        item = files.setdefault(
            rel,
            {
                "file_sha256": str(chunk.get("file_sha256", "")),
                "chunk_ids": [],
                "chunk_count": 0,
            },
        )
        item["chunk_ids"].append(str(chunk.get("chunk_id", "")))
        item["chunk_count"] = len(item["chunk_ids"])
    return files


def _chunk_snapshot_signature(chunks: list[dict]) -> tuple:
    """昇格直前の再読込比較用。mtimeだけは世代差としない。"""
    signature = []
    for chunk in chunks:
        signature.append(
            json.dumps(
                {key: value for key, value in chunk.items() if key != "modifiedAt"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return tuple(sorted(signature))


def build_source_chunks(source_root: Path | None = None) -> list[dict]:
    """検索read経路用のchunk集合。読めないファイルはskipし、検索を止めない。

    戻り値の ``SourceChunkList`` には manifest と完全性が付き、
    ``get_embed_index_status`` は不完全なsnapshotを ready と判定しない。
    """
    chunks, _files, _failed = _read_source_snapshot(source_root)
    return chunks


def _source_manifest_for_chunks(
    chunks: list[dict], source_manifest: dict[str, dict] | None = None
) -> tuple[dict[str, dict], bool]:
    """ready判定に使う manifest と、そのsnapshotが完全かを返す。"""
    if source_manifest is not None:
        return source_manifest, True
    attached = getattr(chunks, "source_manifest", None)
    if isinstance(attached, dict):
        return attached, bool(getattr(chunks, "source_complete", True))
    # 差し替えられた build_source_chunks や明示chunkを渡す既存caller。
    return _source_manifest_from_chunks(chunks), True


# build_source_chunks() / compute_embed_generation() の read 経路専用 memo。
# キーは skill-source の (rel, st_mtime_ns, st_size, sha256) 集合であり、追加・
# 更新・削除に加え、同size/同mtimeの内容変更でもキーが変わり自己無効化する。
# index build の writer 経路は build_source_snapshot() を直接呼び、この memo を
# 通らない。
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
    _invalidate_structure_memo()


def _source_chunk_memo_key(source_root: Path | None = None) -> tuple | None:
    """skill-source の identity キーを返す。stat に失敗したら None（memo 無効）。"""
    entries = []
    for path, rel in iter_source_files(source_root):
        try:
            stat = path.stat()
        except OSError:
            return None
        try:
            file_sha = _file_sha256(path)
        except OSError:
            return None
        entries.append((rel, stat.st_mtime_ns, stat.st_size, file_sha))
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


EMBED_CACHE_VERSION = 5
EMBED_CHECKPOINT_SCHEMA_VERSION = 3
EMBED_COMPATIBILITY_VERSION = 1
EMBED_SOURCE_PARSER_VERSION = 1
EMBED_INDEX_MODES = {"incremental", "full"}
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
        "source_parser_version": EMBED_SOURCE_PARSER_VERSION,
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


def _atomic_write_json(path: Path, payload: dict, *, before_replace=None) -> bool:
    """同一directoryへのwrite、flush/fsync、replaceを一体化する。

    ``before_replace(tmp_path)`` は書き込み済み一時ファイルの検証用hookで、
    例外を送出すると置換せず一時ファイルを削除し、その例外を再送出する。
    """
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
        if before_replace is not None:
            before_replace(tmp_path)
        os.replace(tmp_path, path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("atomic JSON save failed for %s: %s", path.name, exc)
        _unlink_quietly(tmp_path)
        return False
    except BaseException:
        _unlink_quietly(tmp_path)
        raise


def _unlink_quietly(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


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


def _new_checkpoint_state(
    embed_model: str,
    generation: str | None = None,
    *,
    mode: str = "incremental",
    model_identity: dict[str, str] | None = None,
) -> dict:
    now = _utc_now()
    return {
        "schema_version": EMBED_CHECKPOINT_SCHEMA_VERSION,
        "cache_version": EMBED_CACHE_VERSION,
        "embed_model": _canonical_embed_model(embed_model),
        "mode": mode,
        "compatibility": _new_embed_compatibility(
            embed_model, model_identity, None
        ),
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
    state: dict,
    embed_model: str,
    generation: str | None = None,
    *,
    mode: str = "incremental",
    model_identity: dict[str, str] | None = None,
) -> bool:
    chunking = state.get("chunking") if isinstance(state.get("chunking"), dict) else {}
    # digestが一致する場合、または作成時・現在ともにdigest不明の同名モデルだけを
    # 同じjobの途中再開として扱う。片側だけ不明なら別モデルの可能性を排除できない。
    compatibility_matches = _cache_ready_compatibility_matches(
        {"version": state.get("cache_version"), "compatibility": state.get("compatibility")},
        embed_model,
        model_identity,
    )
    return (
        state.get("schema_version") == EMBED_CHECKPOINT_SCHEMA_VERSION
        and state.get("cache_version") == EMBED_CACHE_VERSION
        and state.get("mode", "incremental") == mode
        and _canonical_embed_model(state.get("embed_model"))
        == _canonical_embed_model(embed_model)
        and chunking.get("max_chars") == CHUNK_MAX_CHARS
        and chunking.get("overlap_chars") == CHUNK_OVERLAP_CHARS
        and isinstance(state.get("generation"), str)
        and bool(state.get("generation"))
        and (generation is None or state.get("generation") == generation)
        and isinstance(state.get("next_batch"), int)
        and state.get("next_batch") >= 1
        and compatibility_matches
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


def _new_embed_compatibility(
    embed_model: str,
    model_identity: dict[str, str] | None,
    embedding_dimension: int | None,
) -> dict:
    """cache/checkpoint共通の再利用互換契約。"""
    return {
        "version": EMBED_COMPATIBILITY_VERSION,
        "model": _canonical_embed_model(embed_model),
        "model_digest": (
            str(model_identity.get("digest", "")).strip()
            if isinstance(model_identity, dict)
            else None
        ),
        "source_parser_version": EMBED_SOURCE_PARSER_VERSION,
        "chunking": {
            "max_chars": CHUNK_MAX_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
        },
        "embedding_dimension": embedding_dimension,
    }


def _cache_compatibility_matches(
    cache: dict,
    embed_model: str,
    model_identity: dict[str, str] | None,
) -> bool:
    """generationとは独立した、Embedding再利用の安全条件。"""
    compatibility = cache.get("compatibility")
    if not isinstance(compatibility, dict) or not isinstance(model_identity, dict):
        return False
    digest = str(model_identity.get("digest", "")).strip()
    chunking = compatibility.get("chunking")
    return (
        cache.get("version") == EMBED_CACHE_VERSION
        and compatibility.get("version") == EMBED_COMPATIBILITY_VERSION
        and compatibility.get("model") == _canonical_embed_model(embed_model)
        and compatibility.get("model_digest") == digest
        and bool(digest)
        and compatibility.get("source_parser_version") == EMBED_SOURCE_PARSER_VERSION
        and chunking == {
            "max_chars": CHUNK_MAX_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
        }
    )


def _cache_ready_compatibility_matches(
    cache: dict,
    embed_model: str,
    model_identity: dict[str, str] | None,
) -> bool:
    """ready判定用の互換条件。再利用判定より緩めず、digest不明を明示的に扱う。

    - cacheにdigestがある: 現在のdigestが取得でき、一致する場合だけ ready。
    - cacheのdigestが不明（構築時に取得不能）: 現在も取得不能な場合だけ、
      モデル名一致で ready とする。この cache の Embedding は再利用しない。
    """
    compatibility = cache.get("compatibility")
    if not isinstance(compatibility, dict):
        return False
    base_matches = (
        cache.get("version") == EMBED_CACHE_VERSION
        and compatibility.get("version") == EMBED_COMPATIBILITY_VERSION
        and compatibility.get("model") == _canonical_embed_model(embed_model)
        and compatibility.get("source_parser_version") == EMBED_SOURCE_PARSER_VERSION
        and compatibility.get("chunking")
        == {"max_chars": CHUNK_MAX_CHARS, "overlap_chars": CHUNK_OVERLAP_CHARS}
    )
    if not base_matches:
        return False
    cached_digest = compatibility.get("model_digest")
    current_digest = (
        str(model_identity.get("digest", "")).strip()
        if isinstance(model_identity, dict)
        else ""
    )
    if cached_digest is None:
        return not current_digest
    return bool(current_digest) and cached_digest == current_digest


def _cache_embedding_dimension(cache: dict) -> int | None:
    compatibility = cache.get("compatibility")
    if isinstance(compatibility, dict):
        value = compatibility.get("embedding_dimension")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    dimensions = set()
    entries = cache.get("entries") if isinstance(cache.get("entries"), dict) else {}
    for entry in entries.values():
        vector = entry.get("embedding") if isinstance(entry, dict) else None
        if isinstance(vector, list) and vector:
            dimensions.add(len(vector))
    return next(iter(dimensions)) if len(dimensions) == 1 else None


def _cache_files(cache: dict) -> dict[str, dict]:
    manifest = cache.get("files")
    # file manifest は version 5 形式だけが持つ。旧cacheは読込時の正規化で
    # ``files: {}`` が補われるため、dict かどうかでは旧形式を判別できない
    # （判別を誤ると全ファイルが「追加」と表示される）。
    if cache.get("version") == EMBED_CACHE_VERSION and isinstance(manifest, dict):
        return {
            str(path): value
            for path, value in manifest.items()
            if isinstance(value, dict)
        }
    # version 4以前のcacheは、manifestが無いので再利用には使わない（互換判定で除外）。
    # ただし事前解析の追加・変更・削除件数表示のために、entryから観測できる範囲だけ
    # 再構成する。chunkが0件だったファイルは旧cacheに痕跡がなく「追加」に数えられる。
    files: dict[str, dict] = {}
    entries = cache.get("entries") if isinstance(cache.get("entries"), dict) else {}
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        path = normalize_source_path(str(entry.get("path", "")))
        if not path:
            continue
        item = files.setdefault(
            path,
            {"file_sha256": str(entry.get("file_sha256", "")), "chunk_ids": []},
        )
        chunk_id = str(entry.get("chunk_id", ""))
        if chunk_id:
            item["chunk_ids"].append(chunk_id)
        item["chunk_count"] = len(item["chunk_ids"])
    return files


def _entry_matches_chunk(
    entry: object,
    chunk: dict,
    *,
    expected_dimension: int | None = None,
) -> bool:
    if not _valid_embedding_entry(entry, str(chunk.get("chunk_id", ""))):
        return False
    assert isinstance(entry, dict)
    if (
        entry.get("file_sha256") != chunk.get("file_sha256")
        or entry.get("text_sha256") != chunk.get("text_sha256")
    ):
        return False
    vector = entry.get("embedding")
    return expected_dimension is None or len(vector) == expected_dimension


def _validate_embed_cache_candidate(
    cache: dict,
    embed_model: str,
    model_identity: dict[str, str] | None,
    generation: str,
    chunks: list[dict],
    source_manifest: dict[str, dict],
) -> int | None:
    """保存前・保存後に同じ候補検証を行う。"""
    if not _cache_header_matches(cache, embed_model, generation):
        raise EmbedCachePersistenceError("Embedding cache header validation failed")
    if model_identity is None:
        compatibility = cache.get("compatibility")
        unknown_compatibility = (
            isinstance(compatibility, dict)
            and compatibility.get("version") == EMBED_COMPATIBILITY_VERSION
            and compatibility.get("model") == _canonical_embed_model(embed_model)
            and compatibility.get("model_digest") is None
            and compatibility.get("source_parser_version") == EMBED_SOURCE_PARSER_VERSION
        )
        if not unknown_compatibility:
            raise EmbedCachePersistenceError("Embedding cache compatibility validation failed")
    elif not _cache_compatibility_matches(cache, embed_model, model_identity):
        raise EmbedCachePersistenceError("Embedding cache compatibility validation failed")
    if cache.get("files") != source_manifest:
        raise EmbedCachePersistenceError("Embedding cache manifest validation failed")
    entries = cache.get("entries")
    expected_ids = {str(chunk.get("chunk_id")) for chunk in chunks}
    if not isinstance(entries, dict) or set(entries) != expected_ids:
        raise EmbedCachePersistenceError("Embedding cache entry set validation failed")
    compatibility = cache.get("compatibility")
    declared_dimension = (
        compatibility.get("embedding_dimension")
        if isinstance(compatibility, dict)
        and isinstance(compatibility.get("embedding_dimension"), int)
        and not isinstance(compatibility.get("embedding_dimension"), bool)
        and compatibility.get("embedding_dimension") > 0
        else None
    )
    expected_dimension = declared_dimension or _cache_embedding_dimension(cache)
    actual_dimension: int | None = None
    for chunk in chunks:
        entry = entries.get(chunk.get("chunk_id"))
        if not _entry_matches_chunk(
            entry, chunk, expected_dimension=expected_dimension
        ):
            raise EmbedCachePersistenceError("Embedding cache entry validation failed")
        assert isinstance(entry, dict)
        if actual_dimension is None:
            actual_dimension = len(entry["embedding"])
        elif len(entry["embedding"]) != actual_dimension:
            raise EmbedCachePersistenceError("Embedding vector dimensions differ")
        # 候補の引用情報は現行parserのchunkから組み立てる契約だが、保存後の
        #再読込でも本文・位置が現行chunkと一致していることを確認する。
        for key in ("path", "chunk_id", "text", "heading", "start_line", "end_line"):
            if entry.get(key) != chunk.get(key):
                raise EmbedCachePersistenceError("Embedding citation metadata mismatch")
    if declared_dimension is not None and actual_dimension != declared_dimension:
        raise EmbedCachePersistenceError("Embedding dimension metadata mismatch")
    return actual_dimension


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


def _read_checkpoint_batches(state: dict) -> tuple[dict[str, dict], int]:
    """state と同じ generation/mode の batch entry を読むだけ（削除・修復しない）。"""
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
                or batch.get("mode", state.get("mode", "incremental"))
                != state.get("mode", "incremental")
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
    return entries, highest_sequence


def _peek_checkpoint_entries(
    embed_model: str,
    generation: str,
    *,
    mode: str,
    model_identity: dict[str, str] | None,
) -> dict[str, dict]:
    """事前解析用。条件が一致するcheckpoint entryを、ファイルを変更せずに返す。"""
    state_path = EMBED_CHECKPOINT_PATH / "state.json"
    try:
        if not state_path.is_file() or state_path.is_symlink():
            return {}
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(state, dict) or not _checkpoint_header_matches(
        state, embed_model, generation, mode=mode, model_identity=model_identity
    ):
        return {}
    entries, _highest = _read_checkpoint_batches(state)
    return entries


def _load_checkpoint(
    embed_model: str,
    generation: str | None = None,
    *,
    mode: str = "incremental",
    model_identity: dict[str, str] | None = None,
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

    if state is None or not _checkpoint_header_matches(
        state,
        embed_model,
        generation,
        mode=mode,
        model_identity=model_identity,
    ):
        _remove_checkpoint_files(include_state=True)
        state = _new_checkpoint_state(
            embed_model,
            generation,
            mode=mode,
            model_identity=model_identity,
        )
        if not _atomic_write_json(state_path, state):
            raise EmbedCachePersistenceError("checkpoint stateの初期化に失敗しました")
        return state, {}

    _remove_checkpoint_tmp_files()
    entries, highest_sequence = _read_checkpoint_batches(state)
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
        "mode": state.get("mode", "incremental"),
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
    """空のEmbedding cache（file manifestと互換契約を含む）。"""
    return {
        "version": EMBED_CACHE_VERSION,
        "embed_model": None,
        "generation": None,
        "compatibility": None,
        "files": {},
        "chunking": {
            "max_chars": CHUNK_MAX_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
        },
        "entries": {},
    }


# load_embed_cache() の read 経路専用 memo。キーは (絶対path, st_mtime_ns, st_size)。
# cache は同一directoryの一時ファイルからの原子的置換でだけ更新されるため、
# 置換で mtime/size が変わる。writer 経路は use_memo=False で必ず fresh load し、
# 昇格（save_embed_cache / build 完了）時に同一プロセスの memo を破棄する。返るdictは共有参照であり、
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
    data.setdefault("compatibility", None)
    data.setdefault("files", {})
    return data


def _load_embed_cache_file(path: Path) -> dict:
    streamed = _load_embed_cache_streaming(path)
    if streamed is not None:
        return _normalize_embed_cache(streamed)
    try:
        return _normalize_embed_cache(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return _empty_embed_cache()


def _load_embed_cache_from_disk() -> dict:
    return _load_embed_cache_file(EMBED_CACHE_PATH)


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


@dataclass
class QueryContext:
    """要求ローカルの使い捨てquery embeddingコンテキスト。

    その試行で重複排除済みのqueriesと取得済みvectors、モデル識別を保持する。
    親子展開の子選択（予算超過時の優先順位付け）専用に使い、要求終了時に
    破棄する。ディスクや通常ログへは保存しない。
    """

    queries: list[str]
    vectors: dict[str, list[float]]
    embed_model: str


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


def _embedding_search_multi_impl(
    queries: list[str], embed_model: str, index: _EmbedIndex, top_k: int,
    *, raise_on_error: bool,
) -> tuple[dict[str, list], QueryContext | None]:
    """複数クエリのembeddingを1バッチで取得し、entriesを1パス走査して採点する。

    ``embedding_search_multi`` と ``embedding_search_multi_with_context`` の
    共通実装。戻り値は ``({query: [match, ...]}, QueryContext | None)``。
    query取得に失敗した場合や候補が無い場合は ``QueryContext`` を返さない。
    """
    unique_queries = list(dict.fromkeys(queries))
    if not unique_queries or not index.chunk_ids:
        return {query: [] for query in queries}, None
    try:
        query_vectors = _get_embeddings(unique_queries, embed_model)
    except EmbeddingBatchError as exc:
        if raise_on_error:
            raise
        print(f"[WARN] Embedding取得: {exc.code}", file=sys.stderr)
        return {query: [] for query in queries}, None

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

    context = QueryContext(
        queries=unique_queries,
        vectors=dict(zip(unique_queries, query_vectors)),
        embed_model=embed_model,
    )
    return {query: scored_by_query[query] for query in queries}, context


def embedding_search_multi(
    queries: list[str], embed_model: str, index: _EmbedIndex, top_k: int = 8,
    *, raise_on_error: bool = False,
) -> dict[str, list]:
    """複数クエリのembeddingを1バッチで取得し、entriesを1パス走査して採点する。

    戻り値は ``{query: [match, ...]}``。各クエリの結果は ``embedding_search`` を
    個別に呼んだ場合と ``round(sim, 4)`` の桁で一致する（次元不一致entryのスキップ、
    閾値境界、top_k切り詰めを含む）。既存の戻り値契約を維持する公開wrapper。
    """
    results, _context = _embedding_search_multi_impl(
        queries, embed_model, index, top_k, raise_on_error=raise_on_error
    )
    return results


def embedding_search_multi_with_context(
    queries: list[str], embed_model: str, index: _EmbedIndex, top_k: int = 8,
    *, raise_on_error: bool = False,
) -> tuple[dict[str, list], QueryContext | None]:
    """``embedding_search_multi`` と同じ検索結果に加え、要求ローカルの
    ``QueryContext``（重複排除済みqueriesとその取得済みvectors）を返す。

    親子展開の子選択（予算超過時の優先順位付け）専用。API呼出しを二重化
    しないよう、内部実装は ``embedding_search_multi`` と共有する。
    """
    return _embedding_search_multi_impl(
        queries, embed_model, index, top_k, raise_on_error=raise_on_error
    )


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
        "mode",
        "total",
        "processed",
        "generated",
        "reused",
        "failed",
        "checkpointed",
        "elapsed_seconds",
        "rate_per_second",
        "rate_basis",
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
            "mode",
            "total",
            "processed",
            "generated",
            "reused",
            "failed",
            "checkpointed",
            "elapsed_seconds",
            "rate_per_second",
            "rate_basis",
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


def _generated_rate_record(status: dict, embed_model: str | None) -> dict:
    """同一モデルの実Embedding生成速度（generated基準）だけを取り出す。

    旧形式（processed基準、rate_basisなし）や別モデルの速度は見積もりへ流用しない。
    """
    rate = status.get("rate_per_second") if isinstance(status, dict) else None
    if (
        isinstance(status, dict)
        and status.get("rate_basis") == "generated"
        and embed_model
        and _canonical_embed_model(status.get("embed_model"))
        == _canonical_embed_model(embed_model)
        and isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and math.isfinite(rate)
        and rate > 0
    ):
        return {"rate_per_second": float(rate), "rate_basis": "generated"}
    return {"rate_per_second": None, "rate_basis": None}


def get_embed_index_status(
    embed_model: str | None = None,
    chunks: list[dict] | None = None,
    *,
    cache: dict | None = None,
    validate_entries: bool = True,
    source_manifest: dict[str, dict] | None = None,
    model_identity: dict[str, str] | None = None,
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
    resolved_manifest, source_complete = _source_manifest_for_chunks(
        source_chunks, source_manifest
    )
    generation = compute_embed_generation_cached(model, source_chunks, chunk_memo_key)
    resolved_cache = cache if cache is not None else load_embed_cache()
    if model_identity is None:
        model_identity = get_embed_model_identity(model)
    entries = (
        resolved_cache.get("entries") if isinstance(resolved_cache.get("entries"), dict) else {}
    )
    cache_ready = (
        source_complete
        and _cache_header_matches(resolved_cache, model, generation)
        and _cache_ready_compatibility_matches(resolved_cache, model, model_identity)
        and resolved_cache.get("files") == resolved_manifest
        and set(entries)
        == {chunk.get("chunk_id") for chunk in source_chunks}
    )
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
            **_generated_rate_record(persisted, model),
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
        **_generated_rate_record(persisted, model),
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


def _validate_embed_index_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized not in EMBED_INDEX_MODES:
        raise ValueError(f"invalid index mode: {mode}")
    return normalized


def plan_embed_index_update(
    embed_model: str,
    chunks: list[dict] | None = None,
    *,
    source_manifest: dict[str, dict] | None = None,
    mode: str = "incremental",
    model_identity: dict[str, str] | None = None,
) -> dict:
    """副作用なしに差分件数・再計算件数・見積もりを返す。

    Embedding生成、job開始、cache/checkpoint/statusの書き換えは行わない。
    再利用判定は build と同じ ``_select_reusable_embeddings`` を使う。
    """
    mode = _validate_embed_index_mode(mode)
    if chunks is None:
        source_chunks, resolved_manifest = build_source_snapshot(SKILL_SOURCE_DIR)
    else:
        source_chunks = chunks
        resolved_manifest, _complete = _source_manifest_for_chunks(
            source_chunks, source_manifest
        )
    if model_identity is None:
        model_identity = get_embed_model_identity(embed_model)
    generation = compute_embed_generation(embed_model, source_chunks)
    # read専用memoを共有し、確認のたびに大規模cacheを再parseしない（内容は変更しない）。
    cache = load_embed_cache()
    old_files = _cache_files(cache)
    current_paths = set(resolved_manifest)
    old_paths = set(old_files)
    added = len(current_paths - old_paths)
    deleted = len(old_paths - current_paths)
    changed = sum(
        1
        for path in current_paths & old_paths
        if old_files[path].get("file_sha256") != resolved_manifest[path].get("file_sha256")
    )
    unchanged = len(current_paths & old_paths) - changed
    checkpoint_entries = _peek_checkpoint_entries(
        embed_model, generation, mode=mode, model_identity=model_identity
    )
    selection = _select_reusable_embeddings(
        cache,
        embed_model,
        model_identity,
        mode,
        source_chunks,
        resolved_manifest,
        checkpoint_entries,
    )
    reused = selection.from_cache
    checkpoint_reused = selection.from_checkpoint
    generated = len(source_chunks) - reused - checkpoint_reused
    rate_record = _generated_rate_record(load_index_status(), embed_model)
    rate = rate_record["rate_per_second"]
    if generated == 0:
        estimate = None
        estimate_status = "no_embedding"
    elif rate:
        estimate = round(generated / rate, 3)
        estimate_status = "estimated"
    else:
        estimate = None
        estimate_status = "no_rate"
    digest_available = bool(
        isinstance(model_identity, dict) and str(model_identity.get("digest", "")).strip()
    )
    if mode == "full":
        reason_code = "full_requested"
    elif selection.cache_compatible:
        reason_code = None
    elif not cache.get("entries") and not old_files:
        reason_code = "cache_missing"
    elif not digest_available:
        reason_code = "model_digest_unavailable"
    else:
        reason_code = "cache_incompatible"
    return {
        "mode": mode,
        "generation": generation,
        "added_files": added,
        "changed_files": changed,
        "deleted_files": deleted,
        "unchanged_files": unchanged,
        "total_chunks": len(source_chunks),
        "generated_chunks": generated,
        "reused_chunks": reused,
        "checkpoint_chunks": checkpoint_reused,
        "rate_per_second": rate,
        "rate_basis": rate_record["rate_basis"],
        "estimated_seconds": estimate,
        "estimate_status": estimate_status,
        "estimate_basis": "generated_chunks / generated_rate" if estimate is not None else None,
        "compatibility": "compatible" if selection.cache_compatible else "rebuild",
        "model_digest_available": digest_available,
        "reason": reason_code,
    }


@dataclass
class _ReuseSelection:
    """cache/checkpoint から再利用するEmbeddingの選別結果。"""

    entries: dict[str, dict]
    from_cache: int
    from_checkpoint: int
    cache_compatible: bool
    expected_dimension: int | None


def _select_reusable_embeddings(
    cache: dict,
    embed_model: str,
    model_identity: dict[str, str] | None,
    mode: str,
    source_chunks: list[dict],
    source_manifest: dict[str, dict],
    checkpoint_entries: dict[str, dict],
) -> _ReuseSelection:
    """再利用できるEmbeddingを選ぶ。本文・引用情報は常に現在のchunkを正とする。

    完成cacheはファイル単位の all-or-nothing で扱う。path・file hash・chunk ID列が
    一致し、そのファイルの全entryが健全（本文hash・有限値・次元）な場合だけ
    再利用し、1件でも欠落・不正ならファイル全体を再計算する。checkpoint は
    同じ generation/mode/互換契約の job が生成したものだけが渡される。
    """
    cache_entries = cache.get("entries") if isinstance(cache.get("entries"), dict) else {}
    cache_compatible = mode == "incremental" and _cache_compatibility_matches(
        cache, embed_model, model_identity
    )
    expected_dimension = _cache_embedding_dimension(cache) if cache_compatible else None
    if cache_compatible and cache_entries and expected_dimension is None:
        # 次元を一意に決められないcacheは、どのentryも安全に再利用できない。
        cache_compatible = False
    old_files = _cache_files(cache) if cache_compatible else {}

    chunks_by_path: dict[str, list[dict]] = {}
    for chunk in source_chunks:
        chunks_by_path.setdefault(
            normalize_source_path(str(chunk.get("path", ""))), []
        ).append(chunk)

    selected: dict[str, dict] = {}
    from_cache = 0
    for path, file_chunks in chunks_by_path.items():
        old = old_files.get(path)
        current = source_manifest.get(path)
        if (
            not isinstance(old, dict)
            or not isinstance(current, dict)
            or old.get("file_sha256") != current.get("file_sha256")
            or list(old.get("chunk_ids") or []) != list(current.get("chunk_ids") or [])
        ):
            continue
        if not all(
            _entry_matches_chunk(
                cache_entries.get(chunk.get("chunk_id")),
                chunk,
                expected_dimension=expected_dimension,
            )
            for chunk in file_chunks
        ):
            continue
        for chunk in file_chunks:
            selected[chunk["chunk_id"]] = {
                **_strip_cache_metadata(chunk),
                "embedding": cache_entries[chunk["chunk_id"]]["embedding"],
            }
        from_cache += len(file_chunks)

    from_checkpoint = 0
    for chunk in source_chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in selected:
            continue
        candidate = checkpoint_entries.get(chunk_id)
        if not _entry_matches_chunk(candidate, chunk, expected_dimension=expected_dimension):
            continue
        if expected_dimension is None:
            expected_dimension = len(candidate["embedding"])
        selected[chunk_id] = {
            **_strip_cache_metadata(chunk),
            "embedding": candidate["embedding"],
        }
        from_checkpoint += 1
    return _ReuseSelection(
        entries=selected,
        from_cache=from_cache,
        from_checkpoint=from_checkpoint,
        cache_compatible=cache_compatible,
        expected_dimension=expected_dimension,
    )


def build_or_update_embed_index(
    embed_model: str,
    chunks: list[dict] | None = None,
    *,
    source_manifest: dict[str, dict] | None = None,
    source_root: Path | None = None,
    mode: str = "incremental",
    model_identity: dict[str, str] | None = None,
    emit_progress=None,
    cancel_check=None,
) -> dict:
    """資料単位で差分更新し、検証済み候補だけをcacheへ原子的に昇格する。"""
    mode = _validate_embed_index_mode(mode)
    supplied_chunks = chunks is not None
    resolved_source_root = source_root if source_root is not None else (
        SKILL_SOURCE_DIR if not supplied_chunks else None
    )
    if chunks is None:
        source_chunks, resolved_manifest = build_source_snapshot(resolved_source_root)
    else:
        source_chunks = chunks
        resolved_manifest, _complete = _source_manifest_for_chunks(
            source_chunks, source_manifest
        )
    if model_identity is None and (not supplied_chunks or source_root is not None):
        model_identity = get_embed_model_identity(embed_model)
    total = len(source_chunks)
    generation = compute_embed_generation(embed_model, source_chunks)

    def check_cancel() -> None:
        if cancel_check:
            cancel_check()

    def check_source_still_current() -> None:
        if resolved_source_root is None:
            return
        current_chunks, current_manifest = build_source_snapshot(resolved_source_root)
        if (
            current_manifest != resolved_manifest
            or _chunk_snapshot_signature(current_chunks) != _chunk_snapshot_signature(source_chunks)
        ):
            raise EmbedBuildError(
                "SOURCE_CHANGED_DURING_BUILD",
                "source snapshot changed before cache promotion",
            )

    def check_model_still_current() -> None:
        if model_identity is None:
            return
        current_identity = get_embed_model_identity(embed_model)
        if (
            not isinstance(current_identity, dict)
            or current_identity.get("digest") != model_identity.get("digest")
            or _canonical_model_reference(str(current_identity.get("name", "")))
            != _canonical_model_reference(str(model_identity.get("name", "")))
        ):
            raise EmbedBuildError(
                "MODEL_CHANGED_DURING_BUILD",
                "embedding model identity changed before cache promotion",
            )

    with _embed_writer_lock() as lock_acquired:
        if not lock_acquired:
            existing = _load_embed_cache_impl(use_memo=False)
            if (
                EMBED_CACHE_PATH.exists()
                and _cache_header_matches(existing, embed_model, generation)
                and _cache_ready_compatibility_matches(existing, embed_model, model_identity)
                and existing.get("files") == resolved_manifest
                and set(existing.get("entries", {}))
                == {chunk.get("chunk_id") for chunk in source_chunks}
            ):
                _emit_embed_progress(
                    emit_progress,
                    phase="progress",
                    processed=total,
                    total=total,
                    generated=0,
                    reused=total,
                    failed=0,
                    checkpointed=0,
                    embedding_seconds=0.0,
                )
                return existing
            raise EmbedIndexBusyError("Embeddingインデックスを更新中です")

        check_cancel()
        cache = _load_embed_cache_impl(use_memo=False)
        state, checkpoint_entries = _load_checkpoint(
            embed_model,
            generation,
            mode=mode,
            model_identity=model_identity,
        )
        selection = _select_reusable_embeddings(
            cache,
            embed_model,
            model_identity,
            mode,
            source_chunks,
            resolved_manifest,
            checkpoint_entries,
        )
        if mode == "incremental" and not selection.cache_compatible and cache.get("entries"):
            print(
                "[INFO] Embeddingキャッシュの互換性を確認できないため、全件再構築します"
                + ("（モデルdigestを取得できません）。" if model_identity is None else "。"),
                file=sys.stderr,
            )
        # 旧cacheは読み取り専用の再利用元としてだけ使い、候補の組み立て後は参照を捨てる。
        cache = None
        checkpoint_entries = None
        expected_dimension = selection.expected_dimension
        reusable = selection.entries
        valid_checkpoint_count = selection.from_checkpoint

        entries: dict[str, dict] = dict(reusable)
        processed = len(reusable)
        generated = 0
        failed = 0
        checkpointed = valid_checkpoint_count
        embedding_seconds = 0.0
        last_progress_processed = processed
        last_progress_time = time.monotonic()
        pending_batch: list[dict] = []

        def progress_values() -> dict:
            return {
                "processed": processed,
                "total": total,
                "generated": generated,
                "reused": len(reusable),
                "failed": failed,
                "checkpointed": checkpointed,
                "embedding_seconds": round(embedding_seconds, 3),
            }

        _emit_embed_progress(
            emit_progress,
            phase="resumed" if valid_checkpoint_count else "start",
            **progress_values(),
        )

        def emit_progress_if_due(force: bool = False) -> None:
            nonlocal last_progress_processed, last_progress_time
            now = time.monotonic()
            if force or (
                processed - last_progress_processed >= EMBED_PROGRESS_INTERVAL
                or now - last_progress_time >= EMBED_PROGRESS_SECONDS
            ):
                _emit_embed_progress(emit_progress, phase="progress", **progress_values())
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
                print("[WARN] Embedding checkpointの保存に失敗しました。", file=sys.stderr)
                return
            checkpointed += len(batch)
            pending_batch = pending_batch[len(batch) :]
            _emit_embed_progress(emit_progress, phase="checkpoint", **progress_values())

        try:
            pending_chunks: list[dict] = []
            effective_batch_size = (
                1 if _get_embedding is not _ORIGINAL_GET_EMBEDDING else EMBED_BATCH_SIZE
            )

            def process_batch(batch_chunks: list[dict]) -> None:
                nonlocal processed, generated, failed, expected_dimension, embedding_seconds
                if not batch_chunks:
                    return
                check_cancel()
                batch_started = time.monotonic()
                try:
                    vectors = _get_index_embedding_batch(
                        [chunk["text"] for chunk in batch_chunks], embed_model
                    )
                finally:
                    # 見積もり用の速度は、走査・cache読込/保存を除いた生成時間で測る。
                    embedding_seconds += time.monotonic() - batch_started
                check_cancel()
                for chunk, embedding in zip(batch_chunks, vectors):
                    chunk_id = chunk["chunk_id"]
                    processed += 1
                    candidate = (
                        {**_strip_cache_metadata(chunk), "embedding": embedding}
                        if embedding is not None
                        else None
                    )
                    valid = _valid_embedding_entry(candidate, chunk_id) if candidate else False
                    if valid and expected_dimension is not None:
                        valid = len(candidate["embedding"]) == expected_dimension
                    if valid:
                        if expected_dimension is None:
                            expected_dimension = len(candidate["embedding"])
                        entries[chunk_id] = candidate
                        pending_batch.append(candidate)
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
            flush_pending(force=True)
            if failed:
                raise EmbedBuildError(
                    "EMBED_BATCH_FAILED",
                    f"{failed} embedding item(s) failed; final cache was not promoted",
                )
            check_model_still_current()
            candidate_cache = _empty_embed_cache()
            candidate_cache.update(
                {
                    "embed_model": embed_model,
                    "generation": generation,
                    "compatibility": _new_embed_compatibility(
                        embed_model,
                        model_identity,
                        # entryが0件（全削除・全空ファイル）の候補は次元を宣言しない。
                        expected_dimension if entries else None,
                    ),
                    "files": resolved_manifest,
                    "entries": entries,
                }
            )
            _validate_embed_cache_candidate(
                candidate_cache,
                embed_model,
                model_identity,
                generation,
                source_chunks,
                resolved_manifest,
            )
            promoted: dict = {}

            def verify_written_candidate(tmp_path: Path) -> None:
                # 同一directoryの一時ファイルを再読込して検証し、昇格直前に資料の
                # 実hashとモデル識別を再確認する。失敗時は置換せず旧cacheを残す。
                written = _load_embed_cache_file(tmp_path)
                _validate_embed_cache_candidate(
                    written,
                    embed_model,
                    model_identity,
                    generation,
                    source_chunks,
                    resolved_manifest,
                )
                check_source_still_current()
                check_model_still_current()
                check_cancel()
                promoted["cache"] = written

            if not _atomic_write_json(
                EMBED_CACHE_PATH, candidate_cache, before_replace=verify_written_candidate
            ):
                raise EmbedCachePersistenceError(
                    "最終Embedding cacheの保存に失敗しました。checkpointを保持します。"
                )
            _invalidate_embed_cache_memo()
            _remove_checkpoint_files(include_state=True)
            _emit_embed_progress(emit_progress, phase="completed", **progress_values())
            return promoted["cache"]
        except BaseException:
            try:
                flush_pending(force=True)
            finally:
                _emit_embed_progress(emit_progress, phase="interrupted", **progress_values())
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


def _match_group_key(match: dict) -> str:
    """独立根拠としての同一性キー。親子展開の同一groupは1件として数える。"""
    group_id = match.get("group_id")
    if group_id:
        return f"group:{group_id}"
    chunk_id = match.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    return (
        f"path:{normalize_source_path(match.get('path', ''))}:"
        f"{match.get('start_line')}:{match.get('end_line')}"
    )


def _collapse_to_independent_groups(matches: list[dict]) -> list[dict]:
    """confidence算出用に、同一groupのitemを1件の独立根拠へ集約する。

    子の増加を独立した根拠の増加として数えないため。group内の代表は最初の
    itemとし、rrf_score/embedding_score/keyword_scoreは含めない（親のスコアを
    子の実測スコアとして流用しない）。group以外（直接ヒット）のitemは
    そのまま1件として扱う。
    """
    collapsed: list[dict] = []
    seen_groups: set[str] = set()
    for match in matches:
        group_id = match.get("group_id")
        if not group_id:
            collapsed.append(match)
            continue
        if group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        representative = {
            key: value
            for key, value in match.items()
            if key not in {"rrf_score", "embedding_score", "keyword_score"}
        }
        collapsed.append(representative)
    return collapsed


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
    independent = _collapse_to_independent_groups(matches)
    non_empty = sum(1 for m in independent if m.get("snippet")) / max(
        len(independent), 1
    )
    top_score = float(independent[0].get("rrf_score", independent[0].get("score", 0)) or 0)
    file_count = len({normalize_source_path(m.get("path", "")) for m in independent})
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
    # 展開item自体（source="expanded"）は親のEmbeddingスコアを流用しないため
    # ここには数えない（_has_complete_structural_evidence が別経路で扱う）。
    has_semantic_evidence = any(
        "embedding" in str(m.get("source", "")) for m in independent
    )
    if len(independent) >= 2 and confidence >= 0.55 and has_semantic_evidence:
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


# ---------------------------------------------------------------------------
# 親子展開（見出しだけの親チャンクから配下本文への展開）
#
# 検索チャンク（_line_chunks）自体は変更しない。既存の embedding_search 等が
# 返す「見出しだけの親candidate」に対し、同一資料snapshotから見出しツリーを
# 解析し、配下の子見出しの本文範囲を実在する行範囲として展開する。
#
# finalize_ranked_matches / run_retrieval_pipeline に統合済み。既定ON
# （2026-09-14、P2実機比較受入 result/offline-ai/parent-child-expansion-p2-20260913/
# の結果に基づく利用者判断）。OFFLINE_AI_PARENT_CHILD_EXPANSION=false で無効化できる。
# ---------------------------------------------------------------------------

EXPANSION_MAX_PARENTS_PER_QUERY = env_int(
    "OFFLINE_AI_EXPANSION_MAX_PARENTS", 2, min_value=1
)
EXPANSION_MAX_RANGES_PER_PARENT = env_int(
    "OFFLINE_AI_EXPANSION_MAX_RANGES_PER_PARENT", 4, min_value=1
)
EXPANSION_MAX_TOTAL_RANGES = env_int(
    "OFFLINE_AI_EXPANSION_MAX_TOTAL_RANGES", 4, min_value=1
)
EXPANSION_BUDGET_RATIO = env_float(
    "OFFLINE_AI_EXPANSION_BUDGET_RATIO", 0.5, min_value=0.0, max_value=1.0
)


def _parent_child_expansion_enabled() -> bool:
    return _env_bool("OFFLINE_AI_PARENT_CHILD_EXPANSION", True)


# ファイル単位の見出しツリー memo。source chunk memo と同じ世代キー
# （_source_chunk_memo_key の戻り値）でライフサイクルを揃える。ディスクへは
# 保存しない。要求ごとの展開選択結果（どの子を採用したか）はここへ入れない。
_STRUCTURE_FILE_MEMO_KEY: tuple | None = None
_STRUCTURE_FILE_MEMO_VALUE: dict[str, tuple[list[str], list, str] | None] = {}


def _invalidate_structure_memo() -> None:
    global _STRUCTURE_FILE_MEMO_KEY, _STRUCTURE_FILE_MEMO_VALUE
    _STRUCTURE_FILE_MEMO_KEY = None
    _STRUCTURE_FILE_MEMO_VALUE = {}


def _read_structure_for_file(
    rel_path: str, source_root: Path | None
) -> tuple[list[str], list, str] | None:
    """指定ファイルを読み込み見出しツリーを構築する。読めなければ None。

    ``_chunk_source_bytes`` と同じ decode 規則（utf-8, errors="replace"）を
    使い、同一 bytes から生成した派生情報であることを保つ。
    """
    root = source_root or SKILL_SOURCE_DIR
    path = root / rel_path
    try:
        data = path.read_bytes()
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    nodes = _source_structure.parse_heading_tree(text)
    file_sha256 = hashlib.sha256(data).hexdigest()
    return lines, nodes, file_sha256


def _get_structure_for_file(
    rel_path: str, source_root: Path | None, memo_key: tuple | None
) -> tuple[list[str], list, str] | None:
    """構造ツリーを要求内 memo 付きで取得する。memo 無効時は毎回読み込む。"""
    global _STRUCTURE_FILE_MEMO_KEY, _STRUCTURE_FILE_MEMO_VALUE
    if not SOURCE_CHUNK_MEMO_ENABLED or memo_key is None:
        return _read_structure_for_file(rel_path, source_root)
    if memo_key != _STRUCTURE_FILE_MEMO_KEY:
        _STRUCTURE_FILE_MEMO_KEY = memo_key
        _STRUCTURE_FILE_MEMO_VALUE = {}
    if rel_path in _STRUCTURE_FILE_MEMO_VALUE:
        return _STRUCTURE_FILE_MEMO_VALUE[rel_path]
    result = _read_structure_for_file(rel_path, source_root)
    _STRUCTURE_FILE_MEMO_VALUE[rel_path] = result
    return result


def _child_chunks_in_range(
    source_chunks: list[dict], path: str, start_line: int, end_line: int
) -> list[dict]:
    """配下範囲内にある既存チャンクを原文順（start_line昇順）で返す。"""
    candidates = [
        chunk
        for chunk in source_chunks
        if normalize_source_path(chunk.get("path", "")) == path
        and isinstance(chunk.get("start_line"), int)
        and isinstance(chunk.get("end_line"), int)
        and chunk["start_line"] >= start_line
        and chunk["end_line"] <= end_line
    ]
    candidates.sort(key=lambda chunk: chunk["start_line"])
    return candidates


def _merge_contiguous_line_ranges(
    line_ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """行範囲が連続する（隣接・重複する）チャンクを1つの範囲へまとめる。"""
    merged: list[tuple[int, int]] = []
    for start, end in line_ranges:
        if merged and start <= merged[-1][1] + 1:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def _range_text_from_lines(lines: list[str], start_line: int, end_line: int) -> str:
    """行範囲から本文を再構成する。チャンクの overlap 由来の重複を避けるため、
    チャンクの ``text`` フィールドではなく原資料の行から都度組み立てる。"""
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return "\n".join(lines[start_idx:end_idx])


def _expansion_group_id(file_sha256: str, path: str, parent_heading_line: int) -> str:
    """資料hash・path・親見出し行から決定的に group_id を生成する。"""
    payload = f"{file_sha256}:{path}:{parent_heading_line}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _build_chunk_id_lookup(index: "_EmbedIndex") -> dict[str, int]:
    """``_EmbedIndex.chunk_ids`` から位置を引く要求内lookup。index・共有memoは変更しない。"""
    return {chunk_id: position for position, chunk_id in enumerate(index.chunk_ids)}


def _max_similarity_to_context(
    chunk_id: str,
    chunk_lookup: dict[str, int],
    embed_index: "_EmbedIndex",
    query_context: QueryContext,
) -> float | None:
    """指定chunkの、その試行の全query vectorとの類似度の最大値を返す。

    子の類似度は選択順位だけに使い、展開資格の足切りにしない（閾値なし採点）。
    次元不一致・ゼロnorm・欠損は None（無効）として返し、呼び出し側は原文順
    fallbackへ戻す。
    """
    position = chunk_lookup.get(chunk_id)
    if position is None:
        return None
    vector = embed_index.vectors[position]
    norm = embed_index.norms[position]
    if norm == 0.0:
        return None
    best: float | None = None
    for query in query_context.queries:
        query_vector = query_context.vectors.get(query)
        if query_vector is None or len(query_vector) != len(vector):
            continue
        query_norm = math.sqrt(sum(v * v for v in query_vector))
        if query_norm == 0.0:
            continue
        similarity = sum(a * b for a, b in zip(query_vector, vector)) / (query_norm * norm)
        if best is None or similarity > best:
            best = similarity
    return best


def _order_ranges_for_selection(
    child_chunks: list[dict],
    merged_ranges: list[tuple[int, int]],
    *,
    query_context: QueryContext | None,
    embed_index: "_EmbedIndex | None",
    chunk_lookup: dict[str, int] | None,
) -> tuple[list[tuple[int, int]], str]:
    """予算超過時にどの範囲を優先して採用するかの順序を決める。

    有効な既存クエリベクトルと子の既存ベクトルが利用できる場合、子自身の
    類似度降順、同点は既存キーワードスコア降順、最後に原文順で選ぶ。複数
    チャンクからなる範囲は、配下チャンクの最大類似度を選択用に使う。
    Embedding経路が無効・失敗、または対象子のベクトルが不完全なら原文順
    （既定の fallback）で選び、理由を返す。選択順序は採用可否だけに使い、
    採用後の表示順序は呼び出し側が原文順へ戻す。
    """
    if query_context is None or embed_index is None or chunk_lookup is None:
        return list(merged_ranges), "embedding_unavailable"

    document_order = {range_: position for position, range_ in enumerate(merged_ranges)}
    scored: list[tuple[tuple[int, int], float | None, float]] = []
    any_valid_similarity = False
    for range_ in merged_ranges:
        start, end = range_
        chunks_in_range = [
            chunk
            for chunk in child_chunks
            if chunk["start_line"] >= start and chunk["end_line"] <= end
        ]
        best_similarity: float | None = None
        best_keyword_score = 0.0
        for chunk in chunks_in_range:
            similarity = _max_similarity_to_context(
                str(chunk.get("chunk_id", "")), chunk_lookup, embed_index, query_context
            )
            if similarity is not None:
                any_valid_similarity = True
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity
            keyword_score = float(chunk.get("keyword_score", 0) or 0)
            if keyword_score > best_keyword_score:
                best_keyword_score = keyword_score
        scored.append((range_, best_similarity, best_keyword_score))

    if not any_valid_similarity:
        return list(merged_ranges), "embedding_unavailable"

    def sort_key(item: tuple[tuple[int, int], float | None, float]) -> tuple[float, float, int]:
        range_, similarity, keyword_score = item
        similarity_value = similarity if similarity is not None else float("-inf")
        return (-similarity_value, -keyword_score, document_order[range_])

    ordered = sorted(scored, key=sort_key)
    return [item[0] for item in ordered], ""


def _expand_single_parent(
    parent_match: dict,
    source_chunks: list[dict],
    *,
    source_root: Path | None,
    memo_key: tuple | None,
    max_ranges: int,
    char_budget: int | None,
    query_context: QueryContext | None = None,
    embed_index: "_EmbedIndex | None" = None,
) -> tuple[list[dict], bool, str]:
    """1件の親candidateを展開する。戻り値は ``(展開item群, partialか, 理由)``。

    展開item は ``group_id`` / ``expanded_from`` / ``group_order`` を持つ、
    通常の match と同じ形の dict。

    選択の単位は親の「直接の子見出し」（孫を含む自身の節全体）である。
    行範囲上は連続していても、選択候補としては別々に扱う
    （``_source_structure.direct_child_ranges``）。予算超過時の優先順位は
    ``_order_ranges_for_selection`` が決め、``query_context`` / ``embed_index``
    が利用できない場合は常に原文順（見出し出現順）で選ぶ。採用された子節に
    属する既存チャンクは、行範囲が連続するものを1つの表示範囲へまとめる
    （表示は常に原文順）。
    """
    path = normalize_source_path(parent_match.get("path", ""))
    start_line = parent_match.get("start_line")
    if not path or not isinstance(start_line, int):
        return [], False, "parent_missing_location"

    structure = _get_structure_for_file(path, source_root, memo_key)
    if structure is None:
        return [], False, "structure_unavailable"
    lines, nodes, file_sha256 = structure

    expected_sha = str(
        parent_match.get("source_sha256") or parent_match.get("file_sha256") or ""
    )
    if expected_sha and expected_sha.lower() != file_sha256.lower():
        return [], False, "source_changed"

    parent_node = _source_structure.find_expandable_parent(
        nodes, lines, start_line=start_line
    )
    if parent_node is None:
        return [], False, "not_expandable"

    expand_start, expand_end = _source_structure.expand_range(nodes, parent_node)
    all_child_chunks = _child_chunks_in_range(source_chunks, path, expand_start, expand_end)
    if not all_child_chunks:
        return [], False, "no_child_chunks"

    candidate_sections = _source_structure.direct_child_ranges(nodes, parent_node)
    if not candidate_sections:
        candidate_sections = [(expand_start, expand_end)]

    group_id = _expansion_group_id(file_sha256, path, parent_node.heading_line)
    parent_chunk_id = str(parent_match.get("chunk_id") or "")
    modified_at = parent_match.get("modifiedAt", "")

    chunk_lookup = _build_chunk_id_lookup(embed_index) if embed_index is not None else None
    selection_order, _selection_reason = _order_ranges_for_selection(
        all_child_chunks,
        candidate_sections,
        query_context=query_context,
        embed_index=embed_index,
        chunk_lookup=chunk_lookup,
    )
    chosen_sections = selection_order[:max_ranges]
    partial = len(candidate_sections) > max_ranges

    selected_chunks = [
        chunk
        for chunk in all_child_chunks
        if any(
            section_start <= chunk["start_line"] and chunk["end_line"] <= section_end
            for section_start, section_end in chosen_sections
        )
    ]
    # 採用された子節の表示は選択順序に関わらず常に原文順。行範囲が連続する
    # チャンクは1つの実在範囲へまとめる（離れた子節を1つの広い start/end に
    # 偽装しない）。
    line_ranges = [(chunk["start_line"], chunk["end_line"]) for chunk in selected_chunks]
    merged_ranges = _merge_contiguous_line_ranges(line_ranges)

    items: list[dict] = []
    used_chars = 0
    order = 0
    for start, end in merged_ranges:
        text = _range_text_from_lines(lines, start, end)
        if char_budget is not None:
            remaining = char_budget - used_chars
            if remaining <= 0:
                partial = True
                break
            if len(text) > remaining:
                text = text[:remaining]
                partial = True
        used_chars += len(text)
        order += 1
        items.append(
            {
                "path": path,
                "chunk_id": f"{group_id}#r{order:02d}",
                "heading": parent_node.heading_text,
                "start_line": start,
                "end_line": end,
                "snippet": text,
                "modifiedAt": modified_at,
                "source_sha256": file_sha256,
                "source": "expanded",
                "group_id": group_id,
                "expanded_from": parent_chunk_id,
                "group_order": order,
            }
        )
    reason = "" if items else "budget_exhausted"
    for item in items:
        item["group_partial"] = partial
    return items, partial, reason


def expand_parent_candidates(
    matches: list[dict],
    source_chunks: list[dict],
    *,
    source_root: Path | None = None,
    memo_key: tuple | None = None,
    max_parents: int | None = None,
    max_ranges_per_parent: int | None = None,
    max_total_ranges: int | None = None,
    char_budget: int | None = None,
    query_context: QueryContext | None = None,
    embed_index: "_EmbedIndex | None" = None,
) -> tuple[list[dict], list[dict]]:
    """支持判定・件数制限後の候補から、展開資格のある親candidateを展開する。

    戻り値は ``(expanded_items, expansion_meta)``。呼び出し順位の高い候補から
    ``max_parents`` 件まで、各展開元は ``max_ranges_per_parent`` 範囲まで、
    全体で ``max_total_ranges`` 範囲までを上限とする。直接ヒットとの重複排除
    は行わない（呼び出し側の責務）。``query_context`` / ``embed_index`` が
    利用できる場合は子の類似度で予算超過時の優先順位付けを行い、利用できない
    場合は原文順で選ぶ。
    """
    max_parents = (
        EXPANSION_MAX_PARENTS_PER_QUERY if max_parents is None else max_parents
    )
    max_ranges_per_parent = (
        EXPANSION_MAX_RANGES_PER_PARENT
        if max_ranges_per_parent is None
        else max_ranges_per_parent
    )
    max_total_ranges = (
        EXPANSION_MAX_TOTAL_RANGES if max_total_ranges is None else max_total_ranges
    )
    per_parent_budget = None
    if char_budget is not None and max_parents > 0:
        per_parent_budget = max(0, char_budget // max_parents)

    expanded_items: list[dict] = []
    expansion_meta: list[dict] = []
    parent_count = 0
    total_ranges = 0
    for match in matches:
        if parent_count >= max_parents or total_ranges >= max_total_ranges:
            break
        remaining_ranges = min(max_ranges_per_parent, max_total_ranges - total_ranges)
        if remaining_ranges <= 0:
            break
        items, partial, reason = _expand_single_parent(
            match,
            source_chunks,
            source_root=source_root,
            memo_key=memo_key,
            max_ranges=remaining_ranges,
            char_budget=per_parent_budget,
            query_context=query_context,
            embed_index=embed_index,
        )
        if not items:
            if reason and reason != "not_expandable":
                expansion_meta.append(
                    {
                        "path": normalize_source_path(match.get("path", "")),
                        "start_line": match.get("start_line"),
                        "expanded": False,
                        "reason": reason,
                    }
                )
            continue
        parent_count += 1
        total_ranges += len(items)
        expanded_items.extend(items)
        expansion_meta.append(
            {
                "path": normalize_source_path(match.get("path", "")),
                "start_line": match.get("start_line"),
                "expanded": True,
                "partial": partial,
                "range_count": len(items),
                "group_id": items[0]["group_id"],
            }
        )
    return expanded_items, expansion_meta


def _limit_items_preserving_groups(items: list[dict], limit: int) -> list[dict]:
    """件数上限を適用する。同一 group の item は分断せず全採用/全除外を揃える。

    離れた複数の子節を一つの広い start/end に偽装しないという契約上、
    group（親子展開の1親候補分の範囲群）を件数上限の途中で切ると引用が
    矛盾する。順位順に評価し、残枠に収まる item・group だけを採用する。
    """
    if limit <= 0:
        return []
    result: list[dict] = []
    used = 0
    decided_groups: set[str] = set()
    for item in items:
        if used >= limit:
            break
        group_id = item.get("group_id")
        if not group_id:
            result.append(item)
            used += 1
            continue
        if group_id in decided_groups:
            continue
        group_items = [m for m in items if m.get("group_id") == group_id]
        decided_groups.add(group_id)
        if used + len(group_items) <= limit:
            result.extend(group_items)
            used += len(group_items)
    return result


def _dedupe_expansion_against_direct_hits(
    expanded_items: list[dict], direct_matches: list[dict]
) -> list[dict]:
    """直接ヒットと展開範囲が重なる場合は直接ヒットを優先し、展開itemを除外する。

    重複の削減で独立根拠数を増やさない（直接ヒットは既に direct_matches に
    含まれているため、重なる展開itemを足しても件数は増えない）。
    """
    kept = []
    for item in expanded_items:
        path = item.get("path")
        start, end = item.get("start_line"), item.get("end_line")
        if not isinstance(start, int) or not isinstance(end, int):
            kept.append(item)
            continue
        overlaps_direct = any(
            normalize_source_path(m.get("path", "")) == path
            and isinstance(m.get("start_line"), int)
            and isinstance(m.get("end_line"), int)
            and m["start_line"] <= end
            and start <= m["end_line"]
            for m in direct_matches
        )
        if overlaps_direct:
            continue
        kept.append(item)
    return kept


def _has_complete_structural_evidence(
    matches: list[dict], must_find_terms: list[str] | None
) -> bool:
    """完全展開されたgroupが1件でもあり、必須語が全て充足されているか判定する。

    親の語彙一致と完全展開だけでは自動合格にしない: must_find_terms が
    指定されている場合は、その group の採用本文で全て充足する必要がある。
    """
    terms = [str(t).strip().lower() for t in (must_find_terms or []) if str(t).strip()]
    seen_groups: set[str] = set()
    for match in matches:
        if match.get("source") != "expanded":
            continue
        group_id = match.get("group_id")
        if not group_id or group_id in seen_groups or match.get("group_partial"):
            continue
        seen_groups.add(group_id)
        if not terms:
            return True
        group_text = " ".join(
            (m.get("snippet", "") or "")
            for m in matches
            if m.get("group_id") == group_id
        ).lower()
        if all(term in group_text for term in terms):
            return True
    return False


def finalize_ranked_matches(
    query: str,
    matches: list[dict],
    *,
    must_find_terms: list[str] | None = None,
    max_per_file: int = MAX_CHUNKS_PER_FILE,
    relative_score_floor: float = 0.3,
    source_chunks: list[dict] | None = None,
    source_root: Path | None = None,
    memo_key: tuple | None = None,
    char_budget: int | None = None,
    query_context: QueryContext | None = None,
    embed_index: "_EmbedIndex | None" = None,
) -> tuple[list[dict], float, str]:
    """file別上限・相対スコア足切り・親子展開・confidence算出までの決定的な後段。

    rerank後の順位付き候補を受け取り、`(matches, confidence, evidence_status)` を返す。
    処理順序: 支持判定 → 制限前矛盾検出 → file別上限 → 相対スコア足切り →
    親子展開（既定OFF・source_chunks指定時のみ）→ confidence算出。

    ``source_chunks`` を渡さない（既定）場合は展開ステップを一切実行せず、
    既存呼び出し元との挙動を完全に保つ。``query_context`` / ``embed_index`` は
    展開の予算超過時に子の類似度で優先順位付けする（省略時は原文順）。
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

    expansion_enabled = _parent_child_expansion_enabled() and source_chunks is not None
    if expansion_enabled:
        expansion_budget = (
            int(char_budget * EXPANSION_BUDGET_RATIO) if char_budget else None
        )
        expanded_items, _meta = expand_parent_candidates(
            limited,
            source_chunks,
            source_root=source_root,
            memo_key=memo_key,
            char_budget=expansion_budget,
            query_context=query_context,
            embed_index=embed_index,
        )
        expanded_items = _dedupe_expansion_against_direct_hits(expanded_items, limited)
        if expanded_items:
            limited = limited + expanded_items
            constraint_conflict = constraint_conflict or _has_explicit_constraint_conflict(
                query, expanded_items
            )

    confidence, status = _calculate_confidence(query, limited, must_find_terms)
    if (
        status != "sufficient"
        and expansion_enabled
        and not constraint_conflict
        and confidence >= 0.55
        and _has_complete_structural_evidence(limited, must_find_terms)
    ):
        status = "sufficient"
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
    expansion_enabled = _parent_child_expansion_enabled() and source_chunks is not None
    # 親子展開の構造 memo キー。既定 OFF では計算しない（追加のファイル走査を避ける）。
    source_memo_key = (
        _source_chunk_memo_key(SKILL_SOURCE_DIR) if expansion_enabled else None
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
    evidence_items: list[dict] = []
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
        query_context: QueryContext | None = None
        if embed_model and embed_index is not None:
            try:
                precomputed_by_query, query_context = embedding_search_multi_with_context(
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
                query_context = None
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
        # ranked_candidates（展開前）は combined に保持し、次試行の merge と
        # _retry_queries へはこちらだけを渡す。展開後の採用根拠は
        # evidence_items として毎試行 combined の同一snapshotから作り直し、
        # 再展開で件数やIDが増殖しない決定的処理にする（展開後の根拠を
        # 検索候補へ書き戻すと再統合を招くため）。
        evidence_items, final_confidence, final_status = finalize_ranked_matches(
            query,
            combined,
            must_find_terms=plan.get("must_find_terms", []),
            source_chunks=source_chunks if expansion_enabled else None,
            source_root=SKILL_SOURCE_DIR,
            memo_key=source_memo_key,
            char_budget=PROMPT_EVIDENCE_CHAR_LIMIT,
            query_context=query_context if expansion_enabled else None,
            embed_index=embed_index if expansion_enabled else None,
        )
        attempts.append(
            RetrievalAttempt(
                query=" / ".join(queries),
                keywords=keywords,
                match_count=len(evidence_items),
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
        # 既存の件数設定だけを適用する（既定5、最大8）。同一 group の item は
        # 分断せず、全採用/全除外のどちらかにする。
        matches = _limit_items_preserving_groups(
            evidence_items, min(RETRIEVAL_PROMPT_MATCH_LIMIT, 8)
        )
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
    matches = _limit_items_preserving_groups(evidence_items, RETRIEVAL_PROMPT_MATCH_LIMIT)
    matches = _fit_prompt_budget(matches, evidence_char_limit)
    if len(matches) < len(evidence_items):
        # 最終出力時の追加切詰め。制限前に検出済みの矛盾は維持したまま、
        # 実際に採用した本文だけで信頼度・sufficient判定を再照合する。
        had_conflict = bool(evidence_items and evidence_items[0].get("constraint_conflict"))
        final_confidence, final_status = _calculate_confidence(
            query, matches, plan.get("must_find_terms", [])
        )
        if had_conflict:
            if final_status == "sufficient":
                final_status = "partial"
            if matches:
                matches[0] = {**matches[0], "constraint_conflict": True}
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
