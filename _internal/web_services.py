# 依存方向: web.bat → web_server.py → web_services.py → search.py
# 逆方向の import は禁止（レイヤー違反）
"""
サービスアダプタ層。

search.py への直接依存を隔離し、
import 時副作用の吸収・例外変換・構造化ログ出力を担う。
web_server.py はこのモジュールのみを import する。

共有設定 config.py の検証失敗は吸収せず、起動時に fail-fast する。
"""

import json
from datetime import datetime, timezone
import inspect
import logging
import os
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from config import OLLAMA_HOST, DEEP_SEARCH_TIMEOUT_DEFAULT
from source_view import EvidenceRegistry, EvidenceViewError  # noqa: F401 - web_server re-exports this service boundary error

logger = logging.getLogger("offlineai.services")

SCRIPT_DIR = Path(__file__).resolve().parent


VALID_SEARCH_MODES = {"answer", "search", "deep"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def create_evidence_registry(source_root: Path | None = None) -> EvidenceRegistry:
    return EvidenceRegistry(source_root or (SCRIPT_DIR.parent / "skill-source"))


def view_evidence(registry: EvidenceRegistry, evidence_id: object, session_id: str) -> dict:
    """根拠 registry から安全な閲覧結果を取得するサービス境界。"""
    return registry.view(evidence_id, session_id=session_id)


def _verification_note(retrieval) -> str:
    """検証で根拠ステータスを格下げした場合の注記。search を読めなければ空文字。"""
    try:
        from search import verification_note
    except Exception:
        return ""
    return verification_note(retrieval)


def build_evidence_event(
    retrieval,
    *,
    evidence_registry: EvidenceRegistry | None = None,
    session_id: str = "",
) -> dict:
    """RetrievalResultをWeb公開用の最小・JSON安全な根拠イベントへ変換する。"""
    items = []
    warnings: list[str] = [str(w).strip() for w in getattr(retrieval, "warnings", []) if str(w).strip()]
    for match in list(retrieval.matches)[:8]:
        source_sha256 = str(
            match.get("source_sha256") or match.get("file_sha256") or ""
        ).strip()
        item = {
            "path": str(match.get("path") or ""),
            "sourceTitle": str(match.get("source_title") or ""),
            "heading": str(match.get("heading") or ""),
            "page": match.get("page"),
            "startLine": match.get("start_line"),
            "endLine": match.get("end_line"),
            "layoutType": str(match.get("layout_type") or ""),
            "parser": str(match.get("parser") or ""),
            "source": str(match.get("source") or ""),
            "snippet": str(match.get("snippet") or "")[:500],
            "sourceSha256": source_sha256,
        }
        group_id = match.get("group_id")
        if group_id:
            item["groupId"] = str(group_id)
            item["expandedFrom"] = str(match.get("expanded_from") or "")
            item["groupOrder"] = match.get("group_order")
            item["groupPartial"] = bool(match.get("group_partial"))
        if evidence_registry is not None:
            evidence_id = evidence_registry.register(
                session_id=session_id,
                relative_path=item["path"],
                start_line=item["startLine"],
                end_line=item["endLine"],
                source_sha256=source_sha256,
            )
            if evidence_id:
                item["evidenceId"] = evidence_id
                item["viewable"] = True
            else:
                item["viewable"] = False
                item["viewReason"] = "hashまたは行番号を確認できないため閲覧できません"
        item_warnings = []
        for warning in match.get("parser_warnings") or []:
            normalized = str(warning).strip()[:500]
            if not normalized:
                continue
            item_warnings.append(normalized)
            if normalized not in warnings:
                warnings.append(normalized)
        item["parserWarnings"] = item_warnings[:10]
        items.append(item)
    route_reason = str(getattr(retrieval, "route_reason", "") or "").strip()
    if route_reason and route_reason not in warnings:
        warnings.append(route_reason)
    if getattr(retrieval, "route", "keyword") == "keyword" and not route_reason:
        fallback_warning = "Embedding未準備のためキーワード検索のみ"
        if fallback_warning not in warnings:
            warnings.append(fallback_warning)
    return {
        "type": "evidence",
        "evidenceStatus": str(retrieval.evidence_status),
        "verificationNote": _verification_note(retrieval),
        "confidence": float(retrieval.confidence),
        "route": str(getattr(retrieval, "route", "keyword")),
        "indexState": str(getattr(retrieval, "index_state", "missing")),
        "routeReason": route_reason,
        "items": items,
        "warnings": list(dict.fromkeys(warnings))[:9],
    }


def build_deep_evidence_event(
    result,
    *,
    evidence_registry: EvidenceRegistry | None = None,
    session_id: str = "",
) -> dict:
    """deep結果を既存の根拠表示契約へ変換する。本文は500字まで表示する。"""
    items = []
    for evidence in list(getattr(result, "evidence", [])):
        path = str(evidence.get("path") or "")
        source_sha256 = str(evidence.get("source_sha256") or "")
        item = {
            "path": path,
            "sourceTitle": path,
            "heading": str(evidence.get("heading") or ""),
            "startLine": evidence.get("start_line"),
            "endLine": evidence.get("end_line"),
            "snippet": str(evidence.get("excerpt") or "")[:500],
            "sourceSha256": source_sha256,
            "evidenceId": str(evidence.get("evidence_id") or ""),
            "deepStatus": str(evidence.get("verification_status") or ""),
            "viewable": False,
        }
        if evidence_registry is not None:
            evidence_id = evidence_registry.register(
                session_id=session_id,
                relative_path=path,
                start_line=item["startLine"],
                end_line=item["endLine"],
                source_sha256=source_sha256,
            )
            if evidence_id:
                item["evidenceId"] = evidence_id
                item["viewable"] = True
            else:
                item["viewReason"] = "hashまたは行番号を確認できないため閲覧できません"
        items.append(item)
    return {
        "type": "evidence",
        "evidenceStatus": str(getattr(result, "status", "partial")),
        "verificationNote": "deepは原文行範囲とSHA-256を照合した根拠だけを表示します",
        "confidence": 1.0 if items else 0.0,
        "route": "deep",
        "indexState": "deep",
        "routeReason": str(getattr(result, "stop_reason", "")),
        "items": items,
        "warnings": list(getattr(result, "unconfirmed", [])),
        "deepLedger": list(getattr(result, "ledger", [])),
        "deepDiagnostics": dict(getattr(result, "diagnostics", {}) or {}),
    }

def _get_ollama_host() -> str:
    """Return the shared loopback-only Ollama API base URL."""
    return OLLAMA_HOST

# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------

class CancelledError(Exception):
    """タイムアウト・stall・クライアント切断によるキャンセル。

    ``code`` は SSE の error イベントへそのまま渡され、"cancelled" /
    "timeout" / "stall" を区別する。
    """

    def __init__(self, message: str, *, code: str = "timeout"):
        super().__init__(message)
        self.code = code


class CancellationToken:
    """deadline ベースのキャンセルトークン。

    threading.Event + deadline で統一的にタイムアウトとキャンセルを管理する。
    ``stall_timeout`` を指定すると、``touch()`` で更新される直近活動時刻
    からの無通信時間も監視する（生成フェーズの stall 検出用）。全体
    deadline はハング防止の fail-safe として touch() では延びない。
    """

    def __init__(self, timeout: float, *, stall_timeout: float | None = None,
                 defer_stall: bool = False):
        self._event = threading.Event()
        self.timeout_seconds = timeout
        self.deadline = time.monotonic() + timeout
        self._stall_timeout = stall_timeout
        self._stall_started = not defer_stall
        self._activity_lock = threading.Lock()
        self._last_activity = time.monotonic()

    def cancel(self):
        """キャンセルを要求する。"""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def remaining(self) -> float:
        """deadline までの残り秒数を返す。0 以下ならタイムアウト。"""
        return max(0.0, self.deadline - time.monotonic())

    def touch(self) -> None:
        """直近活動時刻を更新する（thinking/contentデルタの受信も活動に含む）。"""
        with self._activity_lock:
            self._last_activity = time.monotonic()

    def start_stall_monitoring(self) -> None:
        """生成開始時にstall時計を開始する。全体deadlineは延長しない。"""
        with self._activity_lock:
            self._last_activity = time.monotonic()
            self._stall_started = True

    @property
    def stall_timeout(self) -> float | None:
        """stall 上限秒数。未指定なら None。

        stall 時の利用者向け文言を run_search 経路と SSE 経路で
        揃えるために公開する。
        """
        return self._stall_timeout

    def stall_remaining(self) -> float:
        """stall deadline までの残り秒数。``stall_timeout`` 未指定なら無限大。"""
        with self._activity_lock:
            if self._stall_timeout is None or not self._stall_started:
                return float("inf")
            elapsed = time.monotonic() - self._last_activity
        return max(0.0, self._stall_timeout - elapsed)

    def check(self):
        """キャンセル・タイムアウト・stall時に CancelledError を送出する。"""
        if self._event.is_set():
            raise CancelledError("処理がキャンセルされました", code="cancelled")
        if self.remaining() <= 0:
            raise CancelledError("処理がタイムアウトしました", code="timeout")
        if self.stall_remaining() <= 0:
            raise CancelledError(
                f"モデルからの応答が{int(self._stall_timeout)}秒途絶えました",
                code="stall",
            )


# ---------------------------------------------------------------------------
# JobTable (v2): 重複実行防止・途中参加・再接続
# ---------------------------------------------------------------------------


class JobEntry:
    """ジョブテーブルの 1 エントリ。

    state: running → completed | failed | cancelled
    subscriber_queues: 複数購読者への fan-out 用キューリスト
    _event_buffer: 再接続時の replay 用バッファ
    """

    __slots__ = ("state", "cancel_token", "subscriber_queues",
                 "created_at", "completed_at", "fingerprint",
                  "_event_buffer", "_lock", "released")

    def __init__(self, cancel_token: CancellationToken, fingerprint=None):
        self.state: str = "running"
        self.cancel_token = cancel_token
        self.fingerprint = fingerprint
        self.subscriber_queues: list[queue.Queue] = []
        self.created_at: float = time.monotonic()
        self.completed_at: float | None = None
        self._event_buffer: list[dict] = []
        self._lock = threading.Lock()
        self.released = False

    def add_subscriber(self) -> queue.Queue:
        """購読者キューを追加し、バッファ済みイベントを replay する。"""
        deep = bool(self.fingerprint and len(self.fingerprint) >= 5 and self.fingerprint[4] == "deep")
        q: queue.Queue = queue.Queue(maxsize=64 if deep else 0)
        with self._lock:
            for event in self._event_buffer:
                q.put(event)
            if self.state == "running":
                self.subscriber_queues.append(q)
        return q

    def remove_subscriber(self, q: queue.Queue):
        with self._lock:
            try:
                self.subscriber_queues.remove(q)
            except ValueError:
                pass

    def broadcast(self, event: dict):
        """全購読者にイベントを配信し、バッファに蓄積する。

        終了イベント (done / error) で状態遷移し、購読者リストをクリアする。
        """
        with self._lock:
            if self.state != "running":
                return
            deep = bool(self.fingerprint and len(self.fingerprint) >= 5 and self.fingerprint[4] == "deep")
            if deep:
                if event.get("type") in {"deep_progress", "status", "budget"}:
                    self._event_buffer[:] = [e for e in self._event_buffer if e.get("type") != event.get("type")]
                proposed = self._event_buffer + [event]
                if len(proposed) > 32 or len(json.dumps(proposed, ensure_ascii=False).encode("utf-8")) > 1_000_000:
                    self.cancel_token.cancel()
                    event = {"type": "error", "code": "result_too_large", "message": "結果の保持上限に達しました"}
                    # Reserve space for the terminal error even if the previous event filled the budget.
                    self._event_buffer.clear()
            self._event_buffer.append(event)
            for q in self.subscriber_queues:
                if deep and q.full():
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                    for retained in self._event_buffer:
                        q.put_nowait(retained)
                else:
                    q.put(event)
            event_type = event.get("type")
            if event_type == "done":
                self.state = "completed"
                self.completed_at = time.monotonic()
                self.subscriber_queues.clear()
            elif event_type == "error":
                code = event.get("code", "")
                self.state = "cancelled" if code == "cancelled" else "failed"
                self.completed_at = time.monotonic()
                self.subscriber_queues.clear()


class _BroadcastQueue:
    """queue.Queue.put() 互換で JobEntry にブロードキャストするアダプタ。

    run_search の event_queue 引数にそのまま渡せる。
    """

    def __init__(self, entry: JobEntry):
        self._entry = entry

    def put(self, event: dict):
        self._entry.broadcast(event)


class JobTable:
    """ジョブの重複実行防止・途中参加・再接続を管理するテーブル。

    - クライアント生成 request_id による同一ジョブ検知
    - subscriber_queues による複数購読者サポート
    - 完了結果のバッファリングによる再接続
    - 完了済みジョブの TTL 自動削除
    """

    JOB_TTL = 60          # 完了済みジョブの保持秒数
    CLEANUP_INTERVAL = 30  # クリーンアップ間隔（秒）

    def __init__(self, max_concurrent: int = 2):
        self._jobs: dict[str, JobEntry] = {}
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent
        self._running_count = 0
        self._cleanup_stop = threading.Event()
        self._shutdown_once = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True,
            name="job-table-cleanup")
        self._cleanup_thread.start()

    # --- 公開 API ---

    def submit(self, request_id: str,
               cancel_token: CancellationToken, *, fingerprint=None,
               mode: str | None = None) -> tuple[JobEntry, bool]:
        """ジョブを登録する。

        Returns:
            (entry, is_new) — 既存ジョブなら is_new=False
        Raises:
            RuntimeError: 最大同時ジョブ数超過（code: server_busy）
        """
        with self._lock:
            if request_id in self._jobs:
                entry = self._jobs[request_id]
                if fingerprint is not None and entry.fingerprint != fingerprint:
                    raise RequestConflictError("request_id is already bound to another search")
                return entry, False
            if mode == "deep" and any(
                entry.state == "running"
                and entry.fingerprint
                and len(entry.fingerprint) >= 5
                and entry.fingerprint[4] == "deep"
                for entry in self._jobs.values()
            ):
                raise RuntimeError("deep_busy")
            if self._running_count >= self._max_concurrent:
                raise RuntimeError("server_busy")
            entry = JobEntry(cancel_token, fingerprint=fingerprint)
            self._jobs[request_id] = entry
            self._running_count += 1
            return entry, True

    def finish(self, request_id: str):
        """ワーカー完了時に running カウントを減らす。"""
        with self._lock:
            entry = self._jobs.get(request_id)
            if entry is not None and not entry.released:
                entry.released = True
                self._running_count = max(0, self._running_count - 1)

    def get(self, request_id: str) -> JobEntry | None:
        with self._lock:
            return self._jobs.get(request_id)

    @property
    def running_count(self) -> int:
        with self._lock:
            return self._running_count

    def has_running_mode(self, mode: str) -> bool:
        """指定モードのジョブが実行中かを原子的に確認する。"""
        with self._lock:
            return any(
                entry.state == "running"
                and entry.fingerprint
                and len(entry.fingerprint) >= 5
                and entry.fingerprint[4] == mode
                for entry in self._jobs.values()
            )

    # --- TTL クリーンアップ ---

    def _cleanup_loop(self):
        while not self._cleanup_stop.wait(self.CLEANUP_INTERVAL):
            self._cleanup_expired()

    def _cleanup_expired(self):
        now = time.monotonic()
        with self._lock:
            expired = [
                rid for rid, entry in self._jobs.items()
                if entry.completed_at is not None
                and (now - entry.completed_at) > self.JOB_TTL
            ]
            for rid in expired:
                del self._jobs[rid]

    # --- シャットダウン ---

    def shutdown(self):
        """クリーンアップスレッドを停止し、実行中ジョブをキャンセルする。"""
        if self._shutdown_once.is_set():
            return
        self._shutdown_once.set()
        self._cleanup_stop.set()
        with self._lock:
            for entry in self._jobs.values():
                entry.cancel_token.cancel()


class RequestConflictError(RuntimeError):
    """同じ request_id を異なる検索条件へ再利用した。"""


# ---------------------------------------------------------------------------
# search.py import (副作用吸収)
# ---------------------------------------------------------------------------

_search_available = False
_search_error = ""

try:
    import sys as _sys
    # search.py は sys.path.insert + prompt_templates import をモジュールレベルで実行
    _sys.path.insert(0, str(SCRIPT_DIR))
    from search import (
        PromptBudgetError,
        detect_model,
        iter_stream_events,
        build_chat_payload,
        RETRIEVAL_TIMEOUT,
        GENERATION_STALL_TIMEOUT,
        run_retrieval_pipeline,
    )
    from deep_research import (
        DEEP_TIMEOUT_DEFAULT,
        run_deep_research,
        detect_deep_model,
    )

    _search_available = True
except Exception as e:
    _search_error = str(e)
    logger.warning("search.py import failed: %s", e)
    # web_server.py が無条件に import する定数のフォールバック
    # （search.py 側の既定値と揃える）。
    GENERATION_STALL_TIMEOUT = 60
    DEEP_TIMEOUT_DEFAULT = DEEP_SEARCH_TIMEOUT_DEFAULT

    class PromptBudgetError(RuntimeError):
        """search.py を読めない場合のフォールバック（例外節の名前解決用）。"""

# ---------------------------------------------------------------------------
# prompt_templates import (search.py が読み込んでいるが念のため)
# ---------------------------------------------------------------------------

try:
    from prompt_templates import SYSTEM_PROMPT, build_user_prompt
    _prompts_available = True
except Exception:
    _prompts_available = False
    SYSTEM_PROMPT = ""
    build_user_prompt = None


# ---------------------------------------------------------------------------
# エラーコード辞書
# ---------------------------------------------------------------------------

class PagePresenceMonitor:
    """開いているWebページが無くなったらサーバー停止を判断する（web.bat非表示起動用）。

    ページは定期的に heartbeat を送り、閉じる時は ``close`` を送る（sendBeacon）。
    - 再読み込みは close → 新ページの heartbeat になるため、``close_grace_seconds``
      だけ待ってから停止する。別タブが開いていれば停止しない。
    - close を送れずに消えたページ（ブラウザ異常終了、組み込みブラウザ等で pagehide が
      発火しない閉じ方）は、最後の通知が「表示中」なら ``visible_stale_seconds``、
      「非表示」なら ``stale_seconds`` で失効する。背景タブのtimerは最大1分間隔まで
      間引かれるため、非表示側は十分長くする（表示中のタブは間引かれない）。
    - PCのスリープ等でtick間隔が大きく空いた場合は、ページの再送を待つため
      最終確認時刻を現在へ寄せる（スリープ明けに即停止しない）。
    - 一度もページが接続しない場合（ブラウザ起動失敗等）は ``initial_seconds`` で停止する。
    - ``busy``（索引構築・検索実行中）の間は停止を保留する。
    """

    PAGE_ID_MAX_LENGTH = 64
    MAX_PAGES = 32

    def __init__(
        self,
        *,
        stale_seconds: float = 300.0,
        visible_stale_seconds: float = 45.0,
        close_grace_seconds: float = 15.0,
        initial_seconds: float = 600.0,
        sleep_gap_seconds: float = 30.0,
        clock=time.monotonic,
    ):
        self._stale_seconds = stale_seconds
        self._visible_stale_seconds = visible_stale_seconds
        self._close_grace_seconds = close_grace_seconds
        self._initial_seconds = initial_seconds
        self._sleep_gap_seconds = sleep_gap_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._pages: dict[str, tuple[float, bool]] = {}  # page_id -> (最終通知, 表示中)
        self._ever_seen = False
        self._started = clock()
        self._last_tick = self._started
        self._empty_since: float | None = None

    @classmethod
    def valid_page_id(cls, page_id: object) -> bool:
        return (
            isinstance(page_id, str)
            and 8 <= len(page_id) <= cls.PAGE_ID_MAX_LENGTH
            and all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in page_id)
        )

    def heartbeat(self, page_id: str, *, visible: bool = False) -> None:
        if not self.valid_page_id(page_id):
            raise ValueError("invalid page id")
        with self._lock:
            now = self._clock()
            if page_id not in self._pages and len(self._pages) >= self.MAX_PAGES:
                oldest = min(self._pages, key=lambda key: self._pages[key][0])
                del self._pages[oldest]
            self._pages[page_id] = (now, bool(visible))
            self._ever_seen = True
            self._empty_since = None

    def close(self, page_id: str) -> None:
        if not self.valid_page_id(page_id):
            raise ValueError("invalid page id")
        with self._lock:
            if self._pages.pop(page_id, None) is not None and not self._pages:
                self._empty_since = self._clock()

    def open_page_count(self) -> int:
        with self._lock:
            return len(self._pages)

    def should_shutdown(self, *, busy: bool = False) -> bool:
        with self._lock:
            now = self._clock()
            if now - self._last_tick > self._sleep_gap_seconds:
                # スリープ明け: 経過時間でページを失効させず、再送の機会を与える。
                for page_id, (_seen, visible) in self._pages.items():
                    self._pages[page_id] = (now, visible)
                if self._empty_since is not None:
                    self._empty_since = now
                if not self._ever_seen:
                    self._started = now
            self._last_tick = now
            expired = [
                page_id
                for page_id, (seen, visible) in self._pages.items()
                if now - seen
                > (self._visible_stale_seconds if visible else self._stale_seconds)
            ]
            for page_id in expired:
                del self._pages[page_id]
            if self._pages:
                self._empty_since = None
                return False
            if busy:
                return False
            if not self._ever_seen:
                return now - self._started >= self._initial_seconds
            if self._empty_since is None:
                self._empty_since = now
            return now - self._empty_since >= self._close_grace_seconds


ERROR_CODES = {
    "invalid_query": 400,
    "invalid_timeout": 400,
    "invalid_mode": 400,
    "invalid_evidence_id": 400,
    "unauthorized": 403,
    "request_conflict": 409,
    "source_changed": 409,
    "index_source_changed": 409,
    "index_source_unreadable": 409,
    "index_mode_conflict": 409,
    "index_not_configured": 409,
    "evidence_too_large": 413,
    "evidence_not_found": 404,
    "evidence_unavailable": 500,
    "job_not_found": 404,
    "forbidden_origin": 403,
    "server_busy": 503,
    "service_unavailable": None,  # SSE only
    "timeout": None,              # SSE only
    "cancelled": None,            # SSE only
    "deep_failed": None,          # SSE only
    "internal_error": 500,
}


# ---------------------------------------------------------------------------
# request_id 生成
# ---------------------------------------------------------------------------

def generate_request_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# 構造化ログ
# ---------------------------------------------------------------------------

def log_structured(request_id: str, **fields):
    """JSON Lines 形式で構造化ログを出力する。"""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request_id": request_id,
        **fields,
    }
    logger.info(json.dumps(record, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Health チェック
# ---------------------------------------------------------------------------

def check_health() -> dict:
    """Ollama 接続状態と機能別ステータスを返す。"""
    ollama_ok = False
    model_name = ""

    try:
        req = urllib.request.Request(
            f"{_get_ollama_host()}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            ollama_ok = True
            # モデル検出
            if _search_available:
                try:
                    model_name = detect_model()
                except Exception:
                    pass
    except Exception:
        pass

    search_ok = _search_available and ollama_ok
    retrieval_ok = _search_available

    if ollama_ok:
        status = "ok" if search_ok else ("degraded" if retrieval_ok else "unavailable")
    else:
        status = "degraded" if retrieval_ok else "unavailable"

    return {
        "status": status,
        "ollama": ollama_ok,
        "model": model_name,
        "features": {
            "search": search_ok,
            "retrieval": retrieval_ok,
        },
    }


# ---------------------------------------------------------------------------
# 検索サービス
# ---------------------------------------------------------------------------

def run_search(query: str, reasoning: str, event_queue: queue.Queue,
               cancel_token: CancellationToken, request_id: str, *,
               mode: str = "answer", started_at: str | None = None,
               session_id: str = "", evidence_registry: EvidenceRegistry | None = None):
    """検索処理をバックグラウンドスレッドで実行し、イベントを queue に put する。

    イベント形式: {"type": "status"|"chunk"|"done"|"error", ...}
    """
    try:
        if mode not in VALID_SEARCH_MODES:
            event_queue.put({"type": "error", "code": "invalid_mode",
                             "message": "検索モードが不正です"})
            return
        if not _search_available:
            event_queue.put({"type": "error", "code": "service_unavailable",
                             "message": f"検索機能が利用できません: {_search_error}"})
            return

        cancel_token.check()

        # 1. 回答モードだけが chat model を検出する。検索専用はモデル通信を
        # 行わず、retrieval 内の keyword / optional embedding だけで完了する。
        model = None
        if mode == "answer":
            event_queue.put({"type": "status", "text": "モデルを検出しています..."})
            model = detect_model()
            log_structured(request_id, phase="detect_model", model=model)
        elif mode == "deep":
            event_queue.put({"type": "status", "text": "深掘り調査用モデルを検出しています..."})
            model = detect_deep_model()
            log_structured(request_id, phase="detect_model", model=model, mode=mode)

        cancel_token.check()

        if mode == "deep":
            def emit_deep_progress(event: dict) -> None:
                cancel_token.touch()
                event_queue.put(event)

            deep_result = run_deep_research(
                query,
                model=model or "",
                source_root=SCRIPT_DIR.parent / "skill-source",
                timeout_seconds=int(cancel_token.timeout_seconds),
                emit_progress=emit_deep_progress,
                cancel_check=cancel_token.check,
                absolute_deadline=cancel_token.deadline,
            )
            cancel_token.touch()
            event_queue.put({
                "type": "result_meta",
                "requestId": request_id,
                "mode": "deep",
                "startedAt": started_at or _utc_now_iso(),
                "chatModel": model,
                "embeddingModel": None,
                "reasoning": "off",
                "deepStatus": deep_result.status,
                "stopReason": deep_result.stop_reason,
            })
            event_queue.put(build_deep_evidence_event(
                deep_result,
                evidence_registry=evidence_registry,
                session_id=session_id,
            ))
            if deep_result.answer:
                event_queue.put({"type": "chunk", "text": deep_result.answer})
            if deep_result.status == "cancelled":
                event_queue.put({
                    "type": "error",
                    "code": "cancelled",
                    "message": "深掘り調査を中止しました",
                })
            elif deep_result.status == "failed" and not deep_result.evidence:
                event_queue.put({
                    "type": "error",
                    "code": "deep_failed",
                    "message": "深掘り調査を確定できませんでした",
                })
            else:
                event_queue.put({
                    "type": "done",
                    "completedAt": _utc_now_iso(),
                    "status": deep_result.status,
                    "stopReason": deep_result.stop_reason,
                })
            log_structured(
                request_id,
                phase="complete",
                mode="deep",
                status=deep_result.status,
                stop_reason=deep_result.stop_reason,
                evidence_count=len(deep_result.evidence),
                units=deep_result.diagnostics.get("units", 0),
            )
            return

        # 2. 共通 retrieval pipeline
        # Retrieval has its own child budget.  It can never extend the
        # request-wide deadline, and answer generation still uses the parent
        # token's remaining time if retrieval returns.
        retrieval_token = CancellationToken(
            min(RETRIEVAL_TIMEOUT, cancel_token.remaining())
        )

        def check_retrieval_cancel() -> None:
            cancel_token.check()
            retrieval_token.check()

        def emit_status(text: str) -> None:
            cancel_token.touch()
            event_queue.put({"type": "status", "text": text})

        pipeline_kwargs = {
            "model": model or "",
            "reasoning": reasoning if mode == "answer" else "off",
            "emit_status": emit_status,
            "cancel_check": check_retrieval_cancel,
        }
        # 既存のテスト/fake backendが旧keyword-only signatureでも回答の
        # 後方互換を保つ。実装済みpipelineにはmodeを必ず渡す。
        parameters = inspect.signature(run_retrieval_pipeline).parameters
        if "mode" in parameters:
            pipeline_kwargs["mode"] = mode
        if "remaining_seconds" in parameters:
            # 案A（LLM根拠検証）の予算逆算用。request全体の残り秒数を渡す。
            pipeline_kwargs["remaining_seconds"] = cancel_token.remaining
        retrieval = run_retrieval_pipeline(query, **pipeline_kwargs)
        log_structured(
            request_id,
            phase="retrieval",
            match_count=len(retrieval.matches),
            evidence_status=retrieval.evidence_status,
            confidence=retrieval.confidence,
            verification_status=str(getattr(retrieval, "verification_status", "")),
            verification_failure_reason=str(
                getattr(retrieval, "verification_failure_reason", "")
            ),
        )
        cancel_token.touch()
        effective_reasoning = reasoning if reasoning in ("low", "medium", "high") else "off"
        event_queue.put({
            "type": "result_meta",
            "requestId": request_id,
            "mode": mode,
            "startedAt": started_at or _utc_now_iso(),
            "chatModel": model if mode == "answer" else None,
            "embeddingModel": getattr(retrieval, "embedding_model", None),
            "reasoning": "off" if mode == "search" else effective_reasoning,
        })
        event_queue.put(build_evidence_event(
            retrieval,
            evidence_registry=evidence_registry,
            session_id=session_id,
        ))

        cancel_token.check()

        if mode == "search":
            event_queue.put({"type": "done", "completedAt": _utc_now_iso()})
            log_structured(
                request_id,
                phase="complete",
                mode=mode,
                match_count=len(retrieval.matches),
            )
            return

        # 3. プロンプト構築
        if _prompts_available and build_user_prompt is not None:
            user_prompt = retrieval.user_prompt
            system_prompt = SYSTEM_PROMPT
        else:
            user_prompt = query
            system_prompt = "あなたは質問に回答するアシスタントです。"

        cancel_token.check()

        # 4. Ollama ストリーミング（生成フェーズ。以降は stall 上限で監視する）
        cancel_token.start_stall_monitoring()
        event_queue.put({"type": "status", "text": "回答を生成しています..."})
        url = f"{_get_ollama_host()}/api/chat"
        reasoning_val = reasoning if reasoning in ("low", "medium", "high") else None
        body = build_chat_payload(model, system_prompt, user_prompt, reasoning_val)
        payload = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        remaining = cancel_token.remaining()
        if remaining <= 0:
            raise CancelledError("処理がタイムアウトしました", code="timeout")

        thinking_chars = 0
        content_chars = 0
        done_reason = "stop"

        def stream_response(resp) -> None:
            nonlocal thinking_chars, content_chars, done_reason
            for kind, text in iter_stream_events(resp):
                cancel_token.check()
                cancel_token.touch()
                if kind == "thinking":
                    thinking_chars += len(text)
                    # gpt-oss:20b on Ollama 0.31.2 may still return a
                    # ``message.thinking`` field even when ``think: false``
                    # was requested.  Keep the redacted count for diagnostics,
                    # but honor the user-facing reasoning=off contract by not
                    # publishing internal reasoning through SSE/UI.
                    if reasoning in ("low", "medium", "high"):
                        event_queue.put({"type": "thinking", "text": text})
                elif kind == "done":
                    done_reason = text
                else:
                    content_chars += len(text)
                    event_queue.put({"type": "chunk", "text": text})

        def to_stall_error(exc: BaseException) -> CancelledError:
            """生成フェーズの socket timeout を stall へ写像する。

            写像しないと run_search の汎用 except へ落ち、stall なのに
            internal_error として返ってしまう。文言は
            CancellationToken.check 側の stall メッセージと揃える。
            """
            return CancelledError(
                f"モデルからの応答が{int(GENERATION_STALL_TIMEOUT)}秒途絶えました",
                code="stall",
            )

        try:
            try:
                with urllib.request.urlopen(
                    req, timeout=min(remaining, GENERATION_STALL_TIMEOUT)
                ) as resp:
                    stream_response(resp)
            except urllib.error.HTTPError as exc:
                # think キーが原因で拒否された可能性がある場合のみ、除去して1回だけ再送する。
                # 親 cancel_token の残時間内で行い、全体 deadline は延長しない。
                if "think" in body and exc.code >= 400:
                    log_structured(request_id, phase="think_fallback", http_code=exc.code)
                    body.pop("think", None)
                    retry_remaining = cancel_token.remaining()
                    if retry_remaining <= 0:
                        raise CancelledError("処理がタイムアウトしました", code="timeout") from exc
                    retry_req = urllib.request.Request(
                        url,
                        data=json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(
                        retry_req, timeout=min(retry_remaining, GENERATION_STALL_TIMEOUT)
                    ) as resp:
                        stream_response(resp)
                else:
                    raise
        except (TimeoutError, socket.timeout) as exc:
            raise to_stall_error(exc) from exc
        except urllib.error.URLError as exc:
            # 接続不能（Ollama 停止等）まで stall としない。socket timeout に限定する。
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise to_stall_error(exc) from exc
            raise

        # 本文 0 文字で終了した場合は、無言で空を返さず理由を done の前に 1 回だけ通知する。
        if content_chars == 0:
            event_queue.put({"type": "answer_empty", "reason": done_reason})

        event_queue.put({"type": "done", "completedAt": _utc_now_iso()})
        log_structured(
            request_id,
            phase="complete",
            thinking_chars=thinking_chars,
            content_chars=content_chars,
            done_reason=done_reason,
        )

    except CancelledError as exc:
        event_queue.put({"type": "error", "code": exc.code, "message": str(exc)})
        log_structured(request_id, phase="cancelled", code=exc.code)
    except PromptBudgetError as exc:
        # 根拠を強制挿入して本文を潰すより、構成エラーとして止めて理由を示す。
        event_queue.put({"type": "error", "code": "prompt_budget", "message": str(exc)})
        log_structured(request_id, phase="prompt_budget_error")
    except Exception as e:
        logger.exception("Search error [%s]", request_id)
        event_queue.put({"type": "error", "code": "internal_error",
                         "message": "内部エラーが発生しました"})
