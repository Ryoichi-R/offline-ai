# 依存方向: web.bat → web_server.py → web_services.py → search.py
# 逆方向の import は禁止（レイヤー違反）
"""
offline-ai Web UI サーバー。

Python 標準ライブラリのみ使用（Flask/FastAPI 不使用）。
127.0.0.1 のみバインドし、セッショントークンで認証する。

# --- セマフォ設計 ---
# Layer 1: 接続数制御 Semaphore(10) — process_request オーバーライドで適用
#   目的: ThreadingHTTPServer のスレッド無制限増加を防止
#   対象: 全 HTTP 接続（GET /, /api/health 含む）
#   拒否時: ハンドラ未生成のまま _build_error_response() で 503 直接応答
#
# Layer 2: 重処理制御 Semaphore(2) — ハンドラ内で適用
#   目的: Ollama 呼び出しを伴う高負荷 API の同時実行を制限
#   対象: /api/search のみ
#   拒否時: ハンドラ経由で 503 応答（code: server_busy）
#
# 相互作用: Layer 1 を通過した接続のみが Layer 2 に到達する。
# Layer 1 が飽和しても /api/health 等は Layer 2 の対象外なので
# 理論上は到達可能だが、Layer 1 自体が拒否するため到達しない。
# この動作は意図的（DoS 時に軽量エンドポイントも保護するため）。
"""

import argparse
import base64
from datetime import datetime, timezone
import functools
import hashlib
import http.cookies
import inspect
import json
import logging
import math
import os
import queue
import re
import secrets
import signal
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import io

try:
    from config import (
        DEEP_SEARCH_TIMEOUT_DEFAULT,
        DEEP_SEARCH_TIMEOUT_MAX,
        DEEP_SEARCH_TIMEOUT_MIN,
        SEARCH_TIMEOUT_DEFAULT,
        SEARCH_TIMEOUT_MAX,
        SEARCH_TIMEOUT_MIN,
        validate_mode_timeout_seconds,
        validate_search_timeout_seconds,
    )
    from web_services import (
        CancellationToken,
        EvidenceViewError,
        ERROR_CODES,
        GENERATION_STALL_TIMEOUT,
        JobTable,
        JobNotFoundError,
        PagePresenceMonitor,
        RequestConflictError,
        _BroadcastQueue,
        check_health,
        create_evidence_registry,
        generate_request_id,
        log_structured,
        run_search,
        VALID_SEARCH_MODES,
        view_evidence,
    )
    from index_service import (
        IndexCoordinator,
        IndexGenerationChangedError,
        IndexModeConflictError,
    )
    from search import EmbedBuildError, SourceSnapshotError, MODEL_CONFIG, get_model_identity, detect_model, SKILL_SOURCE_DIR
    from model_config import (
        MAC_MODEL_DIGESTS,
        ModelConfigurationError,
        model_call_timeout_seconds,
        resolve_model_selection,
    )
except ValueError as exc:
    if __name__ == "__main__":
        print(f"[設定エラー] {exc}", file=sys.stderr)
        print("OLLAMA_HOST は localhost / 127.0.0.1 / ::1 のみ指定できます。", file=sys.stderr)
        raise SystemExit(2) from None
    raise

import assistant
from assistant_sessions import (
    ID as CONVERSATION_ID, MAX_SAVED_BYTES, REQUEST_ID, ConversationError,
    Conversations, failure_result, import_saved, render_saved,
)
from assistant_tools import SOURCE_ID, verify_source
from web_services import _index_marker

from source_folder import (
    SourceFolderError, describe_source_folder, pick_source_folder, save_source_root,
)
from guided_follow_up import (
    ACTIONS as GUIDED_ACTIONS,
    ACTION_FOLLOW_UP,
    ACTION_HINT,
    ConfirmationError,
    ConfirmationStore,
    guided_enabled,
    normalize_follow_up_text,
)

logger = logging.getLogger("offlineai.server")

SCRIPT_DIR = Path(__file__).resolve().parent
WEB_DIR = SCRIPT_DIR / "web"
INDEX_HTML = WEB_DIR / "index.html"
BOOTSTRAP_HTML = b"""<!doctype html>
<meta charset="utf-8">
<meta name="referrer" content="no-referrer">
<title>offline-ai sign in</title>
<p id="status">Starting offline-ai...</p>
<script>
(async () => {
  const token = new URLSearchParams(location.hash.slice(1)).get("token");
  history.replaceState(null, "", "/bootstrap");
  if (!token) {
    document.getElementById("status").textContent = "Authentication token is missing.";
    return;
  }
  const response = await fetch("/bootstrap", {
    method: "POST",
    credentials: "same-origin",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({token})
  });
  if (response.ok) {
    location.replace("/");
    return;
  }
  document.getElementById("status").textContent = "Authentication failed. Restart web.bat to issue a new one-time token.";
})().catch(() => {
  document.getElementById("status").textContent = "Authentication failed. Restart web.bat to issue a new one-time token.";
});
</script>
"""


@functools.lru_cache(maxsize=1)
def _index_content_security_policy() -> str:
    """index.html のinline assetをexact-byte SHA-256で許可する。"""
    content = INDEX_HTML.read_bytes()
    script_blocks = re.findall(rb"<script(?:\s[^>]*)?>(.*?)</script>", content, re.IGNORECASE | re.DOTALL)
    style_blocks = re.findall(rb"<style(?:\s[^>]*)?>(.*?)</style>", content, re.IGNORECASE | re.DOTALL)

    def directives(blocks: list[bytes]) -> str:
        if not blocks:
            return "'none'"
        hashes = []
        for block in blocks:
            normalized = block.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            digest = base64.b64encode(hashlib.sha256(normalized).digest()).decode("ascii")
            hashes.append(f"'sha256-{digest}'")
        return " ".join(hashes)

    return (
        f"default-src 'none'; script-src {directives(script_blocks)}; "
        f"style-src {directives(style_blocks)}; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    )


def _bootstrap_content_security_policy() -> str:
    script = re.search(rb"<script>(.*?)</script>", BOOTSTRAP_HTML, re.IGNORECASE | re.DOTALL)
    script_source = "'none'"
    if script:
        normalized = script.group(1).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = base64.b64encode(hashlib.sha256(normalized).digest()).decode("ascii")
        script_source = f"'sha256-{digest}'"
    return (
        f"default-src 'none'; script-src {script_source}; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )

# デフォルトタイムアウト（秒）。300秒はハング防止のfail-safeであり、通常の完了
# 時間ではない（受入は実測短縮によるretrieval/生成時間そのもので判定する）。
# 生成フェーズの無通信（stall）上限は GENERATION_STALL_TIMEOUT（既定60秒）で
# 別途監視するため、通常はこの上限に到達する前にstallまたは完了で終わる。
# 有効範囲は config.SEARCH_TIMEOUT_MIN〜SEARCH_TIMEOUT_MAX（300〜600秒）で、
# Web UIの検索単位指定・`--search-timeout`・`OFFLINEAI_SEARCH_TIMEOUT` が共有する。
DEFAULT_SEARCH_TIMEOUT = SEARCH_TIMEOUT_DEFAULT

# サイズ上限
MAX_REQUEST_LINE = 8192  # 8KB
MAX_HEADER_TOTAL = 65536  # 64KB
MAX_HEADER_LINE = 8192  # 8KB
MAX_QUERY_LENGTH = 1000


# ---------------------------------------------------------------------------
# ストリーム結合ヘルパー
# ---------------------------------------------------------------------------

class _ChainedReader:
    """2 つのストリームを直列に結合する読取専用ラッパー。

    検証済みヘッダ（BytesIO）と元の rfile（ボディ）を結合し、
    parse_request() がヘッダを読み、do_POST 等がボディを読む際に
    シームレスに動作する。ボディを事前にメモリに載せない。
    """

    def __init__(self, first: io.BytesIO, second):
        self._first = first
        self._second = second
        self._first_exhausted = False

    def read(self, size=-1):
        if not self._first_exhausted:
            data = self._first.read(size)
            if data:
                if size < 0:
                    return data + self._second.read()
                remaining = size - len(data)
                if remaining > 0:
                    data += self._second.read(remaining)
                    self._first_exhausted = True
                return data
            self._first_exhausted = True
        return self._second.read(size)

    def readline(self, size=-1):
        if not self._first_exhausted:
            line = self._first.readline(size)
            if line:
                return line
            self._first_exhausted = True
        return self._second.readline(size)

    def readlines(self, hint=-1):
        lines = []
        if not self._first_exhausted:
            lines = self._first.readlines(hint)
            self._first_exhausted = True
        lines.extend(self._second.readlines(hint))
        return lines

    def __iter__(self):
        if not self._first_exhausted:
            yield from self._first
            self._first_exhausted = True
        yield from self._second

    def close(self):
        """両方のストリームを閉じる。"""
        if hasattr(self._first, 'close'):
            self._first.close()
        if hasattr(self._second, 'close'):
            self._second.close()


# ---------------------------------------------------------------------------
# エラー応答ヘルパー
# ---------------------------------------------------------------------------

def _build_error_response(code: str, message: str) -> bytes:
    """HTTP エラー応答のバイト列を構築する（低レベル直接書込み用にも使用）。"""
    body = json.dumps({"error": {"code": code, "message": message}},
                      ensure_ascii=False).encode("utf-8")
    http_code = ERROR_CODES.get(code, 500)
    status_text = {
        400: "Bad Request", 403: "Forbidden", 413: "Payload Too Large",
        414: "URI Too Long", 415: "Unsupported Media Type",
        431: "Request Header Fields Too Large",
        500: "Internal Server Error", 503: "Service Unavailable",
    }.get(http_code, "Error")

    header = (
        f"HTTP/1.1 {http_code} {status_text}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    return header.encode("ascii") + body


# ---------------------------------------------------------------------------
# 検索単位timeout（Phase 7: 利用者timeout設定・全体残り時間表示）
# ---------------------------------------------------------------------------

def _parse_search_timeout_param(values: list[str], default: int, mode: str = "answer") -> int:
    """`timeout_seconds` クエリパラメータを解釈する。

    省略時はサーバー既定 ``default`` を使う（後方互換）。重複指定・契約外の
    値は ``ValueError`` を送出し、呼び出し側で ``invalid_timeout`` へ変換する。
    """
    if not values:
        return default
    if len(values) > 1:
        raise ValueError("timeout_seconds must not be repeated")
    return validate_mode_timeout_seconds(values[0], mode)


def _build_budget_event(cancel_token: CancellationToken) -> dict:
    """SSE購読開始時に送る全体timeoutの確定値と残り秒数。

    ``remainingSeconds`` は stall deadline ではなく全体 deadline 基準。
    再購読時にもその時点の残り時間を再計算して返せるよう、都度算出する。
    """
    total_seconds = int(cancel_token.timeout_seconds)
    return {
        "type": "budget",
        "timeoutSeconds": total_seconds,
        # deadline 生成直後は浮動小数の丸めで remaining が総秒数をわずかに
        # 超えることがある。残り時間が総時間を超えて見えないようクリップする。
        "remainingSeconds": min(total_seconds, math.ceil(cancel_token.remaining())),
    }


# ---------------------------------------------------------------------------
# サーバークラス（Layer 1: 接続数制御）
# ---------------------------------------------------------------------------

class LimitedThreadingServer(ThreadingHTTPServer):
    """接続数制御付き ThreadingHTTPServer。"""

    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass,
                 max_connections=10, session_token="",
                 search_timeout=DEFAULT_SEARCH_TIMEOUT,
                 deep_search_timeout=DEEP_SEARCH_TIMEOUT_DEFAULT,
                 bind_port=8080,
                 evidence_registry=None,
                 page_presence=None,
                 chat_model=None, model_selection_source=None, model_digest=None,
                 generation_stall_timeout=GENERATION_STALL_TIMEOUT,
                 source_settings_root=None, source_root=None):
        self.source_settings_root = Path(source_settings_root or SCRIPT_DIR.parent)
        self.source_root = Path(source_root or SKILL_SOURCE_DIR)
        self._source_picker_lock = threading.Lock()
        self._conn_sem = threading.Semaphore(max_connections)
        self.job_table = JobTable(max_concurrent=2)
        self.session_token = session_token
        self.bootstrap_token_used = False
        self._token_lock = threading.Lock()
        self.search_timeout = search_timeout
        self.deep_search_timeout = deep_search_timeout
        self.bind_port = bind_port
        self.active_tokens: list[CancellationToken] = []
        self._tokens_lock = threading.Lock()
        self._shutdown_once = threading.Event()
        self.index_coordinator = IndexCoordinator()
        self.evidence_registry = evidence_registry or create_evidence_registry()
        # 確認案内（既定は無効）の確認情報。サーバーのメモリ内だけに置く。
        self.confirmation_store = ConfirmationStore(evidence_registry=self.evidence_registry)
        self.conversations = Conversations()
        self.assistant_source_epoch = secrets.token_urlsafe(16)
        # web.bat（非表示起動）だけが設定する。None ならページを閉じても停止しない。
        self.page_presence = page_presence
        self.chat_model = chat_model
        self.model_selection_source = model_selection_source
        self.model_digest = model_digest
        self.generation_stall_timeout = generation_stall_timeout
        super().__init__(server_address, RequestHandlerClass)

    def process_request(self, request, client_address):
        if not self._conn_sem.acquire(blocking=False):
            # 接続上限超過: ハンドラを生成せずに直接 503 を返す
            try:
                request.sendall(_build_error_response(
                    "server_busy", "接続数上限に達しています"))
            except Exception:
                pass
            finally:
                try:
                    request.close()
                except Exception:
                    pass
            return

        try:
            super().process_request(request, client_address)
        except Exception:
            self._conn_sem.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._conn_sem.release()

    def register_token(self, token: CancellationToken):
        with self._tokens_lock:
            self.active_tokens.append(token)

    def unregister_token(self, token: CancellationToken):
        with self._tokens_lock:
            try:
                self.active_tokens.remove(token)
            except ValueError:
                pass

    def cancel_all(self):
        with self._tokens_lock:
            for t in self.active_tokens:
                t.cancel()

    def shutdown_jobs_once(self) -> int:
        if self._shutdown_once.is_set():
            return 0
        self._shutdown_once.set()
        self.job_table.shutdown()
        self.index_coordinator.shutdown()
        with self._tokens_lock:
            active_jobs = len(self.active_tokens)
        self.cancel_all()
        return active_jobs


# ---------------------------------------------------------------------------
# リクエストハンドラ
# ---------------------------------------------------------------------------

class OfflineAIHandler(BaseHTTPRequestHandler):
    """offline-ai Web UI HTTP ハンドラ。"""

    server_version = "OfflineAI/0.3.12"

    def log_message(self, format, *args):
        """デフォルトのログ出力を抑制（構造化ログに統一）。"""
        pass

    # --- リクエスト行・ヘッダサイズ制限 ---

    def handle_one_request(self):
        """リクエスト行とヘッダのサイズをバイト単位で検証してからパースする。

        仕様: parse_request() に委譲する前にサイズ検証を完了させ、
        過大なリクエスト行・ヘッダが parse_request 内部で処理されることを防止する。
        """
        try:
            # slowloris 対策: 読取フェーズ前にソケットタイムアウトを設定
            self.connection.settimeout(30)

            # --- リクエスト行のサイズ検証 ---
            self.raw_requestline = self.rfile.readline(MAX_REQUEST_LINE + 1)
            if len(self.raw_requestline) > MAX_REQUEST_LINE:
                self.requestline = ""
                self.request_version = "HTTP/1.1"
                self.command = ""
                self.send_error(414, "URI Too Long")
                return

            if not self.raw_requestline:
                self.close_connection = True
                return

            # --- ヘッダのバイト単位サイズ検証 (parse_request 前に実施) ---
            # rfile からヘッダ行をループ読取し、累積バイト数を計測する。
            # 検証通過後、読み取ったヘッダ行を rfile に戻して parse_request に委譲する。
            header_lines = []
            header_total_bytes = 0
            while True:
                line = self.rfile.readline(MAX_HEADER_LINE + 1)
                if len(line) > MAX_HEADER_LINE:
                    self.requestline = ""
                    self.request_version = "HTTP/1.1"
                    self.command = ""
                    self.send_error(431, "Request Header Fields Too Large")
                    return
                header_total_bytes += len(line)
                if header_total_bytes > MAX_HEADER_TOTAL:
                    self.requestline = ""
                    self.request_version = "HTTP/1.1"
                    self.command = ""
                    self.send_error(431, "Request Header Fields Too Large")
                    return
                header_lines.append(line)
                # 空行 (\r\n or \n) はヘッダ終端
                if line in (b"\r\n", b"\n", b""):
                    break

            # 検証済みヘッダ + 元の rfile（ボディ）を結合したストリームに差替え
            # ボディは元の rfile から遅延読取するため、メモリに全量載せない
            header_blob = b"".join(header_lines)
            self.rfile = _ChainedReader(io.BytesIO(header_blob), self.rfile)

            # --- parse_request に委譲 ---
            if not self.parse_request():
                return

            mname = "do_" + self.command
            if not hasattr(self, mname):
                self.send_error(501, f"Unsupported method ({self.command!r})")
                return
            method = getattr(self, mname)
            method()
            self.wfile.flush()
        except socket.timeout:
            self.close_connection = True
        except Exception:
            self.close_connection = True

    # --- セキュリティ: Origin / Host 検証 ---

    def _check_origin(self) -> bool:
        """Origin/Host ヘッダを検証する。"""
        allowed_hosts = {
            f"127.0.0.1:{self.server.bind_port}",
            f"localhost:{self.server.bind_port}",
            f"[::1]:{self.server.bind_port}",
        }

        host = self.headers.get("Host")
        if not host or host not in allowed_hosts:
            # HTTP/1.1 では Host ヘッダ必須。待受ポートの省略も認めない。
            return False

        origin = self.headers.get("Origin")
        if origin:
            parsed = urlparse(origin)
            if (
                parsed.scheme != "http"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in ("", "/")
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                return False
            host_with_port = parsed.netloc
            if host_with_port not in allowed_hosts or host_with_port != host:
                return False

        return True

    # --- セキュリティ: トークン認証 ---

    @staticmethod
    def _safe_token_eq(candidate, expected: str) -> bool:
        """セッショントークンの定数時間比較。

        secrets.compare_digest は str 比較時に非 ASCII を渡されると TypeError を投げるため、
        ここで型・ASCII を事前検証して安全側に倒す（不正候補は False）。
        生成トークンは secrets.token_urlsafe(32) の URL-safe Base64 で必ず ASCII。
        """
        if not isinstance(candidate, str) or not candidate.isascii():
            return False
        return secrets.compare_digest(candidate, expected)

    def _check_auth(self) -> bool:
        """Cookie 内のセッショントークンを検証する。

        タイミング攻撃耐性のため secrets.compare_digest で定数時間比較する。
        """
        return bool(self._get_auth_session_id())

    def _get_auth_session_id(self) -> str:
        """認証済み Cookie の値を内部の session key として返す。"""
        cookie_header = self.headers.get("Cookie", "")
        cookies = http.cookies.SimpleCookie()
        try:
            cookies.load(cookie_header)
        except http.cookies.CookieError:
            return ""

        token = cookies.get("offlineai_session")
        if not token:
            return ""
        if not self._safe_token_eq(token.value, self.server.session_token):
            return ""
        return token.value

    # --- レスポンスヘルパー ---

    def _send_security_headers(self, *, html: bool = False):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        if html:
            self.send_header("Content-Security-Policy", _index_content_security_policy())

    def _send_json(self, data: dict, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code: str, message: str):
        http_code = ERROR_CODES.get(code, 500)
        body = json.dumps({"error": {"code": code, "message": message}},
                          ensure_ascii=False).encode("utf-8")
        self.send_response(http_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self, max_bytes: int = 8192) -> dict:
        """Read a small control payload; index APIs never accept source text."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length < 0 or length > max_bytes:
            raise ValueError("control payload is too large")
        if length == 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("control payload must be an object")
        return payload

    def _begin_sse(self, *, close_on_terminal: bool = False):
        """SSEストリームを開始する。

        検索SSEはterminalイベント後にHTTP応答を閉じるため、ブラウザの
        ``ReadableStream`` が ``done`` へ進み、UI復帰処理を遅延させない。
        index購読は長時間購読を維持し、terminal時だけ同じclose契約へ進む。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close" if close_on_terminal else "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._send_security_headers()
        self.end_headers()
        # 自動再接続を抑制
        self.wfile.write(b"retry: 0\n\n")
        self.wfile.flush()

    def _send_sse_event(self, data: dict):
        """SSE イベントを 1 件送信する。"""
        payload = json.dumps(data, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_events(self, event_queue: queue.Queue,
                       cancel_token: CancellationToken,
                       cancel_on_disconnect: bool = True, terminal_grace: int = 0):
        """queue からイベントを読み出して SSE 送信する。

        cancel_on_disconnect=False の場合、クライアント切断時にジョブを
        キャンセルしない（途中参加・再接続をサポートするため）。
        """
        try:
            self._send_sse_event(_build_budget_event(cancel_token))
            while True:
                try:
                    event = event_queue.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() >= cancel_token.deadline + terminal_grace:
                        self._send_sse_event({
                            "type": "error", "code": "timeout",
                            "message": "処理がタイムアウトしました"
                        })
                        break
                    if cancel_token.stall_remaining() <= 0:
                        # 文言は run_search / CancellationToken.check 経路と揃える。
                        stall_seconds = int(cancel_token.stall_timeout or 0)
                        self._send_sse_event({
                            "type": "error", "code": "stall",
                            "message": f"モデルからの応答が{stall_seconds}秒途絶えました"
                        })
                        break
                    continue

                self._send_sse_event(event)

                if event.get("type") in ("done", "error"):
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            if cancel_on_disconnect:
                cancel_token.cancel()
        finally:
            # _begin_sse(close_on_terminal=True) と組み合わせ、terminal
            # eventのflush後にこのhandlerのHTTP応答を確実に終了する。
            # cancel_on_disconnect=Falseでも、ここへ到達するのはterminal
            # eventまたはクライアント切断だけであり、job自体の再接続契約は
            # JobTable側のbufferが保持する。
            self.close_connection = True

    def _stream_index_events(self, event_queue: queue.Queue, job_id: str):
        """Stream bounded index progress; disconnect never cancels the job."""
        try:
            self._begin_sse(close_on_terminal=True)
            while True:
                try:
                    event = event_queue.get(timeout=1)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                if event.get("job_id") not in {None, "", job_id}:
                    continue
                self._send_sse_event(event)
                if event.get("state") in {"ready", "cancelled", "failed"}:
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
        finally:
            self.close_connection = True

    # --- GET / ---

    def _handle_root(self):
        """Cookie 認証済みクライアントへ index.html を配信する。"""
        # 認証チェック（Cookie）
        if not self._check_auth():
            self._send_error_json("unauthorized", "認証が必要です")
            return

        # index.html 配信
        if not INDEX_HTML.exists():
            self.send_error(404, "index.html not found")
            return

        html = INDEX_HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers(html=True)
        self.end_headers()
        self.wfile.write(html)

    def _handle_bootstrap_get(self):
        """URL fragment を POST 交換するための固定中間ページを返す。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(BOOTSTRAP_HTML)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            _bootstrap_content_security_policy(),
        )
        self.end_headers()
        self.wfile.write(BOOTSTRAP_HTML)

    def _handle_bootstrap_post(self):
        """fragment から得た使い捨てtokenをHttpOnly Cookieへ交換する。"""
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._send_error_json("unauthorized", "認証形式が不正です")
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_error_json("unauthorized", "認証形式が不正です")
            return
        if content_length < 2 or content_length > 4096:
            self._send_error_json("unauthorized", "認証形式が不正です")
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json("unauthorized", "認証形式が不正です")
            return
        token_value = payload.get("token") if isinstance(payload, dict) else None
        with self.server._token_lock:
            if self.server.bootstrap_token_used:
                self._send_error_json("unauthorized", "トークンは既に使用済みです")
                return
            if not self._safe_token_eq(token_value, self.server.session_token):
                self._send_error_json("unauthorized", "トークンが不正です")
                return
            self.server.bootstrap_token_used = True

        self.send_response(204)
        cookie = (f"offlineai_session={token_value}; "
                  f"HttpOnly; SameSite=Strict; Path=/")
        self.send_header("Set-Cookie", cookie)
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
        self.end_headers()

    # --- ルーティング ---

    def _enforce_origin(self) -> bool:
        """Origin/Host を検証し、失敗時は構造化ログを出力して 403 応答する。"""
        if self._check_origin():
            return True
        # query値には検索語等が含まれ得るため、ログにはpure pathだけを記録する。
        try:
            safe_path = urlparse(self.path).path
        except Exception:
            safe_path = ""
        log_structured(generate_request_id(), phase="origin_rejected",
                       origin=self.headers.get("Origin", ""),
                       host=self.headers.get("Host", ""),
                       remote_addr=self.client_address[0],
                       path=safe_path,
                       method=self.command)
        self._send_error_json("forbidden_origin", "Origin/Host が不正です")
        return False

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if not self._enforce_origin():
            return

        if path == "/":
            self._handle_root()
        elif path == "/bootstrap":
            self._handle_bootstrap_get()
        elif path == "/api/health":
            self._handle_health()
        elif path == "/api/source-folder":
            self._handle_source_folder()
        elif path == "/api/index/status":
            self._handle_index_status()
        elif path == "/api/index/plan":
            self._handle_index_plan(parsed)
        elif path == "/api/index/events":
            self._handle_index_events(parsed)
        elif path == "/api/search":
            self._handle_search()
        elif path == "/api/evidence/view":
            self._handle_evidence_view(parsed)
        elif path.startswith("/api/conversations/"):
            self._handle_conversation_read(path)
        elif path == "/api/confirmation":
            self._handle_confirmation_view(parsed)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if not self._enforce_origin():
            return

        if path == "/bootstrap":
            self._handle_bootstrap_post()
        elif path == "/api/source-folder":
            self._handle_source_folder(save=True)
        elif path == "/api/source-folder/pick":
            self._handle_source_folder(pick=True)
        elif path == "/api/index/start":
            self._handle_index_start()
        elif path == "/api/index/resume":
            self._handle_index_start(resume=True)
        elif path == "/api/index/cancel":
            self._handle_index_cancel()
        elif path == "/api/search/cancel":
            self._handle_search_cancel()
        elif path == "/api/search":
            self._handle_search_continue()
        elif path in {"/api/conversations", "/api/conversations/import"}:
            self._handle_conversation_write(path)
        elif path == "/api/page/heartbeat":
            self._handle_page_presence(closing=False)
        elif path == "/api/page/close":
            self._handle_page_presence(closing=True)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        """CORS プリフライト — 許可しない（ヘッダを付与しない）。"""
        self.send_response(204)
        self.end_headers()

    # --- /api/source-folder ---

    def _handle_source_folder(self, *, save=False, pick=False):
        if not self._check_auth():
            self._send_error_json("unauthorized", "認証が必要です")
            return
        try:
            if pick:
                if self._read_json_body(max_bytes=128):
                    raise SourceFolderError("フォルダー選択の要求が不正です。")
                if not self.server._source_picker_lock.acquire(blocking=False):
                    self._send_error_json("server_busy", "フォルダー選択を開いています。")
                    return
                try:
                    chosen = pick_source_folder()
                finally:
                    self.server._source_picker_lock.release()
                self._send_json({"path": chosen, "cancelled": chosen is None})
                return
            if save:
                payload = self._read_json_body(max_bytes=16384)
                if set(payload) != {"path"}:
                    raise SourceFolderError("フォルダーのパスを指定してください。")
                saved_root = save_source_root(self.server.source_settings_root, payload["path"])
                if saved_root != self.server.source_root.resolve():
                    self.server.assistant_source_epoch = secrets.token_urlsafe(16)
            self._send_json(describe_source_folder(
                self.server.source_settings_root, self.server.source_root
            ))
        except (ValueError, UnicodeError) as exc:
            self._send_error_json("invalid_query", str(exc))
        except OSError:
            self._send_error_json("source_settings_unavailable", "参考資料フォルダーの設定を保存できません。アプリの書き込み権限を確認してください。")

    # --- /api/health ---

    def _handle_health(self):
        # health は認証不要（ただし Origin 検証済み）
        # Cookie がなくても応答する
        result = check_health(getattr(self.server, "chat_model", None))
        result["index"] = self.server.index_coordinator.status()
        result["searchTimeoutDefault"] = self.server.search_timeout
        result["searchTimeoutMin"] = SEARCH_TIMEOUT_MIN
        result["searchTimeoutMax"] = SEARCH_TIMEOUT_MAX
        result["deepSearchTimeoutDefault"] = self.server.deep_search_timeout
        result["deepSearchTimeoutMin"] = DEEP_SEARCH_TIMEOUT_MIN
        result["deepSearchTimeoutMax"] = DEEP_SEARCH_TIMEOUT_MAX
        result["pageCloseStop"] = self.server.page_presence is not None
        self._send_json(result)

    # --- /api/page/* ---

    def _handle_page_presence(self, *, closing: bool):
        """開いているページの在席通知。web.bat起動時はページが無くなると停止する。"""
        if not self._check_auth():
            self._send_error_json("unauthorized", "認証が必要です")
            return
        try:
            payload = self._read_json_body(max_bytes=512)
        except (ValueError, UnicodeDecodeError):
            self._send_error_json("invalid_query", "page payload is invalid")
            return
        page_id = payload.get("page_id")
        if not PagePresenceMonitor.valid_page_id(page_id):
            self._send_error_json("invalid_query", "page id is invalid")
            return
        visible = payload.get("visible", False)
        if not isinstance(visible, bool):
            self._send_error_json("invalid_query", "page visibility is invalid")
            return
        monitor = self.server.page_presence
        if monitor is None:
            self._send_json({"enabled": False})
            return
        if closing:
            monitor.close(page_id)
        else:
            monitor.heartbeat(page_id, visible=visible)
        self._send_json({"enabled": True, "openPages": monitor.open_page_count()})

    # --- /api/index/* ---

    def _handle_index_status(self):
        if not self._check_auth():
            self._send_error_json("unauthorized", "認証が必要です")
            return
        self._send_json(self.server.index_coordinator.status())

    def _handle_index_plan(self, parsed):
        if not self._check_auth():
            self._send_error_json("unauthorized", "認証が必要です")
            return
        values = parse_qs(parsed.query, keep_blank_values=True).get("mode", [])
        if not values:
            mode = "incremental"
        elif len(values) != 1:
            self._send_error_json("invalid_mode", "インデックス更新モードが不正です")
            return
        else:
            mode = values[0]
        try:
            self._send_json(self.server.index_coordinator.plan(mode=mode))
        except ValueError:
            self._send_error_json("invalid_mode", "インデックス更新モードが不正です")
        except SourceSnapshotError as exc:
            self._send_error_json("index_source_unreadable", exc.code)
        except EmbedBuildError as exc:
            self._send_error_json("index_not_configured", exc.code)
        except RuntimeError as exc:
            self._send_error_json("server_busy", str(exc))

    def _handle_index_events(self, parsed):
        if not self._check_auth():
            self._send_error_json("unauthorized", "認証が必要です")
            return
        job_id = parse_qs(parsed.query).get("job_id", [""])[0].strip()
        status = self.server.index_coordinator.status()
        job_id = job_id or str(status.get("job_id") or "")
        if not job_id:
            self._send_error_json("invalid_query", "index job is not available")
            return
        subscriber = self.server.index_coordinator.subscribe(job_id)
        try:
            self._stream_index_events(subscriber, job_id)
        finally:
            self.server.index_coordinator.unsubscribe(subscriber)

    def _handle_index_start(self, *, resume: bool = False):
        if not self._check_auth():
            self._send_error_json("unauthorized", "認証が必要です")
            return
        try:
            payload = self._read_json_body()
            mode = payload.get("mode")
            if mode is not None and not isinstance(mode, str):
                raise ValueError("invalid index mode")
            expected_generation = payload.get("expected_generation")
            if expected_generation is not None and not isinstance(expected_generation, str):
                raise ValueError("invalid expected generation")
            self._send_json(
                self.server.index_coordinator.start(
                    resume=resume,
                    mode=mode,
                    expected_generation=expected_generation,
                ),
                status=202,
            )
        except ValueError:
            self._send_error_json("invalid_mode", "インデックス更新要求が不正です")
        except IndexModeConflictError as exc:
            self._send_error_json("index_mode_conflict", str(exc))
        except IndexGenerationChangedError as exc:
            self._send_error_json("index_source_changed", str(exc))
        except SourceSnapshotError as exc:
            self._send_error_json("index_source_unreadable", exc.code)
        except RuntimeError as exc:
            self._send_error_json("server_busy", str(exc))

    def _handle_index_cancel(self):
        if not self._check_auth():
            self._send_error_json("unauthorized", "認証が必要です")
            return
        try:
            payload = self._read_json_body()
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json("invalid_query", "cancel payload is invalid")
            return
        job_id = payload.get("job_id")
        self._send_json(self.server.index_coordinator.cancel(str(job_id) if job_id else None))

    def _handle_search_cancel(self):
        """所有権を検証して、サーバー側の検索ジョブへ中止を伝播する。"""
        session_id = self._get_auth_session_id()
        if not session_id:
            self._send_error_json("unauthorized", "認証が必要です")
            return
        try:
            payload = self._read_json_body(max_bytes=1024)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json("invalid_query", "cancel payload is invalid")
            return
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            self._send_error_json("invalid_query", "request_id is required")
            return
        entry = self.server.job_table.get(request_id)
        if entry is None:
            self._send_error_json("job_not_found", "検索ジョブが見つかりません")
            return
        fingerprint = entry.fingerprint
        if not fingerprint or fingerprint[0] != session_id:
            self._send_error_json("unauthorized", "この検索ジョブを中止する権限がありません")
            return
        entry.cancel_token.cancel()
        self._send_json({
            "requestId": request_id,
            "cancelled": True,
            "state": entry.state,
        })

    def _handle_evidence_view(self, parsed):
        session_id = self._get_auth_session_id()
        if not session_id:
            self._send_error_json("unauthorized", "認証が必要です")
            return
        params = parse_qs(parsed.query, keep_blank_values=True)
        evidence_ids = params.get("evidence_id", [])
        if len(evidence_ids) != 1 or not evidence_ids[0].strip():
            self._send_error_json("invalid_evidence_id", "根拠IDが不正です")
            return
        try:
            result = view_evidence(
                self.server.evidence_registry,
                evidence_ids[0].strip(),
                session_id,
            )
        except EvidenceViewError as exc:
            self._send_error_json(exc.code, str(exc))
            return
        self._send_json(result)

    # --- /api/search ---

    def _handle_search(self):
        session_id = self._get_auth_session_id()
        if not session_id:
            self._send_error_json("unauthorized", "認証が必要です")
            return

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query, keep_blank_values=True)
        query = params.get("q", [""])[0].strip()
        reasoning = params.get("reasoning", [""])[0].strip()
        mode_values = params.get("mode", [])
        if not mode_values:
            mode = "answer"
        elif len(mode_values) != 1 or mode_values[0] not in VALID_SEARCH_MODES:
            self._send_error_json("invalid_mode", "検索モードが不正です")
            return
        else:
            mode = mode_values[0]
        resume_values = params.get("resume_only", [])
        if resume_values and resume_values != ["1"]:
            self._send_error_json("invalid_query", "再接続指定が不正です")
            return
        resume_only = bool(resume_values)
        if resume_only and (len(params.get("request_id", [])) != 1
                            or not params["request_id"][0].strip()):
            self._send_error_json("invalid_query", "再接続には検索IDが必要です")
            return
        request_id = (params.get("request_id", [""])[0].strip()
                      or self.headers.get("X-Request-ID", "").strip()
                      or generate_request_id())

        if not query:
            self._send_error_json("invalid_query", "検索クエリが空です")
            return
        if len(query) > MAX_QUERY_LENGTH:
            self._send_error_json("invalid_query",
                                  f"クエリは{MAX_QUERY_LENGTH}文字以内にしてください")
            return

        try:
            timeout_default = (
                self.server.deep_search_timeout
                if mode == "deep"
                else self.server.search_timeout
            )
            timeout_seconds = _parse_search_timeout_param(
                params.get("timeout_seconds", []), timeout_default, mode
            )
        except ValueError:
            timeout_label = (
                f"深掘りタイムアウトは{DEEP_SEARCH_TIMEOUT_MIN}〜{DEEP_SEARCH_TIMEOUT_MAX}秒"
                if mode == "deep"
                else f"検索タイムアウトは{SEARCH_TIMEOUT_MIN}〜{SEARCH_TIMEOUT_MAX}秒"
            )
            self._send_error_json(
                "invalid_timeout",
                f"{timeout_label}の整数で、"
                "重複指定はできません",
            )
            return

        # ジョブテーブルに登録（重複なら既存エントリを返す）
        cancel_token = CancellationToken(
            timeout_seconds,
            stall_timeout=getattr(self.server, "generation_stall_timeout", GENERATION_STALL_TIMEOUT),
            defer_stall=True,
        )
        # 確認案内が有効なら、深掘りの確認段階は深掘り枠を消費しない。
        guided = guided_enabled() and mode in {"answer", "deep"}
        try:
            fingerprint = (session_id, query, reasoning, timeout_seconds, mode)
            entry, is_new = self.server.job_table.submit(
                request_id, cancel_token, fingerprint=fingerprint, mode=mode,
                resume_only=resume_only, defer_deep_slot=guided and mode == "deep",
            )
        except JobNotFoundError:
            self._send_error_json(
                "job_not_found",
                "再接続先の調査が見つかりません。保持期限切れ、またはサーバーが再起動した可能性があります。自動で新しい調査は開始しません。",
            )
            return
        except RequestConflictError:
            self._send_error_json(
                "request_conflict",
                "request_id は別の検索条件で既に使用されています",
            )
            return
        except RuntimeError:
            self._send_error_json("server_busy",
                                  "サーバーが処理中です。しばらく待ってから再試行してください")
            return

        if is_new:
            guided_kwargs = {}
            if guided:
                guided_kwargs["confirmation_store"] = self.server.confirmation_store
                if mode == "deep":
                    guided_kwargs["acquire_deep_slot"] = (
                        lambda: self.server.job_table.acquire_deep_slot(request_id)
                    )
            self._spawn_search_worker(
                entry, request_id, query, reasoning, cancel_token, mode, session_id,
                timeout_seconds, guided_kwargs,
            )

        # 購読（新規でも既存でも同じ）
        sub_queue = entry.add_subscriber()
        try:
            self._begin_sse(close_on_terminal=True)
            self._stream_events(sub_queue, entry.cancel_token,
                                cancel_on_disconnect=False)
        finally:
            entry.remove_subscriber(sub_queue)

    def _spawn_search_worker(self, entry, request_id, query, reasoning, cancel_token,
                             mode, session_id, timeout_seconds, extra_kwargs):
        """検索ワーカーを起動する（GET の検索と確認案内の続きで共用）。"""
        self.server.register_token(cancel_token)
        bcast_queue = _BroadcastQueue(entry)

        log_structured(request_id, phase="search_start",
                       query_length=len(query),
                       timeout_seconds=timeout_seconds,
                       user_agent=self.headers.get("User-Agent", ""),
                       remote_addr=self.client_address[0])

        def _worker():
            try:
                kwargs = {
                    "mode": mode,
                    "started_at": datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    ).replace("+00:00", "Z"),
                    "session_id": session_id,
                    "evidence_registry": self.server.evidence_registry,
                }
                run_parameters = inspect.signature(run_search).parameters
                for key, value in extra_kwargs.items():
                    if key in run_parameters:
                        kwargs[key] = value
                if "model_name" in run_parameters:
                    kwargs["model_name"] = getattr(self.server, "chat_model", None)
                if "model_digest" in run_parameters:
                    kwargs["model_digest"] = getattr(self.server, "model_digest", None)
                if "model_selection_source" in run_parameters:
                    kwargs["model_selection_source"] = getattr(
                        self.server, "model_selection_source", None
                    )
                if "mode" in run_parameters:
                    run_search(
                        query, reasoning, bcast_queue, cancel_token, request_id,
                        **kwargs,
                    )
                else:
                    # 旧 fake/reader の worker seam を壊さない。
                    run_search(query, reasoning, bcast_queue, cancel_token, request_id)
            finally:
                self.server.job_table.finish(request_id)
                self.server.unregister_token(cancel_token)
                log_structured(request_id, phase="search_end")

        threading.Thread(target=_worker, daemon=True).start()

    def _assistant_generation(self):
        return assistant.source_generation(
            self.server.source_root, [self.server.assistant_source_epoch, _index_marker()]
        )

    def _handle_conversation_write(self, path):
        session = self._get_auth_session_id()
        if not session:
            self._send_error_json("unauthorized", "認証が必要です")
            return
        try:
            if path.endswith("/import"):
                payload = self._read_json_body(max_bytes=MAX_SAVED_BYTES * 2)
                if set(payload) != {"markdown"}:
                    raise ValueError("保存したMarkdownを指定してください。")
                result = import_saved(payload["markdown"], self.server.source_root,
                                      self.server.evidence_registry, session)
                self._send_json({"result": result})
                return
            if self._read_json_body(max_bytes=128):
                raise ValueError("会話の開始要求が不正です。")
            id_ = self.server.conversations.new(session, self._assistant_generation())
            self._send_json({"conversationId": id_, "nextTurn": 1}, status=201)
        except ConversationError as exc:
            self._send_error_json(exc.code, str(exc))
        except (ValueError, UnicodeError, RecursionError):
            self._send_error_json("invalid_query", "会話の開始または保存回答の読み込みに失敗しました。")

    def _handle_conversation_read(self, path):
        session = self._get_auth_session_id()
        if not session:
            self._send_error_json("unauthorized", "認証が必要です")
            return
        parts = path.split("/")[3:]
        id_ = parts[0]
        if not CONVERSATION_ID.fullmatch(id_):
            self._send_error_json("conversation_not_found", "会話が見つかりません。")
            return
        store = self.server.conversations
        args = id_, session, self._assistant_generation()
        try:
            if len(parts) == 1:
                self._send_json(store.view(*args))
            elif len(parts) == 3 and parts[1] == "sources" and SOURCE_ID.fullmatch(parts[2]):
                source = store.source(*args, parts[2])
                self._send_json(verify_source(self.server.source_root, source,
                                              registry=self.server.evidence_registry, session_id=session))
            elif (len(parts) == 4 and parts[1] == "turns" and parts[3] == "markdown"
                  and re.fullmatch(r"[1-9][0-9]{0,5}", parts[2])):
                self._send_json({"markdown": render_saved(store.result(*args, int(parts[2])))})
            else:
                self._send_error_json("conversation_not_found", "会話または回答が見つかりません。")
        except ConversationError as exc:
            self._send_error_json(exc.code, str(exc))

    def _handle_assistant_search(self, session, payload):
        id_ = payload.get("conversationId")
        request_id = payload.get("requestId")
        turn = payload.get("turn")
        resume = payload.get("resume_only", False)
        if (not isinstance(id_, str) or not CONVERSATION_ID.fullmatch(id_)
                or not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id)
                or type(turn) is not int or not 1 <= turn <= 1_000_000
                or type(resume) is not bool):
            self._send_error_json("invalid_query", "会話またはターンの指定が不正です。")
            return
        store = self.server.conversations
        jobs = self.server.job_table
        generation = self._assistant_generation()
        try:
            if resume:
                if set(payload) != {"mode", "conversationId", "requestId", "turn", "resume_only"}:
                    raise ValueError()
                store.authorize_resume(id_, session, generation, request_id, turn)
                entry = jobs.get(request_id)
                if entry is None or (entry.fingerprint or ())[:5] != (session, id_, turn, generation, "assistant"):
                    raise JobNotFoundError()
            else:
                if set(payload) - {"mode", "conversationId", "requestId", "turn", "resume_only",
                                   "text", "reasoning", "timeout_seconds"}:
                    raise ValueError()
                text, reasoning = payload.get("text"), payload.get("reasoning", "off")
                if (not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_QUERY_LENGTH
                        or reasoning not in ("off", "low", "medium", "high")):
                    raise ValueError()
                try:
                    seconds = assistant.validate_timeout(payload.get("timeout_seconds", assistant.DEFAULT_SECONDS))
                except ValueError as exc:
                    self._send_error_json("invalid_timeout", str(exc))
                    return
                query = text.strip()
                model = self.server.chat_model or detect_model()
                fingerprint = (session, id_, turn, generation, "assistant", model, query, reasoning, seconds)
                request_hash = hashlib.sha256(assistant.encode(fingerprint).encode()).hexdigest()
                token = CancellationToken(seconds)

                def submit(resume_only):
                    entry, created = jobs.submit(request_id, token, fingerprint=fingerprint, mode="assistant",
                                                 resume_only=resume_only)
                    if not resume_only and not created:
                        raise RequestConflictError()
                    return entry

                entry, snapshot = store.start(id_, session, generation, request_id, turn, request_hash, submit)
                if snapshot is not None:
                    self._spawn_assistant_worker(entry, id_, request_id, turn, generation, query,
                                                 model, reasoning, snapshot)
        except ConversationError as exc:
            self._send_error_json(exc.code, str(exc))
            return
        except JobNotFoundError:
            self._send_error_json("job_not_found", "処理の保持期限が過ぎました。会話を再取得してください。")
            return
        except RequestConflictError:
            self._send_error_json("request_conflict", "この処理IDは既に使用されています。")
            return
        except ValueError:
            self._send_error_json("invalid_query", "アシスタントの要求が不正です。")
            return
        except RuntimeError:
            self._send_error_json("server_busy", "サーバーが処理中です。")
            return
        subscriber = entry.add_subscriber()
        try:
            self._begin_sse(close_on_terminal=True)
            self._stream_events(subscriber, entry.cancel_token, cancel_on_disconnect=False, terminal_grace=5)
        finally:
            entry.remove_subscriber(subscriber)

    def _spawn_assistant_worker(self, entry, id_, request_id, turn, generation, query, model, reasoning, snapshot):
        token = entry.cancel_token
        self.server.register_token(token)

        def emit(kind, value):
            entry.broadcast({"type": kind, **value})

        def work():
            result = failure_result(turn, query)
            try:
                turns, prior, key = snapshot
                result = assistant.live_turn(
                    self.server.source_root, generation, turn, query, model, reasoning,
                    token.remaining(), turns, prior, key,
                    lambda: token.is_cancelled or generation != self._assistant_generation(), emit,
                )
            except Exception:
                # No question, output, excerpts, or exception text in logs.
                log_structured(request_id, phase="assistant_worker_failed")
            finally:
                try:
                    stored = self.server.conversations.finish(id_, request_id, result)
                finally:
                    self.server.job_table.finish(request_id)
                    self.server.unregister_token(token)
                if stored is not None:
                    emit("assistant_result", {"result": stored})
                emit("done", {"requestId": request_id})

        thread = threading.Thread(target=work, daemon=True)
        try:
            thread.start()
        except RuntimeError:
            self.server.conversations.finish(id_, request_id, failure_result(turn, query))
            emit("error", {"code": "internal_error", "message": "処理を起動できません。"})
            self.server.job_table.finish(request_id)
            self.server.unregister_token(token)

    # --- 確認案内の続き（POST /api/search）と確認パネルの復元 ---

    def _handle_confirmation_view(self, parsed):
        session_id = self._get_auth_session_id()
        if not session_id:
            self._send_error_json("unauthorized", "認証が必要です")
            return
        if not guided_enabled():
            self.send_error(404)
            return
        params = parse_qs(parsed.query, keep_blank_values=True)
        values = params.get("id", [])
        try:
            if len(values) != 1:
                raise ConfirmationError("confirmation_invalid", "確認IDが不正です")
            # 読み取り専用。期限を延長しない。
            public = self.server.confirmation_store.view(values[0], session_id)
        except ConfirmationError as exc:
            self._send_error_json(exc.code, str(exc))
            return
        self._send_json({"confirmation": public})

    def _handle_search_continue(self):
        session_id = self._get_auth_session_id()
        if not session_id:
            self._send_error_json("unauthorized", "認証が必要です")
            return
        try:
            payload = self._read_json_body(max_bytes=MAX_REQUEST_LINE)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            if not guided_enabled():
                self.send_error(404)
                return
            self._send_error_json("invalid_query", f"送信内容が不正です（{MAX_REQUEST_LINE}バイト以内）")
            return
        mode = payload.get("mode", "answer")
        if mode == "assistant":
            self._handle_assistant_search(session_id, payload)
            return
        if not guided_enabled():
            self.send_error(404)
            return
        if mode not in {"answer", "deep"}:
            self._send_error_json("invalid_mode", "検索モードが不正です")
            return
        action = payload.get("action")
        cid = payload.get("confirmationId")
        if action not in GUIDED_ACTIONS or not isinstance(cid, str):
            self._send_error_json("confirmation_invalid", "確認の操作が不正です")
            return
        resume_only = payload.get("resume_only") is True
        request_id = str(payload.get("requestId") or "").strip()
        if resume_only and not request_id:
            self._send_error_json("invalid_query", "再接続には検索IDが必要です")
            return
        request_id = request_id or generate_request_id()
        reasoning = str(payload.get("reasoning") or "").strip()
        choice = payload.get("choice")
        if choice is not None and (isinstance(choice, bool) or not isinstance(choice, int)):
            self._send_error_json("confirmation_invalid", "選択肢が不正です")
            return
        options_version = payload.get("optionsVersion")
        text = ""
        try:
            if action in {ACTION_HINT, ACTION_FOLLOW_UP}:
                text = normalize_follow_up_text(payload.get("text"))
        except ConfirmationError as exc:
            self._send_error_json(exc.code, str(exc))
            return
        drop_targets = payload.get("dropTargets") or []
        if (
            not isinstance(drop_targets, list)
            or len(drop_targets) > 20
            or any(not isinstance(t, str) or len(t) > 100 for t in drop_targets)
        ):
            self._send_error_json("confirmation_invalid", "外す対象の指定が不正です")
            return
        timeout_values = payload.get("timeout_seconds")
        try:
            timeout_default = (
                self.server.deep_search_timeout if mode == "deep" else self.server.search_timeout
            )
            timeout_seconds = _parse_search_timeout_param(
                [] if timeout_values is None else [str(timeout_values)], timeout_default, mode
            )
        except ValueError:
            self._send_error_json("invalid_timeout", "タイムアウトの指定が不正です")
            return

        identity = json.dumps(
            [cid, action, options_version, choice, text, sorted(drop_targets)], ensure_ascii=False
        )
        fingerprint = (session_id, "guided\x00" + identity, reasoning, timeout_seconds, mode)
        cancel_token = CancellationToken(
            timeout_seconds,
            stall_timeout=getattr(self.server, "generation_stall_timeout", GENERATION_STALL_TIMEOUT),
            defer_stall=True,
        )
        job_table = self.server.job_table
        existing = job_table.get(request_id)
        record = None
        if not resume_only and existing is None:
            # 新規ジョブだけが確認IDの有効性と現在の状態を検証する。
            try:
                record = self.server.confirmation_store.begin(
                    cid, session_id, action=action, options_version=options_version,
                    request_id=request_id,
                )
            except ConfirmationError as exc:
                self._send_error_json(exc.code, str(exc))
                return
        try:
            entry, is_new = job_table.submit(
                request_id, cancel_token, fingerprint=fingerprint, mode=mode,
                resume_only=resume_only or existing is not None, defer_deep_slot=True,
            )
        except JobNotFoundError:
            self._send_error_json(
                "job_not_found",
                "再接続先の調査が見つかりません。保持期限切れ、またはサーバーが再起動した可能性があります。自動で新しい調査は開始しません。",
            )
            return
        except RequestConflictError:
            self._send_error_json("request_conflict", "request_id は別の検索条件で既に使用されています")
            return
        except RuntimeError:
            if record is not None:
                self.server.confirmation_store.finish(cid, request_id, completed=False)
            self._send_error_json("server_busy", "サーバーが処理中です。しばらく待ってから再試行してください")
            return

        if is_new:
            guided_request = {
                "confirmationId": cid,
                "action": action,
                "choice": choice,
                "text": text,
                "dropTargets": list(drop_targets),
                "record": record,
            }
            extra = {
                "confirmation_store": self.server.confirmation_store,
                "guided_request": guided_request,
            }
            if mode == "deep":
                extra["acquire_deep_slot"] = lambda: job_table.acquire_deep_slot(request_id)
            self._spawn_search_worker(
                entry, request_id, record.initial_query, reasoning, cancel_token, mode,
                session_id, timeout_seconds, extra,
            )
        sub_queue = entry.add_subscriber()
        try:
            self._begin_sse(close_on_terminal=True)
            self._stream_events(sub_queue, entry.cancel_token, cancel_on_disconnect=False)
        finally:
            entry.remove_subscriber(sub_queue)

# ---------------------------------------------------------------------------
# サーバー起動
# ---------------------------------------------------------------------------

def find_available_port(start=8080, end=8089) -> int:
    """start-end の範囲で空きポートを探す。"""
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"ポート {start}-{end} が全て使用中です")


def mask_token(token: str) -> str:
    """トークンをマスク表示する（先頭4文字+末尾4文字のみ）。"""
    if len(token) <= 8:
        return "****"
    return token[:4] + "****" + token[-4:]


def run_page_presence_watchdog(server, monitor, stop_event, on_idle, *, interval=2.0) -> None:
    """開いているページが無くなったら ``on_idle`` を1回呼ぶ。

    索引構築中・検索実行中は停止を保留し、終わってから判定する。
    """
    while not stop_event.wait(interval):
        busy = server.index_coordinator.is_running() or server.job_table.running_count > 0
        if monitor.should_shutdown(busy=busy):
            on_idle()
            return


def main():
    parser = argparse.ArgumentParser(description="offline-ai Web UI Server")
    parser.add_argument("--model", type=str, default=None,
                        help="Chat model tag (CLI > OFFLINE_AI_CHAT_MODEL > _internal/.model)")
    parser.add_argument("--port", type=int, default=None,
                        help="バインドポート (default: 8080、競合時 8080-8089 自動探索)")
    parser.add_argument("--search-timeout", type=int, default=None,
                        help=f"検索タイムアウト秒数（{SEARCH_TIMEOUT_MIN}〜{SEARCH_TIMEOUT_MAX}の整数）")
    parser.add_argument("--show-token", action="store_true",
                        help="起動時にトークン全文を表示")
    parser.add_argument("--log-file", type=str, default=None,
                        help="ログファイルパス")
    parser.add_argument("--exit-when-page-closed", action="store_true",
                        help="開いているWebページが無くなったらサーバーを停止する（web.bat非表示起動用）")
    args = parser.parse_args()

    selection = resolve_model_selection(args.model, MODEL_CONFIG)
    chat_model = selection.model
    if sys.platform != "darwin" and selection.source == "model-file":
        chat_model = detect_model()
    model_digest = None
    if sys.platform == "darwin":
        if chat_model not in MAC_MODEL_DIGESTS:
            parser.error("Mac起動では固定digestを持つqwen3.5:122bまたはgpt-oss:120bを選択してください")
        try:
            model_digest = get_model_identity(chat_model)["digest"]
        except (OSError, TimeoutError, ModelConfigurationError) as exc:
            parser.error(f"Macの選択モデルをdigestまで検証できません ({type(exc).__name__})")
    generation_stall_timeout = model_call_timeout_seconds(
        chat_model, GENERATION_STALL_TIMEOUT
    )

    # ログ設定
    log_handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=log_handlers,
    )

    # タイムアウト設定（CLI > 環境変数 > デフォルト）。300〜600秒の整数のみを
    # 受理し、範囲外・小数・非数値はfail-fastする（config.validate_search_timeout_seconds）。
    if args.search_timeout is not None:
        search_timeout = validate_search_timeout_seconds(args.search_timeout)
    else:
        env_raw = os.environ.get("OFFLINEAI_SEARCH_TIMEOUT", "").strip()
        search_timeout = (
            validate_search_timeout_seconds(env_raw) if env_raw else DEFAULT_SEARCH_TIMEOUT
        )

    # ポート決定
    if args.port:
        port = args.port
    else:
        port = find_available_port()

    # セッショントークン生成
    session_token = secrets.token_urlsafe(32)

    # サーバー起動
    server = LimitedThreadingServer(
        ("127.0.0.1", port),
        OfflineAIHandler,
        session_token=session_token,
        search_timeout=search_timeout,
        bind_port=port,
        page_presence=PagePresenceMonitor() if args.exit_when_page_closed else None,
        chat_model=chat_model,
        model_selection_source=selection.source,
        model_digest=model_digest,
        generation_stall_timeout=generation_stall_timeout,
    )
    # 前回プロセスが所有者不在のまま残した building/cancelling を実測へ戻す（F-9）。
    server.index_coordinator.reconcile_orphaned_state()

    # URL 構築
    token_display = session_token if args.show_token else mask_token(session_token)
    url = f"http://127.0.0.1:{port}/bootstrap#token={session_token}"
    display_url = f"http://127.0.0.1:{port}/bootstrap#token={token_display}"

    print(f"offline-ai Web UI started on port {port}", file=sys.stderr)
    print(f"URL: {display_url}", file=sys.stderr)
    print(f"Search timeout: {search_timeout}s", file=sys.stderr)
    print(f"Chat model: {chat_model}", file=sys.stderr)
    if model_digest:
        print(f"Chat model digest: {model_digest}", file=sys.stderr)

    # コンソール出力（web.bat から見える）
    print(f"\noffline-ai Web UI")
    print(f"URL: {display_url}")
    print(f"Press Ctrl+C to stop.\n")

    # ブラウザ自動オープン
    try:
        webbrowser.open(url)
    except Exception:
        print("(Browser auto-open failed. Open the URL manually.)", file=sys.stderr)

    # シグナルハンドラ（正常シャットダウン）
    def shutdown_handler(signum, frame):
        start = time.monotonic()
        print("\nShutting down...", file=sys.stderr)
        active_jobs = server.shutdown_jobs_once()
        threading.Thread(target=server.shutdown, daemon=True).start()
        with server._tokens_lock:
            cancelled_jobs = max(0, active_jobs - len(server.active_tokens))
        latency = int((time.monotonic() - start) * 1000)
        log_structured("shutdown", phase="shutdown",
                       active_jobs=active_jobs,
                       cancelled_jobs=cancelled_jobs,
                       shutdown_latency_ms=latency)

    signal.signal(signal.SIGINT, shutdown_handler)
    try:
        signal.signal(signal.SIGTERM, shutdown_handler)
    except (OSError, AttributeError):
        pass  # Windows では SIGTERM が使えない場合がある
    try:
        signal.signal(signal.SIGBREAK, shutdown_handler)
    except (OSError, AttributeError):
        pass  # SIGBREAK は Windows 専用

    watchdog_stop = threading.Event()
    if server.page_presence is not None:
        def stop_after_pages_closed():
            log_structured("shutdown", phase="page_closed",
                           open_pages=server.page_presence.open_page_count())
            shutdown_handler(None, None)

        threading.Thread(
            target=run_page_presence_watchdog,
            args=(server, server.page_presence, watchdog_stop, stop_after_pages_closed),
            daemon=True,
            name="offline-ai-page-presence",
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown_handler(None, None)
    finally:
        watchdog_stop.set()
        server.shutdown_jobs_once()
        server.server_close()


if __name__ == "__main__":
    main()
