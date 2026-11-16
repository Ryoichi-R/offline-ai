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
# EvidenceViewError は web_server が再エクスポートするサービス境界の例外でもある。
from source_view import EvidenceRegistry, EvidenceViewError

logger = logging.getLogger("offlineai.services")

SCRIPT_DIR = Path(__file__).resolve().parent


VALID_SEARCH_MODES = {"answer", "search", "deep"}

# Deep replay is deliberately bounded so a reconnect cannot retain an
# unbounded amount of model/user-material output. Public deep payloads are
# compacted before they enter this buffer; these limits remain a final safety
# boundary for unexpected payload growth.
DEEP_REPLAY_MAX_EVENTS = 32
DEEP_REPLAY_MAX_BYTES = 1_000_000
DEEP_PUBLIC_LEDGER_MAX = 256
DEEP_PUBLIC_WARNING_MAX = 256
DEEP_PUBLIC_TEXT_MAX_BYTES = 600_000

# 回答生成状態の利用者向け文言。confidence固定表示(1.0/0.0)は根拠の有無しか
# 表さず、回答未生成調査(2026-09-23)の指摘4のとおり回答品質の指標ではない
# ため、deepではconfidenceを送らずこちらへ置き換える。
DEEP_ANSWER_STATE_MESSAGES = {
    "generated": "回答生成済み（原文照合を通過）",
    "verification_failed": "検証に失敗したため原文抜粋を表示",
    "not_generated": "回答は生成されず原文抜粋を表示",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


# 根拠の抜粋（SSE・保存Markdown）の文字数枠。範囲が長い場合は、枠内で必要箇所を
# 優先した連続行を選び、抜粋自体の行範囲と省略の有無を返す（guided-follow-up 4.1）。
EVIDENCE_EXCERPT_MAX_CHARS = 500


def _bigrams(text: str) -> set[str]:
    import unicodedata

    normalized = "".join(unicodedata.normalize("NFKC", str(text or "")).split())
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


EXCERPT_MAX_SEGMENTS = 3
EXCERPT_SEGMENT_SEPARATOR = "\n…\n"
EXCERPT_SEED_MIN_COVERAGE = 0.2


def _focus_segments(lines: list[str], focus, limit: int) -> list[tuple[int, int]]:
    """回答の各主張に最も対応する行を種にして、枠内で最大3区間を選ぶ（R4）。

    主張ごとに、主張の2文字単位を最も多く含む行を種にする（同率なら先頭側）。
    種が2つ未満なら空を返し、呼び出し側は従来の連続1区間を使う。区間は枠の残りで
    前後の行へ広げ、隣接・重複した区間はまとめる。原文の文言は変えない。
    """
    line_grams = [_bigrams(line) for line in lines]
    seeds: list[int] = []
    for value in focus:
        grams = _bigrams(value)
        if not grams:
            continue
        best = max(range(len(lines)), key=lambda i: (len(line_grams[i] & grams), -i))
        if len(line_grams[best] & grams) / len(grams) >= EXCERPT_SEED_MIN_COVERAGE and best not in seeds:
            seeds.append(best)
    if len(seeds) < 2:
        return []
    separator = len(EXCERPT_SEGMENT_SEPARATOR)
    segments: list[list[int]] = []
    used = 0
    for seed in seeds:
        if len(segments) >= EXCERPT_MAX_SEGMENTS:
            break
        if any(a - 1 <= seed <= b + 1 for a, b in segments):
            continue
        cost = len(lines[seed]) + (separator if segments else 0)
        if used + cost > limit:
            continue
        segments.append([seed, seed])
        used += cost
    if len(segments) < 2:
        return []

    def taken(index: int) -> bool:
        return any(a <= index <= b for a, b in segments)

    grew = True
    while grew:
        grew = False
        for segment in sorted(segments):
            for index, edge in ((segment[1] + 1, 1), (segment[0] - 1, 0)):
                if 0 <= index < len(lines) and not taken(index) and used + len(lines[index]) + 1 <= limit:
                    segment[edge] = index
                    used += len(lines[index]) + 1
                    grew = True
    merged: list[list[int]] = []
    for first, end in sorted(segments):
        if merged and first <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([first, end])
    result = []
    for first, end in merged:
        while first < end and not lines[first].strip():
            first += 1
        while end > first and not lines[end].strip():
            end -= 1
        result.append((first, end))
    return result


def select_excerpt(
    text: str,
    start_line: int | None,
    *,
    focus=(),
    limit: int = EVIDENCE_EXCERPT_MAX_CHARS,
) -> dict:
    """根拠範囲の本文から、枠内で必要箇所を優先した連続行の抜粋を選ぶ。

    ``focus``（回答本文・抽出内容・質問など）と2文字単位で多く重なる連続行を選び、
    同点なら先頭に近い範囲を採る。重なりがなければ従来どおり先頭から採る。
    1行が枠を超える場合だけ、その行を枠で切る。原文の文言は変えない。
    ``start_line`` が本文1行目の行番号のとき、抜粋自体の行範囲を返す。
    """
    lines = str(text or "").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
        start_line = start_line + 1 if isinstance(start_line, int) else start_line
    while lines and not lines[-1].strip():
        lines.pop()
    joined = "\n".join(lines)
    aligned = isinstance(start_line, int) and not isinstance(start_line, bool)
    if len(joined) <= limit:
        return {
            "snippet": joined,
            "startLine": start_line if aligned and lines else None,
            "endLine": start_line + len(lines) - 1 if aligned and lines else None,
            "truncated": False,
        }
    segments = _focus_segments(lines, focus, limit)
    if segments:
        texts = []
        for first, end in segments:
            texts.append("\n".join(lines[first : end + 1]) if first != end or len(lines[first]) <= limit
                         else lines[first][:limit])
        result = {
            "snippet": EXCERPT_SEGMENT_SEPARATOR.join(texts),
            "startLine": start_line + segments[0][0] if aligned else None,
            "endLine": start_line + segments[-1][1] if aligned else None,
            "truncated": True,
        }
        if len(segments) > 1:
            result["segments"] = [
                {
                    "startLine": start_line + first if aligned else None,
                    "endLine": start_line + end if aligned else None,
                    "text": text,
                }
                for (first, end), text in zip(segments, texts)
            ]
        return result
    focus_grams: set[str] = set()
    for value in focus:
        focus_grams |= _bigrams(value)
    scores = [len(_bigrams(line) & focus_grams) for line in lines]
    best = None
    for first in range(len(lines)):
        if not lines[first].strip():
            continue
        total = score = 0
        last = first
        while last < len(lines):
            cost = len(lines[last]) + (1 if last > first else 0)
            if total + cost > limit:
                break
            total += cost
            score += scores[last]
            last += 1
        if last == first:
            window = (scores[first], -first, first, first, lines[first][:limit])
        else:
            end = last - 1
            while end > first and not lines[end].strip():
                end -= 1
            window = (score, -first, first, end, "\n".join(lines[first : end + 1]))
        if best is None or window[:2] > best[:2]:
            best = window
    _, _, first, end, snippet = best
    return {
        "snippet": snippet,
        "startLine": start_line + first if aligned else None,
        "endLine": start_line + end if aligned else None,
        "truncated": True,
    }


def _source_range_text(registry, relative_path, sha256, start_line, end_line, cache):
    """検索時と同じハッシュの資料から根拠範囲の原文を読む。読めなければ None。"""
    import hashlib

    from source_view import MAX_SOURCE_FILE_BYTES, _assert_path_boundary

    if registry is None or not isinstance(start_line, int) or not isinstance(end_line, int):
        return None
    key = (relative_path, sha256)
    if key not in cache:
        cache[key] = None
        try:
            path = _assert_path_boundary(registry.source_root, relative_path)
            data = path.read_bytes()
            if len(data) <= MAX_SOURCE_FILE_BYTES and hashlib.sha256(data).hexdigest() == str(sha256).lower():
                cache[key] = data.decode("utf-8", errors="replace").splitlines()
        except (EvidenceViewError, OSError, ValueError):
            cache[key] = None
    lines = cache[key]
    if lines is None or start_line < 1 or end_line < start_line or start_line > len(lines):
        return None
    return "\n".join(lines[start_line - 1 : min(end_line, len(lines))])


def _apply_excerpt(item: dict, selected: dict) -> None:
    item["snippet"] = selected["snippet"]
    if selected["startLine"] is not None:
        item["snippetStartLine"] = selected["startLine"]
        item["snippetEndLine"] = selected["endLine"]
    item["snippetTruncated"] = bool(selected["truncated"])
    if selected.get("segments"):
        # 区間ごとの行範囲。snippet は区間を「…」でつないだ文字列（互換）。
        item["snippetSegments"] = [dict(segment) for segment in selected["segments"]]


def create_evidence_registry(source_root: Path | None = None) -> EvidenceRegistry:
    return EvidenceRegistry(source_root or SKILL_SOURCE_DIR)


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
    source_cache: dict = {}
    query_text = str(getattr(retrieval, "query", "") or "")
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
            "snippet": "",
            "sourceSha256": source_sha256,
        }
        # 行範囲の原文（検索時と同じハッシュ）から抜粋し、抜粋自体の行範囲を記録する。
        # 読めない場合は検索時の抜粋を使い、行範囲は記録しない。
        range_text = _source_range_text(
            evidence_registry, item["path"], source_sha256, item["startLine"], item["endLine"], source_cache
        )
        if range_text is not None:
            _apply_excerpt(item, select_excerpt(range_text, item["startLine"], focus=(query_text,)))
        else:
            _apply_excerpt(item, select_excerpt(str(match.get("snippet") or ""), None, focus=(query_text,)))
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
    query: str = "",
) -> dict:
    """deep結果を既存の根拠表示契約へ変換する。

    未確認事項と調査台帳は、長い実資料で一気に膨らむ可能性がある。
    SSEの再接続バッファへ渡す公開イベントは、利用者が再接続後に
    確認できる範囲を保ったまま、個別文字列と件数を制限する。
    """
    items = []
    answer_lines = str(getattr(result, "answer", "") or "").splitlines()
    for evidence in list(getattr(result, "evidence", [])):
        path = str(evidence.get("path") or "")
        cited = f"[{evidence.get('evidence_id')}]" if evidence.get("evidence_id") else ""
        focus = [line for line in answer_lines if cited and cited in line]
        for key in ("subject", "scope"):
            if evidence.get(key):
                focus.append(str(evidence[key]))
        for key in ("conditions", "exceptions"):
            focus.extend(str(value) for value in evidence.get(key) or [])
        if not focus and query:
            focus.append(query)
        source_sha256 = str(evidence.get("source_sha256") or "")
        item = {
            "path": path,
            "sourceTitle": path,
            "heading": str(evidence.get("heading") or ""),
            "startLine": evidence.get("start_line"),
            "endLine": evidence.get("end_line"),
            "snippet": "",
            "sourceSha256": source_sha256,
            "evidenceId": str(evidence.get("evidence_id") or ""),
            "deepStatus": str(evidence.get("verification_status") or ""),
            "viewable": False,
        }
        # 読んだ単位（最大4000文字）の本文は行範囲そのもの。枠内で回答・抽出内容に
        # 対応する箇所を優先し、抜粋自体の行範囲と省略の有無を記録する。
        _apply_excerpt(
            item,
            select_excerpt(str(evidence.get("excerpt") or ""), item["startLine"], focus=focus),
        )
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
    stop_reason = str(getattr(result, "stop_reason", ""))
    raw_warnings = [
        str(value).strip()[:500]
        for value in getattr(result, "unconfirmed", [])
        if str(value).strip()
    ]
    reason_message = stop_reason_message(stop_reason)
    if reason_message and reason_message not in raw_warnings:
        raw_warnings.insert(0, reason_message)
    # Bulk, per-candidate reasons (candidate_limit等) can vastly outnumber the
    # single important entries (model_error等) that a large-corpus run
    # produces. Sort those to the back so truncation drops repetitive bulk
    # entries first, not the one model_error a run hit (2026-09-23
    # investigation, finding 3: important failures were lost past the
    # display cap).
    bulk_prefixes = tuple(f"{reason}:" for reason in DEEP_BULK_UNCONFIRMED_REASONS)
    important = [w for w in raw_warnings if not w.startswith(bulk_prefixes)]
    bulk = [w for w in raw_warnings if w.startswith(bulk_prefixes)]
    raw_warnings = important + bulk
    warnings_truncated = len(raw_warnings) > DEEP_PUBLIC_WARNING_MAX
    if warnings_truncated:
        omitted = len(raw_warnings) - DEEP_PUBLIC_WARNING_MAX
        raw_warnings = raw_warnings[: DEEP_PUBLIC_WARNING_MAX - 1]
        raw_warnings.append(f"未確認事項は表示上限のため{omitted}件省略しました")

    raw_ledger = list(getattr(result, "ledger", []))
    # As with warnings: per-candidate bulk entries go last so the display cap
    # keeps single entries such as "irrelevant" (read but judged unrelated).
    raw_ledger = [
        e for e in raw_ledger if not (isinstance(e, dict) and e.get("reason") in DEEP_BULK_UNCONFIRMED_REASONS)
    ] + [
        e for e in raw_ledger if isinstance(e, dict) and e.get("reason") in DEEP_BULK_UNCONFIRMED_REASONS
    ]
    ledger_truncated = len(raw_ledger) > DEEP_PUBLIC_LEDGER_MAX
    public_ledger = []
    for entry in raw_ledger[:DEEP_PUBLIC_LEDGER_MAX]:
        if not isinstance(entry, dict):
            continue
        public_entry = {}
        for key, value in entry.items():
            if isinstance(value, str):
                public_entry[key] = value[:500]
            elif value is None or isinstance(value, (bool, int, float)):
                public_entry[key] = value
        public_ledger.append(public_entry)
    if ledger_truncated:
        public_ledger.append(
            {
                "path": "",
                "status": "unconfirmed",
                "reason": "replay_truncated",
                "detail": f"調査台帳は表示上限のため{len(raw_ledger) - DEEP_PUBLIC_LEDGER_MAX}件省略しました",
            }
        )

    answer_state = str(getattr(result, "answer_state", ANSWER_STATE_NOT_GENERATED))
    answer_state_message = DEEP_ANSWER_STATE_MESSAGES.get(answer_state, "")
    answer_quality = dict(getattr(result, "answer_quality", {}) or {})
    if answer_quality:
        # v2: generated は生成状態であり、品質合格とは別。要確認のときは合格と読める
        # 文言を出さない。
        answer_state_message = {
            "passed": "回答生成済み（品質検査を通過。正確性の保証ではありません）",
            "failed": "回答を作成できず参考の原文を表示（要確認）",
        }.get(str(answer_quality.get("state")), "回答生成済み（要確認）")
    if (
        getattr(result, "status", "") == "completed"
        and answer_state == ANSWER_STATE_NOT_GENERATED
        and (getattr(result, "diagnostics", {}) or {}).get("final_answer", {}).get("result")
        == "abstained_no_relevant_evidence"
    ):
        answer_state_message = "資料の確認を完了しましたが、質問に答える根拠はありません"
    event = {
        "type": "evidence",
        "evidenceStatus": str(getattr(result, "status", "partial")),
        "verificationNote": "deepは原文行範囲とSHA-256を照合した根拠だけを表示します",
        # confidenceは根拠の有無だけを表す値(1.0/0.0)を返していたため廃止する。
        # 回答が実際に生成・検証されたかは answerState / answerStateMessage を見る。
        "confidence": None,
        "answerState": answer_state,
        "answerStateMessage": answer_state_message,
        "route": "deep",
        "indexState": "deep",
        "routeReason": stop_reason,
        "routeReasonMessage": reason_message,
        "items": items,
        "warnings": raw_warnings,
        "deepLedger": public_ledger,
        "deepDiagnostics": dict(getattr(result, "diagnostics", {}) or {}),
    }
    if answer_quality:
        event["answerQuality"] = answer_quality
    if warnings_truncated:
        event["deepWarningsTruncated"] = True
    if ledger_truncated:
        event["deepLedgerTruncated"] = True
    return event


def _truncate_deep_text(value: object, *, max_bytes: int = DEEP_PUBLIC_TEXT_MAX_BYTES) -> str:
    """Keep a public deep answer below the replay payload safety limit."""
    text = str(value or "")
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    marker = "\n\n[回答本文は再接続用保持上限のため省略されました]"
    marker_bytes = marker.encode("utf-8")
    keep = max(0, max_bytes - len(marker_bytes))
    return encoded[:keep].decode("utf-8", errors="ignore") + marker

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
                  "_event_buffer", "_lock", "released", "deep_slot")

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
        # 深掘りの実調査枠を保持しているか。終了イベントで state が running でなく
        # なった時点で枠は空く（解放は一度だけ）。
        self.deep_slot = False

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
                if len(proposed) > DEEP_REPLAY_MAX_EVENTS or len(json.dumps(proposed, ensure_ascii=False).encode("utf-8")) > DEEP_REPLAY_MAX_BYTES:
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
               mode: str | None = None,
               resume_only: bool = False,
               defer_deep_slot: bool = False) -> tuple[JobEntry, bool]:
        """ジョブを登録する。

        ``defer_deep_slot``: 確認段階を経る深掘りジョブは登録時に深掘り枠を取らず、
        実調査の開始時に ``acquire_deep_slot`` で取る（確認段階は枠を消費しない）。

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
            if resume_only:
                raise JobNotFoundError("search job is no longer available")
            if mode == "deep" and not defer_deep_slot and self._deep_busy_locked():
                raise RuntimeError("deep_busy")
            if self._running_count >= self._max_concurrent:
                raise RuntimeError("server_busy")
            entry = JobEntry(cancel_token, fingerprint=fingerprint)
            entry.deep_slot = mode == "deep" and not defer_deep_slot
            self._jobs[request_id] = entry
            self._running_count += 1
            return entry, True

    def _deep_busy_locked(self) -> bool:
        return any(
            entry.state == "running" and entry.deep_slot
            for entry in self._jobs.values()
        )

    def acquire_deep_slot(self, request_id: str) -> bool:
        """深掘りの実調査枠を同じロック内で確認・取得する。取れなければ False。"""
        with self._lock:
            entry = self._jobs.get(request_id)
            if entry is None or entry.state != "running":
                return False
            if entry.deep_slot:
                return True
            if self._deep_busy_locked():
                return False
            entry.deep_slot = True
            return True

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


class JobNotFoundError(RuntimeError):
    """Resume must never create a replacement job after expiry or restart."""


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
    from model_config import (
        MAC_MODEL_DIGESTS, ModelConfigurationError, resolve_model_reasoning,
        model_call_timeout_seconds,
    )
    from search import (
        PromptBudgetError,
        detect_model,
        iter_stream_events,
        build_chat_payload,
        RETRIEVAL_TIMEOUT,
        EVIDENCE_VERIFY_GENERATION_RESERVE,
        GENERATION_STALL_TIMEOUT,
        run_retrieval_pipeline,
        get_model_identity,
        run_answer_v2,
        SKILL_SOURCE_DIR,
    )
    from answer_policy import POLICY_V2, resolve_policy_version
    from deep_research import (
        CANDIDATE_TRANSFER_LIMIT_CODE,
        CANDIDATE_TRANSFER_LIMIT_MESSAGE,
        DEEP_BULK_UNCONFIRMED_REASONS,
        DEEP_TIMEOUT_DEFAULT,
        ANSWER_STATE_GENERATED,
        ANSWER_STATE_VERIFICATION_FAILED,
        ANSWER_STATE_NOT_GENERATED,
        run_deep_research,
        detect_deep_model,
        stop_reason_message,
    )

    _search_available = True
except Exception as e:
    _search_error = str(e)
    logger.warning("search.py import failed: %s", e)
    # web_server.py が無条件に import する定数のフォールバック
    # （search.py 側の既定値と揃える）。
    GENERATION_STALL_TIMEOUT = 60
    DEEP_TIMEOUT_DEFAULT = DEEP_SEARCH_TIMEOUT_DEFAULT
    CANDIDATE_TRANSFER_LIMIT_CODE = "candidate_transfer_limit"
    CANDIDATE_TRANSFER_LIMIT_MESSAGE = (
        "検索候補の内部転送容量を超えたため、調査を継続できませんでした"
    )
    DEEP_BULK_UNCONFIRMED_REASONS = frozenset(
        {"candidate_limit", "document_limit", "unit_limit", "viewpoint_limit", "context_budget"}
    )
    ANSWER_STATE_GENERATED = "generated"
    ANSWER_STATE_VERIFICATION_FAILED = "verification_failed"
    ANSWER_STATE_NOT_GENERATED = "not_generated"

    def stop_reason_message(reason: str) -> str:
        return CANDIDATE_TRANSFER_LIMIT_MESSAGE if reason == CANDIDATE_TRANSFER_LIMIT_CODE else ""

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
    "source_settings_unavailable": 500,
    "invalid_query": 400,
    "invalid_timeout": 400,
    "invalid_mode": 400,
    "invalid_evidence_id": 400,
    "unauthorized": 403,
    "request_conflict": 409,
    "conversation_not_found": 404,
    "conversation_full": 409,
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
    "confirmation_expired": 410,
    "confirmation_invalid": 409,
    "confirmation_busy": 409,
    "deep_busy": None,            # SSE only（確認段階を経た深掘り）
    "service_unavailable": None,  # SSE only
    "timeout": None,              # SSE only
    "cancelled": None,            # SSE only
    "deep_failed": None,          # SSE only
    "candidate_transfer_limit": None,  # SSE only
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

def check_health(configured_model: str | None = None) -> dict:
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
                    model_name = configured_model or detect_model()
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

def _run_search_core(query: str, reasoning: str, event_queue: queue.Queue,
                     cancel_token: CancellationToken, request_id: str, *,
                     mode: str = "answer", started_at: str | None = None,
                     session_id: str = "", evidence_registry: EvidenceRegistry | None = None,
                     model_name: str | None = None, model_digest: str | None = None,
                     model_selection_source: str | None = None,
                     fixed_document: dict | None = None,
                     acquire_deep_slot=None):
    """検索処理をバックグラウンドスレッドで実行し、イベントを queue に put する。

    イベント形式: {"type": "status"|"chunk"|"done"|"error", ...}
    ``fixed_document`` は確認案内で固定した資料（根拠をこの資料に限る）。
    ``acquire_deep_slot`` は確認段階を経た深掘りが実調査の直前に枠を取る関数。
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

        # 回答方針は要求ごとに一度だけ解決し、この要求の全経路を固定する。
        try:
            answer_policy = resolve_policy_version()
        except ValueError:
            event_queue.put({"type": "error", "code": "invalid_answer_policy",
                             "message": "回答方針の設定が不正です"})
            return

        # 1. 回答モードだけが chat model を検出する。検索専用はモデル通信を
        # 行わず、retrieval 内の keyword / optional embedding だけで完了する。
        model = None
        if mode == "answer":
            event_queue.put({"type": "status", "text": "モデルを検出しています..."})
            model = model_name or detect_model()
            log_structured(request_id, phase="detect_model", model=model)
        elif mode == "deep":
            event_queue.put({"type": "status", "text": "深掘り調査用モデルを検出しています..."})
            model = model_name or detect_deep_model()
            log_structured(request_id, phase="detect_model", model=model, mode=mode)

        if model_name is not None and model in MAC_MODEL_DIGESTS:
            observed_identity = get_model_identity(model)
            if model_digest is not None and observed_identity["digest"] != model_digest:
                raise ModelConfigurationError("起動後に選択モデルのdigestが変わりました")
            model_digest = observed_identity["digest"]
        reasoning_contract = (
            resolve_model_reasoning(model, reasoning if mode in {"answer", "deep"} else "off")
            if model else None
        )
        cancel_token.check()

        if mode == "deep":
            def emit_deep_progress(event: dict) -> None:
                cancel_token.touch()
                event_queue.put(event)

            if acquire_deep_slot is not None and not acquire_deep_slot():
                # 確認段階は深掘り枠を消費しない。実調査の開始時に枠を取れなければ
                # 入力を画面に残したまま deep_busy で完了する。
                event_queue.put({
                    "type": "error", "code": "deep_busy",
                    "message": "別の深掘り調査を実行中です。終わってから再試行してください。",
                })
                return
            deep_kwargs = {"fixed_document": fixed_document} if fixed_document else {}
            deep_result = run_deep_research(
                query,
                model=model or "",
                reasoning=reasoning,
                source_root=SKILL_SOURCE_DIR,
                timeout_seconds=int(cancel_token.timeout_seconds),
                emit_progress=emit_deep_progress,
                cancel_check=cancel_token.check,
                absolute_deadline=cancel_token.deadline,
                answer_policy=answer_policy,
                **deep_kwargs,
            )
            cancel_token.touch()
            event_queue.put({
                "type": "result_meta",
                "requestId": request_id,
                "answerPolicy": answer_policy,
                "mode": "deep",
                "startedAt": started_at or _utc_now_iso(),
                "chatModel": model,
                "embeddingModel": deep_result.diagnostics.get("embedding_model"),
                "reasoning": reasoning_contract["requested"] if reasoning_contract else "off",
                "requestedReasoning": reasoning_contract["requested"] if reasoning_contract else "off",
                "effectiveReasoning": reasoning_contract["effective"] if reasoning_contract else "off",
                "apiThinkValue": reasoning_contract["think"] if reasoning_contract else False,
                "thinkingVisible": False,
                "modelDigest": model_digest,
                "modelSelectionSource": model_selection_source,
                "deepStatus": deep_result.status,
                "stopReason": deep_result.stop_reason,
                "stopReasonMessage": stop_reason_message(deep_result.stop_reason),
            })
            event_queue.put(build_deep_evidence_event(
                deep_result,
                evidence_registry=evidence_registry,
                session_id=session_id,
                query=query,
            ))
            if deep_result.answer:
                event_queue.put({"type": "chunk", "text": _truncate_deep_text(deep_result.answer)})
            if deep_result.status == "cancelled":
                event_queue.put({
                    "type": "error",
                    "code": "cancelled",
                    "message": "深掘り調査を中止しました",
                })
            elif deep_result.status == "failed" and not deep_result.evidence:
                error_code = (
                    deep_result.stop_reason
                    if deep_result.stop_reason == CANDIDATE_TRANSFER_LIMIT_CODE
                    else "deep_failed"
                )
                event_queue.put({
                    "type": "error",
                    "code": error_code,
                    "message": stop_reason_message(deep_result.stop_reason)
                    or "深掘り調査を確定できませんでした",
                })
            else:
                done_event = {
                    "type": "done",
                    "completedAt": _utc_now_iso(),
                    "status": deep_result.status,
                    "stopReason": deep_result.stop_reason,
                    "stopReasonMessage": stop_reason_message(deep_result.stop_reason),
                }
                if deep_result.answer_quality:
                    done_event["answerQuality"] = dict(deep_result.answer_quality)
                event_queue.put(done_event)
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
        request_remaining = max(0.0, cancel_token.remaining())
        retrieval_limit = RETRIEVAL_TIMEOUT
        if mode == "answer" and model in MAC_MODEL_DIGESTS:
            # Cold large-model planning can exceed the legacy 120s budget.
            # Keep generation time inside the unchanged request-wide deadline.
            generation_reserve = min(EVIDENCE_VERIFY_GENERATION_RESERVE, request_remaining / 2)
            retrieval_limit = min(
                model_call_timeout_seconds(model, RETRIEVAL_TIMEOUT),
                request_remaining - generation_reserve,
            )
        retrieval_token = CancellationToken(min(retrieval_limit, request_remaining))

        def retrieval_remaining() -> float:
            return min(cancel_token.remaining(), retrieval_token.remaining())

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
            # Planning and evidence verification must respect both deadlines.
            pipeline_kwargs["remaining_seconds"] = retrieval_remaining
        if "answer_policy" in parameters:
            pipeline_kwargs["answer_policy"] = answer_policy
        if fixed_document:
            pipeline_kwargs["fixed_document"] = fixed_document
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
            "answerPolicy": answer_policy if mode == "answer" else None,
            "mode": mode,
            "startedAt": started_at or _utc_now_iso(),
            "chatModel": model if mode == "answer" else None,
            "embeddingModel": getattr(retrieval, "embedding_model", None),
            "reasoning": "off" if mode == "search" else effective_reasoning,
            "requestedReasoning": reasoning_contract["requested"] if reasoning_contract else "off",
            "effectiveReasoning": reasoning_contract["effective"] if reasoning_contract else "off",
            "apiThinkValue": reasoning_contract["think"] if reasoning_contract else False,
        "thinkingVisible": (
            reasoning_contract["thinking_visible"]
            if reasoning_contract and mode == "answer"
            else False
        ),
            "modelDigest": model_digest,
            "modelSelectionSource": model_selection_source,
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

        if mode == "answer" and answer_policy == POLICY_V2:
            # 回答方針v2: 主張単位の草稿→原文・意味・質問適合・文章の検査→検査済みの主張だけを
            # 一度に表示する（途中の未検査本文をストリーミングしない）。
            cancel_token.start_stall_monitoring()
            event_queue.put({"type": "status",
                             "text": "回答を構成し、原文・意味・質問への適合を検査しています..."})
            outcome = run_answer_v2(
                query,
                retrieval,
                model=model,
                reasoning=reasoning if reasoning in ("low", "medium", "high") else None,
                total_seconds=float(cancel_token.timeout_seconds),
                remaining_seconds=cancel_token.remaining,
                cancel_check=cancel_token.check,
                touch=cancel_token.touch,
            )
            cancel_token.touch()
            event_queue.put({"type": "chunk", "text": _truncate_deep_text(outcome.answer)})
            event_queue.put({
                "type": "done",
                "completedAt": _utc_now_iso(),
                "answerQuality": outcome.quality,
            })
            log_structured(
                request_id,
                phase="complete",
                answer_policy="v2",
                quality_state=outcome.quality.get("state"),
                kind=outcome.kind,
                llm_calls=outcome.diagnostics.get("calls", {}).get("llm_calls"),
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
                f"モデルからの応答が{int((cancel_token.stall_timeout or GENERATION_STALL_TIMEOUT))}秒途絶えました",
                code="stall",
            )

        try:
            try:
                with urllib.request.urlopen(
                    req, timeout=min(remaining, (cancel_token.stall_timeout or GENERATION_STALL_TIMEOUT))
                ) as resp:
                    stream_response(resp)
            except urllib.error.HTTPError as exc:
                # think キーが原因で拒否された可能性がある場合のみ、除去して1回だけ再送する。
                # 親 cancel_token の残時間内で行い、全体 deadline は延長しない。
                if model not in MAC_MODEL_DIGESTS and "think" in body and exc.code >= 400:
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
                        retry_req, timeout=min(retry_remaining, (cancel_token.stall_timeout or GENERATION_STALL_TIMEOUT))
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


# ---------------------------------------------------------------------------
# 確認案内と追質問（guided-follow-up 段階2、既定は無効）
# ---------------------------------------------------------------------------


class _GuidedQueue:
    """回答の正常完了（stage のない done）だけで確認情報を更新するキュー。"""

    def __init__(self, inner, on_finish):
        self._inner = inner
        self._on_finish = on_finish
        self.finished = False

    def put(self, event: dict):
        if not self.finished and event.get("type") == "done" and not event.get("stage"):
            self.finished = True
            state = self._on_finish(True)
            if state:
                self._inner.put({"type": "confirmation_state", **state})
        elif not self.finished and event.get("type") == "error":
            self.finished = True
            self._on_finish(False)
        self._inner.put(event)


def _index_marker() -> list | None:
    try:
        from search import EMBED_CACHE_PATH

        stat = Path(EMBED_CACHE_PATH).stat()
        return [stat.st_mtime_ns, stat.st_size]
    except (ImportError, OSError):
        return None


def _guided_source_ok(document: dict) -> bool:
    """参考資料フォルダー・資料のハッシュ・インデックスが確定時から変わっていないか。"""
    from guided_follow_up import read_document

    try:
        active = Path(SKILL_SOURCE_DIR).resolve()
        if document.get("sourceRoot") != str(active):
            return False
        from source_folder import load_source_root

        if load_source_root(SCRIPT_DIR.parent).resolve() != active:
            return False  # 参考資料フォルダーの切替が再起動待ち
    except (ImportError, OSError, ValueError):
        return False
    if document.get("indexMarker") != _index_marker():
        return False
    return all(read_document(active, part["path"], part["sha256"]) is not None for part in _document_paths(document))


def _document_paths(document: dict) -> list[dict]:
    """固定資料の部分のパスとハッシュ（分割していない資料は1件）。"""
    return list(document.get("paths") or [{"path": document["path"], "sha256": document.get("sha256", "")}])


def _with_document_text(document: dict) -> dict | None:
    """固定資料の全部分を、確定時と同じハッシュで読み直す。"""
    from guided_follow_up import read_document

    parts = []
    for member in _document_paths(document):
        read = read_document(Path(SKILL_SOURCE_DIR), member["path"], member["sha256"])
        if read is None:
            return None
        parts.append({"path": member["path"], "sha256": read[1], "text": read[0]})
    return {**document, "text": parts[0]["text"], "parts": parts}


def _fixed_for_answer(document: dict, **extra) -> dict:
    """回答処理（通常回答・深掘り）へ渡す固定資料。本文は渡さない。"""
    return {
        "path": document["path"], "title": document["title"], "sha256": document["sha256"],
        "paths": [{"path": m["path"], "sha256": m["sha256"]} for m in _document_paths(document)],
        **extra,
    }


def _emit_clarification(event_queue, request_id: str, public: dict, kind: str, ask: dict | None = None) -> None:
    from guided_follow_up import clarification_event

    event_queue.put(clarification_event(public, kind=kind, ask=ask))
    # 確認段階では result_meta と chunk を送らない。stage で通常の完了と区別する。
    event_queue.put({"type": "done", "stage": "clarify", "requestId": request_id})


def run_search(query: str, reasoning: str, event_queue: queue.Queue,
               cancel_token: CancellationToken, request_id: str, *,
               mode: str = "answer", started_at: str | None = None,
               session_id: str = "", evidence_registry: EvidenceRegistry | None = None,
               model_name: str | None = None, model_digest: str | None = None,
               model_selection_source: str | None = None,
               confirmation_store=None, guided_request: dict | None = None,
               acquire_deep_slot=None):
    """検索処理。確認案内が有効なら、モデル検出の前に資料と質問範囲を振り分ける。

    無効時（既定）は従来と同じ経路・イベント・モデル呼び出し回数で
    ``_run_search_core`` を呼ぶ。
    """
    core_kwargs = {
        "mode": mode, "started_at": started_at, "session_id": session_id,
        "evidence_registry": evidence_registry, "model_name": model_name,
        "model_digest": model_digest, "model_selection_source": model_selection_source,
    }
    try:
        from guided_follow_up import guided_enabled
    except ImportError:
        guided_enabled = None
    guided_on = (
        confirmation_store is not None
        and guided_enabled is not None
        and guided_enabled()
        and mode in {"answer", "deep"}
        and _search_available
    )
    if not guided_on:
        _run_search_core(query, reasoning, event_queue, cancel_token, request_id,
                         acquire_deep_slot=acquire_deep_slot, **core_kwargs)
        return
    try:
        step = _guided_step(query, event_queue, request_id, session_id=session_id,
                            store=confirmation_store, guided_request=guided_request)
    except CancelledError as exc:
        event_queue.put({"type": "error", "code": exc.code, "message": str(exc)})
        return
    except Exception:
        logger.exception("Guided step error [%s]", request_id)
        if guided_request:
            confirmation_store.finish(guided_request["confirmationId"], request_id, completed=False)
        event_queue.put({"type": "error", "code": "internal_error", "message": "内部エラーが発生しました"})
        return
    if step is None:
        return
    effective_query, fixed_document, on_finish = step
    guided_queue = _GuidedQueue(event_queue, on_finish) if on_finish else event_queue
    try:
        _run_search_core(effective_query, reasoning, guided_queue, cancel_token, request_id,
                         fixed_document=fixed_document, acquire_deep_slot=acquire_deep_slot,
                         **core_kwargs)
    finally:
        if on_finish and not guided_queue.finished:
            guided_queue.finished = True
            on_finish(False)


def _guided_step(query, event_queue, request_id, *, session_id, store, guided_request):
    """確認段階を処理する。回答へ進む場合は ``(問い, 固定資料, 完了時の処理)`` を返す。

    確認結果を返した場合と、エラーで完了した場合は None を返す。
    """
    import guided_follow_up as gf
    from answer_policy import GUIDED_SCOPE_OVERVIEW
    from search import load_source_chunks_cached

    def emit_error(code, message):
        event_queue.put({"type": "error", "code": code, "message": message})

    if guided_request is None:
        event_queue.put({"type": "status", "text": "資料と質問の範囲を確認しています..."})
        chunks, _ = load_source_chunks_cached(SKILL_SOURCE_DIR)
        decision = gf.decide(query, chunks, Path(SKILL_SOURCE_DIR), initial=True)
        if decision.kind == "direct":
            return query, None, None
        if decision.document is not None:
            decision.document["indexMarker"] = _index_marker()
        record = store.create(session_id, query, decision)
        _emit_clarification(event_queue, request_id, store.view(record.confirmation_id, session_id), decision.kind)
        return None

    cid = guided_request["confirmationId"]
    action = guided_request["action"]
    snapshot = guided_request["record"]  # store.begin が返した写し（web_server が登録前に取得）

    def finish(**kwargs):
        return store.finish(cid, request_id, **kwargs)

    def fail(code, message, *, discard=False):
        finish(completed=False)
        if discard:
            store.discard(cid)
        emit_error(code, message)

    if snapshot.document is not None and not _guided_source_ok(snapshot.document):
        return fail("source_changed", "資料または参考資料フォルダーが変更されました。もう一度質問してください。", discard=True)

    if action in {gf.ACTION_SELECT_DOCUMENT, gf.ACTION_HINT}:
        chunks, _ = load_source_chunks_cached(SKILL_SOURCE_DIR)
        if action == gf.ACTION_SELECT_DOCUMENT:
            choice = guided_request.get("choice")
            if not isinstance(choice, int) or not 0 <= choice < len(snapshot.candidates):
                return fail("confirmation_invalid", "選択肢が無効です")
            card = snapshot.candidates[choice]
            documents = {doc["path"]: doc for doc in gf.documents_from_chunks(chunks)}
            current = documents.get(card["path"])
            if current is None or str(current.get("file_sha256") or "") != card.get("fileSha256"):
                return fail("source_changed", "候補の資料が変更されました。もう一度質問してください。", discard=True)
            fixed = gf._fixed_document(Path(SKILL_SOURCE_DIR), current)
            if fixed is None:
                return fail("source_changed", "候補の資料を読み出せません。もう一度質問してください。", discard=True)
            decision = gf.route_for_document(
                snapshot.initial_query, fixed, gf.find_related({**fixed, "head_text": current.get("head_text")}, chunks)
            )
        else:
            combined = f"{snapshot.initial_query} {guided_request['text']}"
            decision = gf.decide(combined, chunks, Path(SKILL_SOURCE_DIR), initial=False)
            if decision.kind == "direct":
                decision = gf.Decision("hints", message=(
                    "まだ資料を確定できません。別の手掛かり（発出年・発出者・文書番号など）を入力してください。"
                ))
        if decision.document is not None:
            decision.document["indexMarker"] = _index_marker()
        if decision.kind in {"documents", "hints", "topics"}:
            public = finish(decision=decision, completed=True)
            if public is None:
                emit_error("confirmation_expired", "確認の有効期限が切れました。もう一度質問してください。")
                return None
            _emit_clarification(event_queue, request_id, public, decision.kind)
            return None
        # 具体的な質問・全体概要の指定は固定資料で回答する。
        document = decision.document
        fixed_document = _fixed_for_answer(document, primary=decision.scope == GUIDED_SCOPE_OVERVIEW)
        return (
            snapshot.initial_query,
            fixed_document,
            lambda ok: finish(decision=decision, completed=ok),
        )

    # 論点選択・追質問（資料は固定済み）
    document = _with_document_text(snapshot.document)
    if document is None:
        return fail("source_changed", "資料が変更されました。もう一度質問してください。", discard=True)
    try:
        plan = gf.resolve_follow_up(
            document, snapshot.topic, snapshot.options,
            action=action, choice=guided_request.get("choice"),
            text=guided_request.get("text") or "",
            drop_targets=tuple(guided_request.get("dropTargets") or ()),
        )
    except gf.ConfirmationError as exc:
        return fail(exc.code, str(exc))
    if plan.ask is not None:
        # 聞き返しでは直前に確定した論点を保持し、候補を新しい選択肢にする。
        public = finish(completed=True, options=plan.ask["options"] or None)
        if public is None:
            emit_error("confirmation_expired", "確認の有効期限が切れました。もう一度質問してください。")
            return None
        _emit_clarification(event_queue, request_id, public, "ask", ask=plan.ask)
        return None
    if plan.related is not None:
        # 関連資料は選んだ時点のハッシュで読めることを確かめ、固定資料と合わせて対象にする。
        from guided_follow_up import read_document

        if any(read_document(Path(SKILL_SOURCE_DIR), m["path"], m["sha256"]) is None for m in plan.related["paths"]):
            return fail("source_changed", "関連資料が変更されました。もう一度質問してください。")
        combined = {**document, "paths": _document_paths(document) + [
            m for m in plan.related["paths"] if m["path"] not in {d["path"] for d in _document_paths(document)}
        ]}
        fixed_document = _fixed_for_answer(combined, primary=True)
        return plan.query, fixed_document, lambda ok: finish(
            topic=plan.topic, add_paths=plan.related["paths"], completed=ok,
        )
    fixed_document = _fixed_for_answer(
        document, focus_ranges=[list(r) for r in plan.focus_ranges], primary=plan.overview,
    )
    if action == gf.ACTION_SELECT_TOPIC and plan.focus_ranges:
        fixed_document["focus_only"] = True
    return plan.query, fixed_document, lambda ok: finish(topic=plan.topic, completed=ok)
