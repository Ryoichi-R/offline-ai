"""Authenticated in-memory conversations and portable saved-answer records."""

from __future__ import annotations

import base64
import binascii
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

from assistant_tools import SOURCE_ID, TOOL_ARGUMENTS, Source, encode, verify_source
from source_view import EvidenceViewError

TTL_SECONDS = 1800
MAX_CONVERSATIONS = 16
MAX_CONVERSATION_BYTES = 8 * 1024 * 1024
MAX_ALL_BYTES = 64 * 1024 * 1024
TURN_RESERVE = 2 * 1024 * 1024
MAX_SAVED_BYTES = 5 * 1024 * 1024
# A batch can exceed the eight-tool execution budget. Retain rejected requests too.
MAX_TOOL_TRACE = 104
BASE_BYTES = 1024
ID = re.compile(r"[A-Za-z0-9_-]{22,64}\Z")
REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
SAVED_MARKER = re.compile(r"\n<!-- offline-ai-assistant-v1: ([A-Za-z0-9_=-]+) -->\s*\Z")
STATUSES = {
    "completed",
    "timeout",
    "cancelled",
    "model_error",
    "output_limit",
    "context_limit",
}
ANNOTATIONS = {
    "outside_sources": "資料外の補足",
    "citation_not_explicit": "出典の明示なし",
    "number_unattributable": "照合対象を特定できない",
    "number_spelling_absent": "出典範囲に同表記なし",
    "unknown_citation": "出典不明",
    "prior_sources_only": "今回の資料取得なし・過去の出典を参照",
    "no_source_reference": "資料を参照していない回答",
}
VERIFICATION = {
    "matched": "取得時と同じ原文範囲",
    "changed": "資料変更で失効",
    "missing": "資料が見つからない",
    "unavailable": "原文を開けない",
    "excerpt_mismatch": "保存した抜粋と原文が一致しない",
    "not_rechecked": "取得後の再照合未実施",
}


class ConversationError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass
class Conversation:
    owner: str
    generation: str
    touched: float
    key: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    records: list[bytes] = field(default_factory=list)
    requests: dict[str, tuple[int, str]] = field(default_factory=dict)
    active: tuple[str, int, str] | None = None
    size: int = BASE_BYTES


def failure_result(turn, question, status="model_error"):
    return {
        "turn": turn,
        "question": question,
        "answer": "",
        "status": status,
        "sources": [],
        "annotations": [],
        "trace": [],
        "history_omitted": 0,
    }


class Conversations:
    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._items = {}

    def _purge(self):
        now = self._clock()
        for id_, conv in list(self._items.items()):
            if conv.active is None and now - conv.touched >= TTL_SECONDS:
                del self._items[id_]

    def _get(self, id_, owner, generation):
        self._purge()
        conv = self._items.get(id_)
        if not conv or conv.owner != owner or conv.generation != generation:
            raise ConversationError(
                "conversation_not_found",
                "会話が失効しました。新しい会話を開始してください。",
            )
        conv.touched = self._clock()
        return conv

    def _capacity(self, extra, *, new=False, protected=None):
        self._purge()
        size = sum(
            c.size + (TURN_RESERVE if c.active else 0) for c in self._items.values()
        )
        candidates = sorted(
            (c.touched, id_)
            for id_, c in self._items.items()
            if c.active is None and c is not protected
        )
        while (
            size + extra > MAX_ALL_BYTES
            or len(self._items) + int(new) > MAX_CONVERSATIONS
        ):
            if not candidates:
                raise ConversationError("server_busy", "会話の保持容量がいっぱいです。")
            _, id_ = candidates.pop(0)
            size -= self._items.pop(id_).size

    def new(self, owner, generation):
        with self._lock:
            self._capacity(BASE_BYTES, new=True)
            id_ = secrets.token_urlsafe(24)
            self._items[id_] = Conversation(owner, generation, self._clock())
            return id_

    def view(self, id_, owner, generation):
        with self._lock:
            conv = self._get(id_, owner, generation)
            return {
                "conversationId": id_,
                "turns": [json.loads(row) for row in conv.records],
                "active": {"requestId": conv.active[0], "turn": conv.active[1]}
                if conv.active
                else None,
                "nextTurn": len(conv.records) + 1,
            }

    def authorize_resume(self, id_, owner, generation, request_id, turn):
        with self._lock:
            conv = self._get(id_, owner, generation)
            if (conv.active and conv.active[:2] == (request_id, turn)) or (
                request_id in conv.requests and conv.requests[request_id][0] == turn
            ):
                return
            raise ConversationError("job_not_found", "再接続する処理が見つかりません。")

    def start(self, id_, owner, generation, request_id, turn, request_hash, submit):
        with self._lock:
            conv = self._get(id_, owner, generation)
            if conv.active == (request_id, turn, request_hash) or conv.requests.get(
                request_id
            ) == (turn, request_hash):
                return submit(True), None
            if (
                conv.active
                or request_id in conv.requests
                or turn != len(conv.records) + 1
            ):
                raise ConversationError(
                    "request_conflict",
                    "会話で処理中か、送信したターンが既に存在します。",
                )
            if conv.size + TURN_RESERVE > MAX_CONVERSATION_BYTES:
                raise ConversationError(
                    "conversation_full",
                    "会話の保持容量に達しました。新しい会話を開始してください。",
                )
            self._capacity(TURN_RESERVE, protected=conv)
            turns = [json.loads(row) for row in conv.records]
            prior = {}
            for result in turns:
                for item in result["sources"]:
                    source = Source.parse(item)
                    prior.setdefault(source.id, source)
            entry = submit(False)
            conv.active = (request_id, turn, request_hash)
            return entry, (turns, prior, conv.key)

    def finish(self, id_, request_id, outcome):
        """Commit exactly once, before publishing the SSE terminal event."""
        with self._lock:
            conv = self._items.get(id_)
            if not conv or not conv.active or conv.active[0] != request_id:
                return None
            _, turn, request_hash = conv.active
            value = encode(outcome).encode()
            if len(value) + 256 > TURN_RESERVE:
                outcome = {
                    **failure_result(turn, outcome["question"], "output_limit"),
                    "answer": outcome["answer"],
                }
                value = encode(outcome).encode()
            conv.records.append(value)
            conv.requests[request_id] = (turn, request_hash)
            conv.size += len(value) + 256
            conv.touched = self._clock()
            conv.active = None
            return outcome

    def result(self, id_, owner, generation, turn):
        with self._lock:
            conv = self._get(id_, owner, generation)
            if not 1 <= turn <= len(conv.records):
                raise ConversationError(
                    "conversation_not_found", "回答が見つかりません。"
                )
            return json.loads(conv.records[turn - 1])

    def source(self, id_, owner, generation, source_id):
        with self._lock:
            conv = self._get(id_, owner, generation)
            for record in reversed(conv.records):
                for item in json.loads(record)["sources"]:
                    if item["id"] == source_id:
                        return Source.parse(item)
            raise ConversationError("evidence_not_found", "出典が見つかりません。")


def annotation_text(note):
    details = [ANNOTATIONS[note["code"]]]
    if "line" in note:
        details.append(f"回答の{note['line']}行目")
    if "source_id" in note:
        details.append(note["source_id"])
    if "value" in note:
        details.append(note["value"])
    return " / ".join(details)


def render_saved(result):
    portable = dict(result)
    portable["sources"] = [
        {k: v for k, v in item.items() if k != "evidence"} for item in result["sources"]
    ]
    metadata = base64.urlsafe_b64encode(
        encode({"version": 1, "result": portable}).encode()
    ).decode()
    lines = [
        "# 資料参照アシスタント",
        "",
        "## 質問",
        "",
        result["question"],
        "",
        "## 回答",
        "",
        result["answer"],
        "",
        f"終了状態 {result['status']}",
    ]
    if result["history_omitted"]:
        lines.append(
            f"モデル送信時に省略した過去のターン {result['history_omitted']}件"
        )
    lines.extend(("", "## 出典", ""))
    for item in result["sources"]:
        obtained = "原文読取済み" if item["state"] == "read" else "検索抜粋のみ"
        when = "過去ターンで取得" if item["turn"] != result["turn"] else "今回取得"
        lines.extend(
            (
                f"### {item['citation']} {item['title']}",
                "",
                item["path"],
                f"SHA-256 `{item['sha256']}`",
                f"{obtained} / {when} / {VERIFICATION[item['verification']]}",
            )
        )
        for span in item["spans"]:
            lines.extend(
                (
                    "",
                    (
                        f"{span['start_line']}–{span['end_line']}行"
                        f"（開始文字 {span['start_char']}、終了文字 {span['end_char']}）"
                    ),
                    "",
                    *(f"> {line}" for line in span["text"].split("\n")),
                )
            )
        lines.append("")
    lines.extend(
        (
            "",
            "## 照合注記",
            "",
            "同じ表記があっても意味、条件、適用対象の正しさは保証されません。",
            "",
        )
    )
    lines.extend("- " + annotation_text(note) for note in result["annotations"])
    lines.extend(("", "## 検索・読取の経過", ""))
    for entry in result["trace"]:
        if "chat_call" in entry:
            # JSON renders newlines as escapes; the portable copy keeps bounded diagnostics.
            lines.append("- " + encode(entry))
        elif entry["tool"] == "search_sources":
            lines.append(f"- 検索 {entry['query']}（{entry['retrieval']}、{entry['results']}件）")
        elif entry["tool"] == "read_source":
            lines.append(f"- 読取 {entry['path']} [{entry['source_id']}]")
    lines.extend(("", f"<!-- offline-ai-assistant-v1: {metadata} -->", ""))
    return "\n".join(lines)


def import_saved(text, root, registry, owner):
    if not isinstance(text, str) or len(text.encode()) > MAX_SAVED_BYTES:
        raise ValueError("保存回答のサイズが上限を超えています。")
    marker = SAVED_MARKER.search(text)
    if not marker:
        raise ValueError("資料参照アシスタントの保存データが見つかりません。")
    try:
        payload = json.loads(base64.b64decode(marker[1], altchars=b"-_", validate=True))
        if (
            not isinstance(payload, dict)
            or type(payload.get("version")) is not int
            or payload["version"] != 1
        ):
            raise ValueError()
        result = payload["result"]
        if not isinstance(result, dict) or set(result) != set(failure_result(1, "")):
            raise ValueError()
        if (
            type(result["turn"]) is not int
            or not 1 <= result["turn"] <= 1_000_000
            or not isinstance(result["question"], str)
            or not 1 <= len(result["question"]) <= 1000
            or not isinstance(result["answer"], str)
            or len(result["answer"].encode()) > 64 * 1024
            or result["status"] not in STATUSES
            or type(result["history_omitted"]) is not int
            or not 0 <= result["history_omitted"] < result["turn"]
            or not isinstance(result["sources"], list)
            or len(result["sources"]) > 200
            or not isinstance(result["annotations"], list)
            or len(result["annotations"]) > 10_000
            or not isinstance(result["trace"], list)
            or len(result["trace"]) > MAX_TOOL_TRACE
        ):
            raise ValueError()
        for note in result["annotations"]:
            if (
                not isinstance(note, dict)
                or note.get("code") not in ANNOTATIONS
                or set(note) - {"code", "line", "cell", "source_id", "value"}
                or any(
                    type(note[k]) is not int or not 1 <= note[k] <= 65536
                    for k in ("line", "cell")
                    if k in note
                )
                or (
                    "source_id" in note
                    and (
                        not isinstance(note["source_id"], str)
                        or not SOURCE_ID.fullmatch(note["source_id"])
                    )
                )
                or (
                    "value" in note
                    and (not isinstance(note["value"], str) or len(note["value"]) > 65536)
                )
            ):
                raise ValueError()
        for entry in result["trace"]:
            if not isinstance(entry, dict):
                raise TypeError()
            if "chat_call" in entry:
                tool = entry.get("tool")
                args = entry["arguments"]
                if (
                    not isinstance(tool, str)
                    or not 1 <= len(tool) <= 80
                    or type(entry["chat_call"]) is not int
                    or not 1 <= entry["chat_call"] <= 10
                    or type(entry["arguments_omitted"]) is not bool
                    or any(
                        entry.get(k) is not None
                        and (
                            not isinstance(entry[k], str)
                            or not re.fullmatch(r"[a-z0-9_]{1,80}", entry[k])
                        )
                        for k in ("error", "detail")
                    )
                    or (
                        args is not None
                        and (
                            not isinstance(args, dict)
                            or args.keys() - TOOL_ARGUMENTS.get(tool, set())
                            or any(
                                not (
                                    v is None
                                    or type(v) is bool
                                    or type(v) is int
                                    and abs(v) <= 10**9
                                    or isinstance(v, str)
                                    and len(v) <= 1000
                                )
                                for v in args.values()
                            )
                        )
                    )
                ):
                    raise ValueError()
                if entry["error"] or tool == "list_sources":
                    continue
            if entry.get("tool") == "search_sources":
                if (
                    not isinstance(entry.get("query"), str)
                    or len(entry["query"]) > 1000
                    or entry.get("retrieval") not in {"keyword", "hybrid", "unavailable"}
                    or type(entry.get("results")) is not int
                    or not 0 <= entry["results"] <= 12
                ):
                    raise ValueError()
            elif entry.get("tool") == "read_source":
                if not isinstance(entry.get("path"), str) or not isinstance(
                    entry.get("source_id"), str
                ):
                    raise ValueError()
                from source_view import _normalize_relative_path

                if (
                    len(entry["path"]) > 4096
                    or _normalize_relative_path(entry["path"]) != entry["path"]
                    or not SOURCE_ID.fullmatch(entry["source_id"])
                ):
                    raise ValueError()
            else:
                raise ValueError()
        sources = [Source.parse(item) for item in result["sources"]]
        if len({s.id for s in sources}) != len(sources) or any(
            s.turn > result["turn"] for s in sources
        ):
            raise ValueError()
    except (
        KeyError,
        TypeError,
        ValueError,
        RecursionError,
        binascii.Error,
        EvidenceViewError,
    ) as exc:
        raise ValueError("保存回答のデータ形式が不正です。") from exc
    return {
        **result,
        "sources": [
            verify_source(root, source, registry=registry, session_id=owner) for source in sources
        ],
    }
