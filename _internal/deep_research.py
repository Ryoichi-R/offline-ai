"""ローカル資料を節単位で読み直す deep 調査モード。

通常の bounded retrieval と契約を混ぜないため、このモジュールは候補の
取得・節範囲・調査台帳・最終統合を所有する。本文は診断 receipt へ出さず、
利用者向けの結果を確定するまでメモリ上だけで扱う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
from pathlib import Path
from typing import Callable, Any

from config import OLLAMA_HOST
from deep_transport import DeepWorkerError, run_worker
from deep_candidate_selection import (
    CANDIDATE_TRANSFER_LIMIT_CODE,
    CANDIDATE_TRANSFER_LIMIT_MESSAGE,
    hydrate_selected,
    validate_many_response,
    validate_response,
)
from prompt_templates import (
    DEEP_FINAL_SYSTEM,
    DEEP_SECTION_SYSTEM,
    DEEP_VERIFY_SYSTEM,
    DEEP_LINE_VERIFY_SYSTEM,
    DEEP_GAPS_SYSTEM,
    build_deep_final_prompt,
    build_deep_section_prompt,
)
from source_structure import parse_heading_tree, section_range_for_line, node_for_line


DEEP_SCHEMA_VERSION = 1
DEEP_TIMEOUT_MIN = 300
DEEP_TIMEOUT_MAX = 1800
DEEP_TIMEOUT_DEFAULT = 1800
DEEP_NUM_CTX = 32768
DEEP_TEXT_NUM_PREDICT = 4096
# Verification JSON can contain one diagnostic entry per claim. Keep its
# output budget separate from the final answer budget.
DEEP_JSON_NUM_PREDICT = 8192
DEEP_MAX_ROUNDS = 3
DEEP_MAX_DOCUMENTS = 5
DEEP_MAX_UNITS = 20
DEEP_MAX_UNIT_CHARS = 4000
DEEP_MAX_EVIDENCE_CHARS = 12000
DEEP_MAX_CANDIDATES_PER_FILE = 8
DEEP_MAX_CANDIDATES_TOTAL = 40
DEEP_INTEGRATION_RESERVE_SECONDS = 120
DEEP_FINALIZATION_RESERVE_SECONDS = 5
DEEP_MAX_VIEWPOINTS = 5
# Documents are read in order of their summed reciprocal rank over every
# viewpoint of a round (same constant as the search-side RRF merge), so the
# document limit is spent on the documents most viewpoints agree on rather
# than on whichever documents happen to appear first in the first viewpoint.
DEEP_DOCUMENT_RRF_K = 60
# A citation-rule rewrite of the final answer is attempted only if at least
# this much time remains before the finalization reserve.
DEEP_FINAL_RETRY_MIN_SECONDS = 30
# Per-line support checks stop while this much time is still left (beyond the
# finalization reserve) for the whole-answer check that follows them.
DEEP_LINE_CHECK_RESERVE_SECONDS = 60
# Ledger reasons that report what was done to the final answer; they are shown
# to the user but do not explain why exploration stopped.
DEEP_FINAL_ANSWER_NOTE_REASONS = frozenset({"limitation_repaired", "enumeration_repaired", "line_unsupported"})
# Reasons that can fire once per candidate/unit and dominate a large corpus run
# (hundreds of thousands of characters observed in practice). These are
# collapsed to a count in the final-answer prompt; individual entries stay in
# result.unconfirmed / ledger for the UI and diagnostics, unabridged.
DEEP_BULK_UNCONFIRMED_REASONS = frozenset(
    {"candidate_limit", "document_limit", "unit_limit", "viewpoint_limit", "context_budget"}
)
DEEP_UNCONFIRMED_PROMPT_CHAR_LIMIT = 2000

STOP_REASONS = frozenset(
    {
        "scope_processed",
        "no_new_evidence",
        "time_budget",
        "round_limit",
        "document_limit",
        "unit_limit",
        "evidence_budget",
        "source_changed",
        CANDIDATE_TRANSFER_LIMIT_CODE,
        "model_error",
        "unresolved_scope",
        "cancelled",
    }
)

STOP_REASON_MESSAGES = {
    CANDIDATE_TRANSFER_LIMIT_CODE: CANDIDATE_TRANSFER_LIMIT_MESSAGE,
}


def stop_reason_message(reason: str) -> str:
    """Return a stable Japanese explanation for a machine stop reason."""
    return STOP_REASON_MESSAGES.get(str(reason), "")


_STOP_REASON_PRIORITY = (
    "cancelled",
    "time_budget",
    "source_changed",
    CANDIDATE_TRANSFER_LIMIT_CODE,
    "evidence_budget",
    "unit_limit",
    "document_limit",
    "round_limit",
    "model_error",
)


def _select_stop_reason(initial: str, reasons: set[str]) -> str:
    """Keep transfer failure visible instead of hiding it behind a later limit."""
    observed = set(reasons)
    if initial not in {"scope_processed", "cancelled"}:
        observed.add(initial)
    for reason in _STOP_REASON_PRIORITY:
        if reason in observed:
            return reason
    if reasons and initial == "scope_processed":
        return "unresolved_scope"
    return initial


class DeepExplorationFinished(RuntimeError):
    """Exploration reservation reached; integration may still proceed."""


class DeepBudgetExpired(RuntimeError):
    """深掘りの全体期限を超えた。"""


class _FinalAnswerRejected(ValueError):
    """A returned final answer failed a check; `detail` names which one for the ledger."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass
class DeepBudget:
    timeout_seconds: float
    cancel_check: Callable[[], None] | None = None
    started_at: float = field(default_factory=time.monotonic)
    rounds: int = 0
    documents: set[str] = field(default_factory=set)
    units: int = 0
    evidence_chars: int = 0
    absolute_deadline: float | None = None
    # Monotonic stage-time accounting. time_budget alone does not prove where
    # the elapsed time went (2026-09-23 investigation, finding 5): the
    # reported wall-clock delta mixes call timeouts, stalls, and URLErrors.
    # This keeps a per-stage breakdown independent of any single call's
    # timeout, without depending on wall-clock timestamps that a Markdown
    # save can disagree with.
    stage_seconds: dict[str, float] = field(default_factory=dict)
    _current_stage: str = "planning"
    _stage_started_at: float = field(default_factory=time.monotonic)

    @property
    def deadline(self) -> float:
        own = self.started_at + self.timeout_seconds
        return min(own, self.absolute_deadline) if self.absolute_deadline is not None else own

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def check(self) -> None:
        if self.cancel_check is not None:
            self.cancel_check()
        if self.remaining() <= 0:
            raise DeepBudgetExpired("deep research time budget expired")

    def can_start_work(self, *, reserve: float = 0.0) -> bool:
        return self.remaining() > max(0.0, reserve)

    def enter_stage(self, stage: str) -> None:
        """Charge elapsed time to the stage being left, then switch to `stage`."""
        now = time.monotonic()
        self.stage_seconds[self._current_stage] = self.stage_seconds.get(
            self._current_stage, 0.0
        ) + (now - self._stage_started_at)
        self._current_stage = stage
        self._stage_started_at = now

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at


# 根拠との原文一致(verification_status)とは別に、"回答文そのもの"がどう
# 確定したかを表す。web_services側の固定confidence=1.0表示(回答未生成調査
# 記録の指摘4)を置き換えるための状態で、原文一致・関連性の判定とは混ぜない。
ANSWER_STATE_GENERATED = "generated"
ANSWER_STATE_VERIFICATION_FAILED = "verification_failed"
ANSWER_STATE_NOT_GENERATED = "not_generated"


@dataclass
class DeepResearchResult:
    status: str
    stop_reason: str
    answer: str = ""
    answer_state: str = ANSWER_STATE_NOT_GENERATED
    evidence: list[dict] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)
    progress: list[dict] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_public_dict(self) -> dict:
        """本文は利用者向け、diagnostics/ledgerは許可項目だけに限定する。"""
        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "answer": self.answer,
            "answer_state": self.answer_state,
            "evidence": [
                {
                    key: item.get(key)
                    for key in (
                        "evidence_id",
                        "path",
                        "source_sha256",
                        "heading",
                        "start_line",
                        "end_line",
                        "char_start",
                        "char_end",
                        "excerpt",
                        "subject",
                        "scope",
                        "conditions",
                        "exceptions",
                        "references",
                        "verification_status",
                        "context",
                    )
                    if key in item
                }
                for item in self.evidence
            ],
            "unconfirmed": list(self.unconfirmed),
            "ledger": list(self.ledger),
            "diagnostics": dict(self.diagnostics),
        }


def validate_deep_timeout_seconds(value: object) -> int:
    """deep の時間予算を整数 [300, 1800] として検証する。"""
    if isinstance(value, bool):
        raise ValueError("deep timeout must be an integer")
    if isinstance(value, str):
        if not re.fullmatch(r"[+-]?\d+", value.strip()):
            raise ValueError("deep timeout must be an integer")
        value = int(value.strip())
    if not isinstance(value, int):
        raise ValueError("deep timeout must be an integer")
    if value < DEEP_TIMEOUT_MIN or value > DEEP_TIMEOUT_MAX:
        raise ValueError(
            f"deep timeout must be between {DEEP_TIMEOUT_MIN} and {DEEP_TIMEOUT_MAX} seconds"
        )
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _candidate_key(candidate: dict) -> tuple[str, int | None, int | None]:
    return (
        str(candidate.get("path") or ""),
        candidate.get("start_line"),
        candidate.get("end_line"),
    )


def _initial_viewpoints(query: str) -> list[str]:
    """閉じた正解一覧にせず、制度資料で抜けやすい軸を追加する。"""
    terms = [part for part in re.split(r"[\s、。・/\\]+", query.strip()) if part]
    candidates = [
        query,
        f"{query} 条件",
        f"{query} 例外 除外",
        f"{query} 適用範囲 期限 時期",
        f"{query} 参照 規定",
    ]
    if terms:
        candidates.append(" ".join(terms[:4] + ["条件", "例外"]))
    return list(dict.fromkeys(candidates))[:DEEP_MAX_VIEWPOINTS]


def _record_progress(
    budget: DeepBudget,
    progress: list[dict],
    emit_progress: Callable[[dict], None] | None,
    *,
    stage: str,
    message: str = "",
    path: str = "",
    heading: str = "",
) -> None:
    budget.enter_stage(stage)
    event = {
        "type": "deep_progress",
        "stage": stage,
        "round": budget.rounds,
        "documents": len(budget.documents),
        "units": budget.units,
        "evidenceChars": budget.evidence_chars,
        "remainingSeconds": int(budget.remaining()),
    }
    if message:
        event["message"] = message
    if path:
        event["path"] = path
    if heading:
        event["heading"] = heading
    progress.append(event)
    del progress[:-64]
    if emit_progress is not None:
        emit_progress(event)


def _split_line_range(
    lines: list[str],
    start_line: int,
    end_line: int,
    *,
    max_chars: int = DEEP_MAX_UNIT_CHARS,
) -> list[dict]:
    """連続範囲を行・段落境界で分割し、長い1行は文字範囲を保持する。"""
    if start_line > end_line or not lines:
        return []
    units: list[dict] = []
    current: list[str] = []
    current_start = start_line
    current_chars = 0

    def flush(end: int, *, char_start: int | None = None, char_end: int | None = None):
        nonlocal current, current_start, current_chars
        if not current:
            return
        units.append(
            {
                "start_line": current_start,
                "end_line": end,
                "char_start": char_start,
                "char_end": char_end,
                "text": "\n".join(current),
            }
        )
        current = []
        current_chars = 0

    for line_no in range(start_line, min(end_line, len(lines)) + 1):
        line = lines[line_no - 1]
        if len(line) > max_chars:
            flush(line_no - 1)
            for offset in range(0, len(line), max_chars):
                part = line[offset : offset + max_chars]
                units.append(
                    {
                        "start_line": line_no,
                        "end_line": line_no,
                        "char_start": offset,
                        "char_end": offset + len(part) - 1,
                        "text": part,
                    }
                )
            current_start = line_no + 1
            continue
        add_chars = len(line) + (1 if current else 0)
        if current and current_chars + add_chars > max_chars:
            flush(line_no - 1)
            current_start = line_no
            add_chars = len(line)
        current.append(line)
        current_chars += add_chars
    flush(min(end_line, len(lines)))
    return units


def _section_range(text: str, hit_start: int, hit_end: int) -> tuple[int, int, str]:
    """ヒットを最も内側のATX節へ割り当て、親導入本文を失わない。"""
    lines = text.splitlines()
    nodes = parse_heading_tree(text)
    if not nodes:
        return 1, len(lines), ""
    start, end, heading = section_range_for_line(
        nodes,
        hit_start,
        include_heading=True,
        preserve_parent_intro=True,
    )
    return start, end, heading


def _source_file_map(source_root: Path, source_files: list[tuple[Path, str]]) -> dict[str, Path]:
    return {rel.replace("\\", "/"): path for path, rel in source_files}


def _load_source_texts(source_root: Path, source_files: list[tuple[Path, str]]):
    texts: dict[str, tuple[Path, bytes, str, list[str]]] = {}
    for path, rel in source_files:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        normalized = rel.replace("\\", "/")
        texts[normalized] = (
            path,
            data,
            _sha256_bytes(data),
            _safe_text(data).splitlines(),
        )
    return texts


def _trim_strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError("invalid extraction list")
    return [item.strip() for item in value]


def _validate_section_json(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("section extraction is not an object")
    if not all(
        key in data for key in ("subject", "scope", "conditions", "exceptions", "references")
    ):
        raise ValueError("missing extraction fields")
    if not isinstance(data["subject"], str) or not isinstance(data["scope"], str):
        raise ValueError("invalid extraction fields")
    relevance = data.get("relevance", "relevant")
    if relevance not in {"relevant", "irrelevant", "uncertain"}:
        raise ValueError("invalid relevance")
    return {
        "subject": data["subject"].strip(),
        "scope": data["scope"].strip(),
        "relevance": relevance,
        "conditions": _trim_strings(data.get("conditions")),
        "exceptions": _trim_strings(data.get("exceptions")),
        "references": _trim_strings(data.get("references")),
    }


def _json_from_response(data: dict) -> object:
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content", "") if isinstance(message, dict) else ""
    content = str(content).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S)
    return json.loads(content)


def _call_ollama_json(
    model: str,
    system: str,
    user: str,
    *,
    timeout: float,
    cancel_check: Callable[[], None] | None = None,
) -> dict:
    content = _call_ollama_content(
        model,
        system,
        user,
        timeout=timeout,
        cancel_check=cancel_check,
        num_predict=DEEP_JSON_NUM_PREDICT,
    )
    parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S))
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _call_ollama_text(
    model: str,
    system: str,
    user: str,
    *,
    timeout: float,
    cancel_check: Callable[[], None] | None = None,
) -> str:
    return _call_ollama_content(
        model,
        system,
        user,
        timeout=timeout,
        cancel_check=cancel_check,
    )


def _evidence_record(
    *,
    evidence_number: int,
    path: str,
    source_sha256: str,
    heading: str,
    unit: dict,
    extracted: dict,
    verification_status: str,
) -> dict:
    record = {
        "evidence_id": f"E{evidence_number}",
        "path": path,
        "source_sha256": source_sha256,
        "heading": heading,
        "start_line": unit["start_line"],
        "end_line": unit["end_line"],
        "excerpt": unit["text"],
        "subject": extracted.get("subject", ""),
        "scope": extracted.get("scope", ""),
        "conditions": extracted.get("conditions", []),
        "exceptions": extracted.get("exceptions", []),
        "references": extracted.get("references", []),
        "verification_status": verification_status,
    }
    if unit.get("char_start") is not None:
        record["char_start"] = unit["char_start"]
        record["char_end"] = unit["char_end"]
    return record


def _exception_stop_reason(exc: BaseException) -> str:
    from search import PromptBudgetError

    if isinstance(exc, PromptBudgetError):
        return "evidence_budget"
    code = str(getattr(exc, "code", "") or "")
    if code == "cancelled":
        return "cancelled"
    if code in {"timeout", "stall"}:
        return "time_budget"
    if code == CANDIDATE_TRANSFER_LIMIT_CODE:
        return CANDIDATE_TRANSFER_LIMIT_CODE
    if isinstance(exc, DeepBudgetExpired):
        return "time_budget"
    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return "time_budget"
    return "model_error"


def _check_prompt(
    system: str, user: str, *, reserve_tokens: int = DEEP_TEXT_NUM_PREDICT
) -> None:
    from search import compute_evidence_char_limit

    compute_evidence_char_limit(
        len(system) + len(user),
        num_ctx=DEEP_NUM_CTX,
        reserve_tokens=reserve_tokens,
        safety_tokens=1024,
        char_limit=DEEP_MAX_EVIDENCE_CHARS,
    )


def detect_deep_model():
    """Read the selected model without migrating or rewriting protected settings."""
    from search import MODEL_CONFIG
    from model_config import DEFAULT_MODEL

    return (
        MODEL_CONFIG.read_text(encoding="utf-8-sig").strip() if MODEL_CONFIG.exists() else ""
    ) or DEFAULT_MODEL


def _call_ollama_content(
    model,
    system,
    user,
    *,
    timeout,
    cancel_check=None,
    num_predict=DEEP_TEXT_NUM_PREDICT,
):
    _check_prompt(system, user, reserve_tokens=num_predict)
    return run_worker(
        {
            "operation": "chat",
            "host": OLLAMA_HOST,
            "timeout": timeout,
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": True,
                "think": False,
                "options": {
                    "num_ctx": DEEP_NUM_CTX,
                    "num_predict": num_predict,
                    "temperature": 0,
                },
            },
        },
        timeout=timeout,
        cancel_check=cancel_check,
    )


def _retrieve_candidates(query, chunks, *, timeout, cancel_check):
    return validate_response(run_worker(
        {"operation": "retrieve", "query": query, "chunks": chunks},
        timeout=timeout,
        cancel_check=cancel_check,
    ))


def _retrieve_candidates_many(queries, chunks, *, timeout, cancel_check):
    """Rank every viewpoint of a round in one worker call."""
    return validate_many_response(
        run_worker(
            {"operation": "retrieve_many", "queries": list(queries), "chunks": chunks},
            timeout=timeout,
            cancel_check=cancel_check,
        ),
        len(queries),
    )


def _retrieve_round(queries, chunks, budget, reserve):
    """One batched worker call per round; per-viewpoint calls only if the batch is too large."""

    def timeout(count):
        return min(60 * count, budget.remaining() - reserve)

    if len(queries) > 1:
        try:
            return _retrieve_candidates_many(
                queries, chunks, timeout=timeout(len(queries)), cancel_check=budget.check
            )
        except DeepWorkerError as exc:
            if exc.code != CANDIDATE_TRANSFER_LIMIT_CODE:
                raise
    responses = []
    for query in queries:
        responses.append(
            _retrieve_candidates(query, chunks, timeout=timeout(1), cancel_check=budget.check)
        )
    return responses


def _order_by_document_relevance(selected_lists, extra_by_path=None):
    """Order candidates document by document, most relevant document first.

    A document's relevance is the sum of 1/(k + rank) over its selected
    candidates in every viewpoint; candidates inside a document follow their
    own summed score. Heading-derived candidates (no rank) follow the ranked
    candidates of their document. Ties keep first-seen order.
    """
    document_scores: dict[str, float] = {}
    candidate_scores: dict[tuple, float] = {}
    candidates: dict[tuple, dict] = {}
    first_seen: dict[tuple, int] = {}
    document_first_seen: dict[str, int] = {}
    for selected in selected_lists:
        for rank, candidate in enumerate(selected, start=1):
            key = _candidate_key(candidate)
            weight = 1.0 / (DEEP_DOCUMENT_RRF_K + rank)
            document_scores[key[0]] = document_scores.get(key[0], 0.0) + weight
            document_first_seen.setdefault(key[0], len(document_first_seen))
            candidate_scores[key] = candidate_scores.get(key, 0.0) + weight
            candidates.setdefault(key, candidate)
            first_seen.setdefault(key, len(first_seen))
    for path in extra_by_path or {}:
        document_scores.setdefault(path, 0.0)
        document_first_seen.setdefault(path, len(document_first_seen))
    by_document: dict[str, list[tuple]] = {}
    for key in candidates:
        by_document.setdefault(key[0], []).append(key)
    ordered: dict[tuple, dict] = {}
    for path in sorted(document_scores, key=lambda p: (-document_scores[p], document_first_seen[p])):
        for key in sorted(
            by_document.get(path, []), key=lambda k: (-candidate_scores[k], first_seen[k])
        ):
            ordered[key] = candidates[key]
        for extra in (extra_by_path or {}).get(path, []):
            ordered.setdefault(_candidate_key(extra), extra)
    return ordered


def _prepare_retrieve_response(response, chunks):
    """Validate the worker contract and restore selected candidates locally."""
    return hydrate_selected(response, chunks)


def _is_call_timeout(exc):
    """A single model/worker call hit its own timeout (not cancel, not the run deadline)."""
    return isinstance(exc, TimeoutError) or getattr(exc, "code", "") in {"timeout", "stall"}


def _raise_control(exc):
    if isinstance(exc, (DeepExplorationFinished, DeepBudgetExpired, TimeoutError)) or getattr(
        exc, "code", ""
    ) in {"cancelled", "timeout", "stall"}:
        raise exc


def _json_stage(model, system, user, budget, validator, *, reserve):
    """One retry for malformed output, charged to the same absolute deadline."""
    _check_prompt(system, user, reserve_tokens=DEEP_JSON_NUM_PREDICT)
    for attempt in range(2):
        budget.check()
        available = budget.remaining() - reserve
        if available <= 1:
            if reserve > DEEP_FINALIZATION_RESERVE_SECONDS:
                raise DeepExplorationFinished("reserved time")
            raise DeepBudgetExpired("reserved time")
        try:
            result = _call_ollama_json(
                model,
                system,
                user,
                timeout=min(60, available),
                cancel_check=budget.check,
            )
            budget.check()
            return validator(result)
        except (ValueError, TypeError, KeyError):
            if attempt:
                raise ValueError("invalid model schema") from None


def _verification(data):
    if not isinstance(data, dict) or type(data.get("supported")) is not bool:
        raise ValueError("invalid support verdict")
    for key in ("contradictions", "missing_conditions", "unsupported_claims"):
        _trim_strings(data.get(key))
    return data["supported"] and not any(
        data[key] for key in ("contradictions", "missing_conditions", "unsupported_claims")
    )


LINE_SUPPORTED = "supported"
LINE_UNSUPPORTED = "unsupported"
LINE_OFF_QUESTION = "off_question"


def _same_actor(source_actor, claim_actor):
    """Names the model read from the source and from the line; unnamed on either side is no conflict."""
    source_name = _normalize_for_match(source_actor).replace("の", "")
    claim_name = _normalize_for_match(claim_actor).replace("の", "")
    if not source_name or not claim_name:
        return True
    return source_name in claim_name or claim_name in source_name


def _line_verdict(data):
    """Per-line verdict from separate readings rather than one yes/no.

    One yes/no let a line that restated the setting-up notice's background as
    a condition pass (2026-09-24 run 1). The model classifies the source
    sentence and names who acts in the source and in the line; the program
    decides from those.
    """
    if not isinstance(data, dict) or type(data.get("supported")) is not bool:
        raise ValueError("invalid support verdict")
    kind = data.get("source_kind")
    if kind not in {"answer", "background"}:
        # No source sentence found: the model leaves the kind empty.
        if not data["supported"] and not kind:
            return LINE_UNSUPPORTED
        raise ValueError("invalid source kind")
    actors = [data.get(key, "") for key in ("source_actor", "claim_actor")]
    if not all(isinstance(actor, str) for actor in actors):
        raise ValueError("invalid actor")
    if kind == "background":
        return LINE_OFF_QUESTION
    if not data["supported"] or not _same_actor(*actors):
        return LINE_UNSUPPORTED
    return LINE_SUPPORTED


def _gaps(data):
    return {key: _trim_strings(data.get(key)) for key in ("queries", "unresolved")}


def _ranges_with_context(text, start, end):
    """Prioritize the hit's own section; parent introductions follow as lower-priority context.

    The hit section must reach extraction/verification before budget runs out,
    so it is processed first. Parent introductions remain separate, real
    ranges (not merged into one oversized prompt) but are queued after it.
    """
    nodes = parse_heading_tree(text)
    node = node_for_line(nodes, start)
    parent_ranges = []
    while node is not None and node.parent_index is not None:
        parent = nodes[node.parent_index]
        stop = nodes[parent.children_indices[0]].heading_line - 1
        if stop >= parent.heading_line:
            parent_ranges.append((parent.heading_line, stop, parent.heading_text))
        node = parent
    if nodes and nodes[0].heading_line > 1:
        parent_ranges.append((1, nodes[0].heading_line - 1, ""))
    parent_ranges.reverse()
    ranges = [_section_range(text, start, end), *parent_ranges]
    return list(dict.fromkeys(ranges))


def _snapshot(root, budget, note):
    from search import iter_source_files, _chunk_source_bytes

    sources, chunks = {}, []
    for path, rel in iter_source_files(root):
        budget.check()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            note("source_missing", rel)
            continue
        try:
            # Incremental reads permit deadline/cancellation checks between blocks.
            parts = []
            with resolved.open("rb") as stream:
                while True:
                    budget.check()
                    part = stream.read(65536)
                    if not part:
                        break
                    parts.append(part)
            data = b"".join(parts)
        except OSError:
            note("source_missing", rel)
            continue
        digest = _sha256_bytes(data)
        text = _safe_text(data)
        sources[rel] = (resolved, digest, text, text.splitlines())
        budget.check()
        chunks.extend(_chunk_source_bytes(data, resolved, rel))
    return sources, chunks


def _source_unchanged(source, budget):
    path, expected, _, _ = source
    try:
        if path.resolve() != path:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                budget.check()
                part = stream.read(65536)
                if not part:
                    break
                digest.update(part)
        return digest.hexdigest() == expected
    except OSError:
        return False


_TABLE_RULE_RE = re.compile(r"^[\s:|-]*-[\s:|-]*$")


def _dequote(stripped: str) -> str:
    while stripped.startswith(">"):
        stripped = stripped[1:].strip()
    return stripped


def _is_heading_like(stripped: str) -> bool:
    """ATX headings and bold-only pseudo-headings label a section; they assert nothing."""
    if stripped.startswith("#"):
        return True
    return stripped.startswith("**") and stripped.endswith("**") and stripped.count("**") == 2


_CITATION_GROUP_RE = re.compile(
    r"[\[［【]\s*([EＥ][0-9０-９]+(?:\s*[,，、・/／\s]\s*[EＥ][0-9０-９]+)*)\s*[\]］】]"
)
_UNCONFIRMED_HEADING_RE = re.compile(r"未確認(?:事項|の事項|事項等)?(?:[（(][^）)]{0,20}[）)])?")


def _normalize_citations(answer):
    """Rewrite citation spelling variants to [E1][E2]; IDs themselves are checked later.

    `[E8, E6]`, `[E8・E6]`, `【E8】` and full-width forms carry the same IDs as
    `[E8][E6]`; normalizing them only changes notation, never which evidence
    a sentence cites.
    """
    count = 0

    def expand(match):
        nonlocal count
        ids = re.findall(r"[EＥ]([0-9０-９]+)", match.group(1))
        rewritten = "".join(f"[E{int(unicodedata.normalize('NFKC', number))}]" for number in ids)
        if rewritten != match.group(0):
            count += 1
        return rewritten

    return _CITATION_GROUP_RE.sub(expand, answer), count


def _is_unconfirmed_heading(stripped):
    """A heading (ATX, bold, or a bare label line) whose text is just "未確認事項"."""
    if re.search(r"\[E\d+\]", stripped):
        return False
    label = stripped.lstrip("#").strip().rstrip(":：").strip().strip("*").strip().rstrip(":：").strip()
    return _UNCONFIRMED_HEADING_RE.fullmatch(label) is not None


def _strip_unconfirmed_section(answer):
    """Drop a model-written "未確認事項" section; the program lists unconfirmed items itself.

    Its lines carry no citations by nature, so keeping them would always fail
    the citation check (2026-09-23 real-corpus run). The section ends at the
    next heading that is not itself about unconfirmed items.
    """
    kept, dropped, skipping = [], 0, False
    for line in answer.splitlines():
        stripped = _dequote(line.strip())
        if _is_unconfirmed_heading(stripped):
            skipping = True
            dropped += 1
            continue
        if skipping and stripped and _is_heading_like(stripped):
            skipping = False
        if skipping:
            if stripped:
                dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept).strip(), dropped


_PAREN_STOP_MASK = ""


def _mask_parenthesized_stops(text):
    """Hide 。！？ inside (full- or half-width) parentheses from sentence splitting."""
    depth, out = 0, []
    for char in text:
        if char in "（(":
            depth += 1
        elif char in "）)" and depth:
            depth -= 1
        out.append(_PAREN_STOP_MASK if depth and char in "。！？" else char)
    return "".join(out)


def _final_answer_issues(answer, evidence_ids):
    """Line numbers (1-based) that fail the citation rules; semantic support is checked separately."""
    issues = {"uncited_lines": [], "unknown_id_lines": [], "has_citation": False}
    if not answer.strip():
        return issues
    lines = answer.splitlines()

    def check(index, cited):
        if not cited:
            issues["uncited_lines"].append(index + 1)
            return False
        if not cited <= evidence_ids:
            issues["unknown_id_lines"].append(index + 1)
            return False
        return True

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _is_heading_like(_dequote(stripped)) or _TABLE_RULE_RE.fullmatch(stripped):
            continue
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        if is_table_row:
            next_stripped = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if _TABLE_RULE_RE.fullmatch(next_stripped):
                continue  # header row: column labels, not a claim of their own
        # A table row or a blockquote aside states one fact per row/line, so a
        # single trailing citation covers it; plain prose still needs a
        # citation after every sentence so an uncited claim cannot ride along
        # next to a cited one.
        if is_table_row or stripped.startswith(">"):
            check(index, set(re.findall(r"\[(E\d+)\]", stripped)))
            continue
        # A citation may follow sentence punctuation; an uncited trailing sentence cannot.
        # Punctuation inside parentheses ("（…を含む。）") does not end a sentence.
        claims = re.findall(
            r"[^。！？\n]+[。！？]?(?:\s*\[E\d+\])*", _mask_parenthesized_stops(stripped)
        )
        for claim in claims:
            if claim.strip() and not check(index, set(re.findall(r"\[(E\d+)\]", claim))):
                break
    issues["has_citation"] = bool(re.search(r"\[E\d+\]", answer))
    return issues


def _final_answer_passes(issues):
    return issues["has_citation"] and not issues["uncited_lines"] and not issues["unknown_id_lines"]


def _validate_final_answer(answer, evidence_ids):
    """Every non-heading paragraph must carry real citations; semantic check is separate."""
    return _final_answer_passes(_final_answer_issues(answer, evidence_ids))


# Parenthetical limits that narrow or widen a condition ("…を除く", "…を含む",
# "…のほか、…の案件") are found by `_limiting_spans`, nesting included.
# Aliases such as "（秘密随意契約）" carry no limiting word.
# A lead sentence that limits which earlier items its list applies to.
_SCOPE_LEAD_RE = re.compile(
    r"(?:前項|前条|同項|第[0-9]+項)?(?:第[0-9]+号(?:(?:又は|及び|若しくは|並びに|、|・)第[0-9]+号)*)?の?規定に(?:関わらず|かかわらず)"
)
# A bullet (optionally numbered) or a kanji/circled item number. A bare
# arabic number ("2 前項…") is a paragraph number, not a list item.
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*・]\s*(?:(?:[一二三四五六七八九十]+|[0-9]+|[①-⑳])[\s　.．、)）])?"
    r"|(?:[一二三四五六七八九十]+|[①-⑳])[\s　.．、)）])"
)
_BOILERPLATE_RE = re.compile(r"(?:この項において|次号において|同項において|以下)同じ")
_NORMALIZE_DROP_RE = re.compile(r"[\s、。，．,.・「」『』:：;；\[\]［］]+")
DEEP_LIMITATION_COVERAGE = 0.6
# An answer line also restates a source line when at least this share of its
# (at least this many) character bigrams occur in that source line.
DEEP_MATCH_BIGRAM_SHARE = 0.5
DEEP_MATCH_MIN_BIGRAMS = 12


def _normalize_for_match(text):
    text = unicodedata.normalize("NFKC", re.sub(r"\[E\d+\]", "", text))
    return _NORMALIZE_DROP_RE.sub("", text)


def _bigram_coverage(haystack, needle):
    grams = {needle[i : i + 2] for i in range(len(needle) - 1)}
    if not grams:
        return 1.0 if needle in haystack else 0.0
    present = {haystack[i : i + 2] for i in range(len(haystack) - 1)}
    return len(grams & present) / len(grams)


_LIMITING_WORD_RE = re.compile(r"除く|除き|含む|含み|限る|限り|ほか|以外|のみ")
# Limits written outside parentheses: "必要に応じ" (so not always), "なるべく", or a
# clause ending in "著しく困難" / "支障がないと認め". The clause from the preceding
# comma up to the word must survive. 2026-09-24: all three runs of the 予定価格
# question dropped such wording with the parenthesis-only check reporting none.
_QUALIFIER_RE = re.compile(
    r"必要に応じ|なるべく|原則として|できる限り|やむを得ない|著しく困難|支障がないと認め"
    r"|(?:場合|とき|もの)に限り|を除き"
)
DEEP_QUALIFIER_MAX_CHARS = 40
# One source line enumerating items inline ("…とする。 ⑴ … ⑵ …").
_INLINE_ITEM_RE = re.compile(r"[⑴-⒇]")
_INLINE_ITEM_MARK_RE = re.compile(r"^\s*\(\d+\)\s*")
# A heading at least this long, or with a full stop, is body text a
# resegmentation put on a heading line (n-42 記１ and 記２).
DEEP_HEADING_BODY_MIN_CHARS = 40
# Enumerations the answer must complete once it uses one of their items.
DEEP_ENUMERATION_MAX_ITEMS = 10
DEEP_ENUMERATION_COVERAGE = 0.8


def _paren_spans(text):
    """Balanced full- or half-width parenthesis spans as (start, end), end inclusive."""
    spans, stack = [], []
    for index, char in enumerate(text):
        if char in "（(":
            stack.append(index)
        elif char in "）)" and stack:
            spans.append((stack.pop(), index))
    return spans


def _limiting_spans(text):
    """Outermost parentheses whose own level (nested parentheses removed) carries a limiting word.

    第1条's limit sits outside a nested definition: "（…事業（以下「基金事業」という。）等
    であって、…選定を厚生労働省で行う場合を含む。以下同じ。）". Only matching innermost
    parentheses missed it (2026-09-23, second and third of three runs).
    """
    spans = _paren_spans(text)

    def own_level(start, end):
        chars = list(text[start + 1 : end])
        for inner_start, inner_end in spans:
            if start < inner_start and inner_end < end:
                for position in range(inner_start - start - 1, inner_end - start):
                    chars[position] = ""
        return "".join(chars)

    limiting = [
        (start, end, own_level(start, end))
        for start, end in spans
        if _LIMITING_WORD_RE.search(own_level(start, end))
    ]
    return [
        (start, end, own)
        for start, end, own in limiting
        if not any(other_start < start and end < other_end for other_start, other_end, _ in limiting)
    ]


def _qualifier_limits(body, spans):
    """Limits outside parentheses: the clause from the preceding comma up to each qualifier."""
    limits = []
    for match in _QUALIFIER_RE.finditer(body):
        if any(start <= match.start() <= end for start, end, _ in spans):
            continue
        start = max(body.rfind("、", 0, match.start()), body.rfind("。", 0, match.start())) + 1
        clause = body[max(start, match.end() - DEEP_QUALIFIER_MAX_CHARS) : match.end()]
        norm = _normalize_for_match(clause)
        if norm:
            limits.append((norm, clause))
    return limits


def _source_segments(display):
    """A source line as (text, is_inline_item) segments, split at inline ⑴⑵ items."""
    marks = [match.start() for match in _INLINE_ITEM_RE.finditer(display)]
    if len(marks) < 2:
        return [(display, False)]
    lead = display[: marks[0]].strip()
    bounds = [*marks, len(display)]
    return [*([(lead, False)] if lead else []), *((display[a:b].strip(), True) for a, b in zip(bounds, bounds[1:]))]


def _without_spans(text, spans):
    chars = list(text)
    for start, end, _ in spans:
        for position in range(start, end + 1):
            chars[position] = ""
    return "".join(chars)


def _best_source_row(sentence, cited, rows):
    """The cited source row an answer line restates most closely (body or limiting parenthesis)."""
    best, strength = None, 0
    for row in rows:
        if row["evidence_id"] not in cited:
            continue
        value = max(_match_strength(sentence, text) for text in row["match_texts"])
        if value > strength:
            best, strength = row, value
    return best


def _is_verbatim_source_line(line, rows):
    sentence = _normalize_for_match(line)
    return any(sentence == _normalize_for_match(row["raw"]) for row in rows)


def _check_lines_support(candidate, evidence, *, query, model, budget, record):
    """Check each cited answer line alone against only the evidence it cites.

    The whole-answer check passed a line that overgeneralized one clause
    (2026-09-23). A line judged unsupported is replaced by the source line it
    restates, or dropped if none matches; a line that restates background the
    question does not ask about is dropped. Lines that already are verbatim
    source text are not re-checked. Checks stop, leaving the rest to the
    whole-answer check, when the time kept for that check is reached.
    """
    rows = _source_limits(evidence)
    by_id = {e["evidence_id"]: e for e in evidence}
    counts = {"checked": 0, "unsupported": 0, "replaced": 0, "dropped": 0, "off_question": 0, "unchecked": 0}
    record["line_checks"] = counts
    lines = candidate.splitlines()
    kept, notes, stopped = [], [], False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        cited = [i for i in dict.fromkeys(re.findall(r"\[(E\d+)\]", stripped)) if i in by_id]
        if not cited or _is_heading_like(_dequote(stripped)) or _is_verbatim_source_line(stripped, rows):
            kept.append(line)
            continue
        if stopped:
            counts["unchecked"] += 1
            kept.append(line)
            continue
        try:
            verdict = _json_stage(
                model,
                DEEP_LINE_VERIFY_SYSTEM,
                json.dumps(
                    {"question": query, "source": [by_id[i] for i in cited], "claims": stripped},
                    ensure_ascii=False,
                ),
                budget,
                _line_verdict,
                reserve=DEEP_FINALIZATION_RESERVE_SECONDS + DEEP_LINE_CHECK_RESERVE_SECONDS,
            )
        except DeepExplorationFinished:
            stopped = True
            counts["unchecked"] += 1
            kept.append(line)
            continue
        except Exception as exc:
            if not _is_call_timeout(exc):
                _raise_control(exc)
                if not isinstance(exc, ValueError):
                    raise
                # Malformed verdict twice: undecided; the whole-answer check still applies.
                counts["unchecked"] += 1
                kept.append(line)
                continue
            stopped = True
            counts["unchecked"] += 1
            kept.append(line)
            continue
        counts["checked"] += 1
        if verdict == LINE_SUPPORTED:
            kept.append(line)
            continue
        counts["unsupported"] += 1
        if verdict == LINE_OFF_QUESTION:
            # Restoring the source line would keep the background in the answer.
            counts["off_question"] += 1
            counts["dropped"] += 1
            notes.append(f"最終回答の{number}行目は質問が問う事項ではない背景の記述と照合されたため、回答から除いた")
            continue
        row = _best_source_row(_normalize_for_match(stripped), set(cited), rows)
        if row is not None:
            counts["replaced"] += 1
            kept.append(_cite_verbatim(row["raw"], row["evidence_id"]))
            notes.append(f"最終回答の{number}行目は根拠に支持されないと照合されたため、原文の文言に置き換えた")
        else:
            counts["dropped"] += 1
            notes.append(f"最終回答の{number}行目は根拠に支持されないと照合されたため、回答から除いた")
    return "\n".join(kept), notes


def _source_limits(evidence):
    """Per source line of each evidence excerpt: its core text, the limits it carries, and its list.

    Items of one list (consecutive bullets or numbered lines, or ⑴⑵ inline on one
    line) share a `group` number so that an answer using one item can be held to the rest.
    """
    rows = []
    group_count = 0
    for item in evidence:
        lead_scope = None
        group = None
        for raw in str(item.get("excerpt") or "").splitlines():
            display = raw.lstrip("> ").strip()
            if display.startswith("#"):
                display = display.lstrip("#").strip()
                if len(display) < DEEP_HEADING_BODY_MIN_CHARS and "。" not in display:
                    group = None
                    continue
            if not display or display.startswith("|"):
                continue
            for segment, inline_item in _source_segments(display):
                line = unicodedata.normalize("NFKC", segment)
                is_item = inline_item or bool(_LIST_ITEM_RE.match(line))
                if not is_item:
                    scope = _SCOPE_LEAD_RE.search(line)
                    lead_scope = (scope.group(0), line, segment) if scope else None
                    group = None
                elif group is None:
                    group_count += 1
                    group = group_count
                if inline_item:
                    body = _INLINE_ITEM_MARK_RE.sub("", line, count=1)
                elif is_item:
                    body = _LIST_ITEM_RE.sub("", line, count=1)
                else:
                    body = line
                spans = _limiting_spans(body)
                limits = [
                    (_normalize_for_match(_BOILERPLATE_RE.sub("", own)), body[start : end + 1])
                    for start, end, own in spans
                ]
                limits = [(norm, original) for norm, original in limits if norm]
                limits += _qualifier_limits(body, spans)
                core = _normalize_for_match(_without_spans(body, spans))
                # Lines without limits stay as match targets so an answer line
                # restating them is not attributed to a neighbouring limited line.
                if core:
                    rows.append(
                        {
                            "evidence_id": item.get("evidence_id"),
                            "core": core,
                            # An answer line may restate the parenthetical itself
                            # (e.g. the definition of 基金事業), so it is a match target too.
                            "match_texts": [core, *(norm for norm, _ in limits)],
                            "limits": limits,
                            "scope": lead_scope if is_item else None,
                            "group": group if is_item else None,
                            # Source text as written, minus a bullet, for verbatim repair.
                            "raw": re.sub(r"^[-*・]\s*", "", segment),
                        }
                    )
    return rows


def _bigrams(text):
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _match_strength(sentence, core):
    """How closely an answer line restates a source line (0 = not a restatement).

    Full containment or a long common run counts. So does a paraphrase whose
    bigrams mostly come from the source line: a 92-character 第1条 was restated
    with only a 24-character common run, below the run threshold (2026-09-23).
    """
    if core in sentence:
        return len(core)
    match = difflib.SequenceMatcher(None, sentence, core, autojunk=False).find_longest_match(
        0, len(sentence), 0, len(core)
    )
    by_run = match.size if match.size >= max(12, 0.4 * len(core)) else 0
    sentence_grams = _bigrams(sentence)
    shared = len(sentence_grams & _bigrams(core))
    by_share = (
        shared
        if len(sentence_grams) >= DEEP_MATCH_MIN_BIGRAMS
        and shared / len(sentence_grams) >= DEEP_MATCH_BIGRAM_SHARE
        else 0
    )
    return max(by_run, by_share)


def _limitation_findings(answer, evidence):
    """Like `_limitation_issues`, with the matched source line kept for repair."""
    rows = _source_limits(evidence)
    if not any(row["limits"] or row["scope"] for row in rows):
        return []
    answer_norm = _normalize_for_match(answer)
    findings, scopes_reported = [], set()
    for index, line in enumerate(answer.splitlines()):
        stripped = line.strip()
        cited = set(re.findall(r"\[(E\d+)\]", stripped))
        if not cited or _is_heading_like(_dequote(stripped)):
            continue
        sentence = _normalize_for_match(stripped)
        best = _best_source_row(sentence, cited, rows)
        if best is None:
            continue
        limits_missing = [
            original
            for norm, original in best["limits"]
            if _bigram_coverage(sentence, norm) < DEEP_LIMITATION_COVERAGE
        ]
        scope_missing = None
        if best["scope"] is not None:
            phrase, lead, _ = best["scope"]
            if (
                _bigram_coverage(answer_norm, _normalize_for_match(phrase)) < DEEP_LIMITATION_COVERAGE
                and lead not in scopes_reported
            ):
                scopes_reported.add(lead)
                scope_missing = phrase
        if limits_missing or scope_missing:
            findings.append(
                {
                    "line": index + 1,
                    "row": best,
                    "limits_missing": limits_missing,
                    "scope_missing": scope_missing,
                }
            )
    return findings


def _limitation_issues(answer, evidence):
    """Answer lines that restate a cited source line but drop its limits.

    Each line is matched to the cited source line it restates most closely;
    the parenthetical limits of that line must survive (bigram coverage), and
    a list item under a scope lead ("前項第1号又は第2号の規定に関わらず") needs
    that scope somewhere in the answer. The same-model support check passed an
    answer that dropped such a limit (2026-09-23 real-corpus run).
    """
    return [
        (f["line"], [*f["limits_missing"], *([f["scope_missing"]] if f["scope_missing"] else [])])
        for f in _limitation_findings(answer, evidence)
    ]


def _enumeration_findings(answer, evidence):
    """Lists the answer uses only in part: the items it leaves out, and where to add them.

    An answer line restating one item of a cited list (2 to 10 items) makes the
    whole list required. 2026-09-24: answers to the 予定価格 question gave only
    ⑴ of 記１ or of 記２, which the checks did not look at.
    """
    rows = _source_limits(evidence)
    groups: dict[int, list[dict]] = {}
    for row in rows:
        if row["group"] is not None:
            groups.setdefault(row["group"], []).append(row)
    if not groups:
        return []
    covered, last_line = set(), {}
    for index, line in enumerate(answer.splitlines()):
        stripped = line.strip()
        cited = set(re.findall(r"\[(E\d+)\]", stripped))
        if not cited or _is_heading_like(_dequote(stripped)):
            continue
        best = _best_source_row(_normalize_for_match(stripped), cited, rows)
        if best is not None and best["group"] is not None:
            covered.add(id(best))
            last_line[best["group"]] = index + 1
    answer_norm = _normalize_for_match(answer)
    findings = []
    for group, members in groups.items():
        if group not in last_line or not 2 <= len(members) <= DEEP_ENUMERATION_MAX_ITEMS:
            continue
        missing = [
            row
            for row in members
            if id(row) not in covered and _bigram_coverage(answer_norm, row["core"]) < DEEP_ENUMERATION_COVERAGE
        ]
        if missing:
            findings.append({"after_line": last_line[group], "evidence_id": members[0]["evidence_id"], "missing": missing})
    return findings


def _enumeration_issues(answer, evidence):
    return [(f["evidence_id"], [row["raw"] for row in f["missing"]]) for f in _enumeration_findings(answer, evidence)]


def _repair_enumerations(answer, findings):
    """Add the left-out items verbatim after the last answer line that uses their list."""
    lines = answer.splitlines()
    inserts: dict[int, list[str]] = {}
    notes = []
    for finding in findings:
        inserts.setdefault(finding["after_line"] - 1, []).extend(
            _cite_verbatim(row["raw"], row["evidence_id"]) for row in finding["missing"]
        )
        notes.append(
            f"最終回答に根拠{finding['evidence_id']}の列挙の項目（"
            + "／".join(_shorten(row["raw"]) for row in finding["missing"])
            + "）が欠けていたため、原文の項目を補った"
        )
    repaired = []
    for index, line in enumerate(lines):
        repaired.append(line)
        repaired.extend(inserts.get(index, []))
    return "\n".join(repaired), notes


def _drop_duplicate_lines(answer):
    """Drop a line that repeats an earlier one (citations aside); returns the answer and the count.

    Verbatim repair can turn two paraphrases of one source line into the same
    line: an answer to the 予定価格 question gave 類型⑴⑵ twice (2026-09-24).
    """
    seen, kept, dropped = set(), [], 0
    for line in answer.splitlines():
        key = _normalize_for_match(line)
        if key and key in seen:
            dropped += 1
            continue
        if key:
            seen.add(key)
        kept.append(line)
    return "\n".join(kept), dropped


def _cite_verbatim(text, evidence_id):
    """Source text with its evidence ID after every sentence, so it passes the citation rules."""
    masked = _mask_parenthesized_stops(text)
    parts, start = [], 0
    for match in re.finditer(r"[。！？]", masked):
        parts.append(f"{text[start:match.end()]}[{evidence_id}]")
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(f"{tail}[{evidence_id}]")
    return "".join(parts)


def _repair_limitations(answer, findings):
    """Replace lines that dropped a limit with the source line itself, and insert a missing scope lead.

    Used only after the one rewrite still drops limits. The inserted text is
    the cited source verbatim, so no model-authored claim loses its limits and
    one missing limit no longer discards the whole answer. Returns the repaired
    answer and user-facing notes naming the restored source text.
    """
    lines = answer.splitlines()
    inserts: dict[int, list[str]] = {}
    notes = []
    for finding in findings:
        index, row = finding["line"] - 1, finding["row"]
        if finding["limits_missing"]:
            lines[index] = _cite_verbatim(row["raw"], row["evidence_id"])
            notes.append(
                f"最終回答の{finding['line']}行目は原文の限定（"
                + "／".join(_shorten(text) for text in finding["limits_missing"])
                + "）が欠けていたため、原文の文言に置き換えた"
            )
        if finding["scope_missing"]:
            inserts.setdefault(index, []).append(_cite_verbatim(row["scope"][2], row["evidence_id"]))
            notes.append(
                f"最終回答の{finding['line']}行目の前に、欠けていた原文の適用範囲（{_shorten(finding['scope_missing'])}）を補った"
            )
    repaired = []
    for index, line in enumerate(lines):
        repaired.extend(inserts.get(index, []))
        repaired.append(line)
    return "\n".join(repaired), notes


def _shorten(text, limit=60):
    """User-facing label for a source limit: outer parentheses dropped, length capped.

    A long limit is cut to the clause carrying its limiting word ("…場合を含む"),
    which is the part the reader needs, rather than to its opening words.
    """
    text = str(text).strip("()（）")
    if len(text) <= limit:
        return text
    matches = list(_LIMITING_WORD_RE.finditer(text))
    if matches:
        end = matches[-1].end()
        start = max(text.rfind("、", 0, matches[-1].start()) + 1, end - (limit - 1))
        return "…" + text[start:end]
    return text[: limit - 1] + "…"


def _check_final_candidate(candidate, evidence, record):
    """Prepare one final-answer attempt, check citations and limits, and record counts only."""
    prepared, changes = _prepare_final_answer(candidate)
    issues = _final_answer_issues(prepared, {e["evidence_id"] for e in evidence})
    issues["limitation_lines"] = _limitation_issues(prepared, evidence)
    issues["enumeration_gaps"] = _enumeration_issues(prepared, evidence)
    record["attempts"].append(
        {
            **changes,
            "uncited_lines": len(issues["uncited_lines"]),
            "unknown_id_lines": len(issues["unknown_id_lines"]),
            "limitation_lines": len(issues["limitation_lines"]),
            "enumeration_gaps": sum(len(texts) for _, texts in issues["enumeration_gaps"]),
        }
    )
    return prepared, issues


def _final_retry_prompt(prompt, candidate, issues):
    problems = []
    if issues["uncited_lines"]:
        problems.append("根拠IDがない行: " + "、".join(f"{n}行目" for n in issues["uncited_lines"][:20]))
    if issues["unknown_id_lines"]:
        problems.append("提示していない根拠IDを使った行: " + "、".join(f"{n}行目" for n in issues["unknown_id_lines"][:20]))
    if not issues["has_citation"]:
        problems.append("根拠IDが1つもありません")
    for line_no, missing in issues.get("limitation_lines", [])[:20]:
        problems.append(f"{line_no}行目で根拠の限定が抜けています（原文: {'／'.join(missing)}）")
    for evidence_id, missing in issues.get("enumeration_gaps", [])[:20]:
        problems.append(f"根拠{evidence_id}の列挙のうち、次の項目が回答にありません（原文: {'／'.join(missing)}）")
    return (
        f"{prompt}\n\n前回の回答案（規則違反あり）:\n{candidate}\n\n"
        f"前回の回答案の問題: {' / '.join(problems)}\n"
        "すべての事実文の文末に、提示済みの根拠IDを [E1] の形式で付けて、回答全体を書き直してください。"
        "根拠IDを付けられない文と、未確認事項の欄は書かないでください。"
    )


def _prepare_final_answer(candidate):
    """Normalize notation and drop a model-written unconfirmed section before checking."""
    normalized, rewritten = _normalize_citations(candidate)
    stripped, dropped = _strip_unconfirmed_section(normalized)
    return stripped, {"normalized_citations": rewritten, "dropped_unconfirmed_lines": dropped}


def _collapse_bulk_unconfirmed(
    unconfirmed: list[str], ledger: list[dict], *, bulk_note: str
) -> list[str]:
    """Count bulk per-candidate reasons; keep every other entry verbatim.

    Bulk reasons (candidate/document/unit limits etc.) fire once per excluded
    candidate and can reach hundreds of thousands of characters on a large
    corpus (2026-09-23 investigation, finding 2). Non-bulk entries (model
    errors, unresolved gaps, etc.) are never collapsed. Counted items remain
    unread/unresolved; nothing here marks them completed.
    """
    counts: dict[str, int] = {}
    for entry in ledger:
        if entry.get("status") == "irrelevant":
            continue
        reason = str(entry.get("reason") or "")
        if reason in DEEP_BULK_UNCONFIRMED_REASONS:
            counts[reason] = counts.get(reason, 0) + 1
    bulk_lines = [
        f"{reason}: {count}件が上限のため未読のまま（{bulk_note}）"
        for reason, count in sorted(counts.items())
    ]
    detail_lines = [
        item
        for item in unconfirmed
        if not any(item.startswith(f"{reason}:") for reason in DEEP_BULK_UNCONFIRMED_REASONS)
    ]
    return [*bulk_lines, *detail_lines]


def _summarize_unconfirmed_for_prompt(
    unconfirmed: list[str],
    ledger: list[dict],
    *,
    max_chars: int = DEEP_UNCONFIRMED_PROMPT_CHAR_LIMIT,
) -> list[str]:
    """Collapse bulk exclusion reasons before they reach the model input.

    Feeding every excluded candidate into the final-answer prompt as free text
    pushes the prompt over budget before evidence is even added. The remaining
    lines are additionally capped at `max_chars`.
    """
    lines = _collapse_bulk_unconfirmed(
        unconfirmed,
        ledger,
        bulk_note="個別詳細は集約済み、診断のledger_countsに件数を保持",
    )
    result: list[str] = []
    total = 0
    for index, line in enumerate(lines):
        total += len(line) + 1
        if total > max_chars:
            result.append(f"...他{len(lines) - index}件は文字数上限のため省略（詳細はresult.unconfirmedを参照）")
            break
        result.append(line)
    return result


def _fallback_answer(evidence, unconfirmed, ledger, stop_reason):
    """Excerpt-only answer. Bulk exclusions are shown as counts, not one line each.

    The full per-candidate list stays in result.unconfirmed / ledger (warning
    list and ledger in the UI); repeating it here made the answer text grow
    past the replay size limit and truncated the excerpts themselves.
    """
    unconfirmed = _collapse_bulk_unconfirmed(
        unconfirmed,
        ledger,
        bulk_note="個別の一覧は未確認事項欄・調査台帳を参照",
    )
    lines = ["## 原文抜粋（未確認の解釈は断定しません）", ""]
    for item in evidence:
        location = f"L{item['start_line']}-{item['end_line']}"
        if item.get("char_start") is not None:
            location += f" 文字位置{item['char_start']}-{item['char_end']}"
        lines.extend(
            [
                f"- [{item['evidence_id']}] `skill-source/{item['path']}` {location}",
                *["> " + line for line in item["excerpt"].splitlines()],
                "",
            ]
        )
    if not evidence:
        lines.append("確認済みの原文を取得できませんでした。")
    if unconfirmed:
        lines.extend(["## 未確認事項", *[f"- {item}" for item in unconfirmed]])
    message = stop_reason_message(stop_reason)
    display_reason = f"{stop_reason}（{message}）" if message else stop_reason
    lines.append(f"調査終了理由: `{display_reason}`")
    return "\n".join(lines)


def run_deep_research(
    query: str,
    *,
    model: str,
    source_root: Path | None = None,
    timeout_seconds: int = DEEP_TIMEOUT_DEFAULT,
    max_rounds: int = DEEP_MAX_ROUNDS,
    max_documents: int = DEEP_MAX_DOCUMENTS,
    max_units: int = DEEP_MAX_UNITS,
    emit_progress=None,
    cancel_check=None,
    absolute_deadline: float | None = None,
) -> DeepResearchResult:
    timeout_seconds = validate_deep_timeout_seconds(timeout_seconds)
    for value in (max_rounds, max_documents, max_units):
        if type(value) is not int or value < 1:
            raise ValueError("deep limits must be positive integers")
    budget = DeepBudget(
        timeout_seconds, cancel_check=cancel_check, absolute_deadline=absolute_deadline
    )
    root = (source_root or Path(__file__).resolve().parent.parent / "skill-source").resolve()
    evidence, progress, ledger, unconfirmed = [], [], [], []
    seen, searched, references, resolved_references = set(), set(), set(), set()
    sources = {}
    retrieval_counts = []
    stop_reason, answer = "scope_processed", ""
    answer_state = ANSWER_STATE_NOT_GENERATED
    # Numbers and fixed codes only: never answer text (diagnostics contract).
    final_answer = {"attempts": []}
    note_keys = set()

    def note(reason, path="", unit=None, detail=""):
        entry = {"reason": reason, "path": path, "status": "unconfirmed"}
        if unit:
            entry.update(
                {
                    k: unit[k]
                    for k in ("start_line", "end_line", "char_start", "char_end")
                    if k in unit
                }
            )
        key = json.dumps(entry, sort_keys=True) + detail
        if key not in note_keys:
            note_keys.add(key)
            ledger.append(entry)
            location = (
                f" L{unit['start_line']}-{unit['end_line']}"
                if unit and "start_line" in unit
                else ""
            )
            unconfirmed.append(f"{reason}: {path}{location}" + (f" — {detail}" if detail else ""))

    reserve = DEEP_INTEGRATION_RESERVE_SECONDS + DEEP_FINALIZATION_RESERVE_SECONDS
    try:
        budget.check()
        _record_progress(
            budget,
            progress,
            emit_progress,
            stage="planning",
            message="資料と調査観点を確認しています",
        )
        try:
            sources, chunks = _snapshot(root, budget, note)
            pending = _initial_viewpoints(query)
            for round_index in range(1, min(max_rounds, DEEP_MAX_ROUNDS) + 1):
                budget.rounds = round_index
                round_queries = list(dict.fromkeys(pending))
                queries = []
                for viewpoint in dict.fromkeys(pending):
                    budget.check()
                    if viewpoint in searched:
                        continue
                    if not budget.can_start_work(reserve=reserve + 1):
                        raise DeepExplorationFinished("reserved time")
                    searched.add(viewpoint)
                    queries.append(viewpoint)
                selected_lists = []
                if queries:
                    _record_progress(
                        budget,
                        progress,
                        emit_progress,
                        stage="search",
                        message=f"候補を検索しています（巡回{round_index}、{len(queries)}観点）",
                    )
                    for retrieve_response in _retrieve_round(queries, chunks, budget, reserve):
                        retrieve_response = _prepare_retrieve_response(
                            retrieve_response, chunks
                        )
                        retrieval_counts.append(dict(retrieve_response["counts"]))
                        for excluded in retrieve_response["excluded"]:
                            note("candidate_limit", excluded["path"], excluded)
                        selected_lists.append(retrieve_response["selected"])
                pending = []
                # Headings in discovered documents supply additional exploration independent of the model.
                headings_by_path = {}
                for path in dict.fromkeys(
                    str(c.get("path") or "") for selected in selected_lists for c in selected
                ):
                    if path not in sources:
                        continue
                    for node in parse_heading_tree(sources[path][2]):
                        if re.search(
                            r"条件|例外|除外|適用|期限|時期|参照|定義|但書|付則",
                            node.heading_text,
                        ):
                            headings_by_path.setdefault(path, []).append(
                                {
                                    "path": path,
                                    "start_line": node.heading_line,
                                    "end_line": node.section_end_line,
                                    "source_sha256": sources[path][1],
                                }
                            )
                batch = _order_by_document_relevance(selected_lists, headings_by_path)
                before = len(evidence)
                for candidate in batch.values():
                    budget.check()
                    path = str(candidate.get("path") or "").replace("\\", "/")
                    source = sources.get(path)
                    if source is None:
                        note("source_missing", path)
                        continue
                    expected = candidate.get("source_sha256") or candidate.get("file_sha256")
                    if (expected and expected != source[1]) or not _source_unchanged(
                        source, budget
                    ):
                        note("source_changed", path)
                        continue
                    if path not in budget.documents and len(budget.documents) >= min(
                        max_documents, DEEP_MAX_DOCUMENTS
                    ):
                        note("document_limit", path, candidate)
                        continue
                    start, end = (
                        int(candidate.get("start_line") or 1),
                        int(candidate.get("end_line") or 1),
                    )
                    for section_start, section_end, heading in _ranges_with_context(
                        source[2], start, end
                    ):
                        units = _split_line_range(source[3], section_start, section_end)
                        # Table headings remain separate source ranges, not synthetic text in a quoted range.
                        headers = [
                            (n, n + 1)
                            for n in range(section_start, section_end)
                            if "|" in source[3][n - 1] and re.fullmatch(r"[\s|:\-]+", source[3][n])
                        ]
                        for unit in units:
                            budget.check()
                            key = (
                                path,
                                source[1],
                                unit["start_line"],
                                unit["end_line"],
                                unit["char_start"],
                                unit["char_end"],
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            if budget.units >= min(max_units, DEEP_MAX_UNITS):
                                note("unit_limit", path, unit)
                                continue
                            if not budget.can_start_work(reserve=reserve + 1):
                                note("time_budget", path, unit)
                                raise DeepExplorationFinished("reserved time")
                            budget.documents.add(path)
                            budget.units += 1
                            _record_progress(
                                budget,
                                progress,
                                emit_progress,
                                stage="read",
                                message="節を読んで原文と照合しています",
                                path=path,
                                heading=heading,
                            )
                            extracted = {
                                "subject": "",
                                "scope": "",
                                "conditions": [],
                                "exceptions": [],
                                "references": [],
                            }
                            verification = "raw_unverified"
                            try:
                                context = []
                                for a, b in headers:
                                    if b < unit["start_line"]:
                                        context.append(
                                            {
                                                "path": path,
                                                "start_line": a,
                                                "end_line": b,
                                                "text": "\n".join(source[3][a - 1 : b]),
                                            }
                                        )
                                user = build_deep_section_prompt(
                                    query,
                                    path,
                                    heading,
                                    unit["start_line"],
                                    unit["end_line"],
                                    unit["text"],
                                )
                                if context:
                                    user += "\n表の補助文脈（別範囲）:\n" + json.dumps(
                                        context, ensure_ascii=False
                                    )
                                if (
                                    len(unit["text"]) + sum(len(c["text"]) for c in context)
                                    > DEEP_MAX_UNIT_CHARS
                                ):
                                    note("context_budget", path, unit)
                                    context = []
                                    raise ValueError("context limit")
                                if not model:
                                    raise ValueError("no model")
                                extracted = _json_stage(
                                    model,
                                    DEEP_SECTION_SYSTEM,
                                    user,
                                    budget,
                                    _validate_section_json,
                                    reserve=reserve,
                                )
                                verify_input = json.dumps(
                                    {
                                        "question": query,
                                        "source": unit["text"],
                                        "context": context,
                                        "claims": extracted,
                                    },
                                    ensure_ascii=False,
                                )
                                supported = _json_stage(
                                    model,
                                    DEEP_VERIFY_SYSTEM,
                                    verify_input,
                                    budget,
                                    _verification,
                                    reserve=reserve,
                                )
                                if not supported or extracted["relevance"] == "uncertain":
                                    raise ValueError("unsupported extraction")
                                if extracted["relevance"] == "irrelevant":
                                    ledger.append(
                                        {
                                            "path": path,
                                            "status": "irrelevant",
                                            "reason": "irrelevant",
                                            "start_line": unit["start_line"],
                                            "end_line": unit["end_line"],
                                        }
                                    )
                                    continue
                                verification = "semantic_checked"
                                references.update(extracted["references"])
                            except Exception as exc:
                                _raise_control(exc)
                                note(_exception_stop_reason(exc), path, unit)
                                extracted = {
                                    "subject": "",
                                    "scope": "",
                                    "conditions": [],
                                    "exceptions": [],
                                    "references": [],
                                }
                            record = _evidence_record(
                                evidence_number=len(evidence) + 1,
                                path=path,
                                source_sha256=source[1],
                                heading=heading,
                                unit=unit,
                                extracted=extracted,
                                verification_status=verification,
                            )
                            if context:
                                record["context"] = context
                            chars = len(json.dumps(record, ensure_ascii=False))
                            if budget.evidence_chars + chars > DEEP_MAX_EVIDENCE_CHARS:
                                note("evidence_budget", path, unit)
                                continue
                            budget.evidence_chars += chars
                            evidence.append(record)
                # References are resolved only by actually retained source text, never by a search hit alone.
                for reference in references:
                    if any(
                        reference in e["excerpt"] or reference == e["heading"]
                        for e in evidence
                        if reference not in e["references"]
                        and e["verification_status"] == "semantic_checked"
                    ):
                        resolved_references.add(reference)
                pending.extend(sorted(references - resolved_references - searched))
                if evidence and len(evidence) > before and model:
                    try:
                        data = _json_stage(
                            model,
                            DEEP_GAPS_SYSTEM,
                            json.dumps(
                                {"question": query, "evidence": evidence},
                                ensure_ascii=False,
                            ),
                            budget,
                            _gaps,
                            reserve=reserve,
                        )
                        for issue in data["unresolved"]:
                            note("unresolved", detail=issue)
                        pending.extend(q for q in data["queries"] if q not in searched)
                    except Exception as exc:
                        _raise_control(exc)
                        note("model_error", detail="追加観点の確認に失敗")
                pending = list(dict.fromkeys(pending))
                if round_index > 1 and len(evidence) == before:
                    for viewpoint in round_queries:
                        note("unresolved_query", detail=viewpoint)
                if len(pending) > DEEP_MAX_VIEWPOINTS:
                    for viewpoint in pending[DEEP_MAX_VIEWPOINTS:]:
                        note("viewpoint_limit", detail=viewpoint)
                    pending = pending[:DEEP_MAX_VIEWPOINTS]
                if not pending:
                    break
                if round_index == min(max_rounds, DEEP_MAX_ROUNDS):
                    note("round_limit", detail="残る追加検索観点")
            for reference in references - resolved_references:
                note("unresolved_reference", detail=reference)
        except DeepExplorationFinished:
            note("time_budget", detail="統合・終了の予約時間に到達")
        except Exception as exc:
            # Exploration calls get "remaining minus the integration reserve"
            # as their timeout, so a call that runs out that clock ends
            # exploration exactly at the reserve boundary. That is the planned
            # end of exploration, not a failure of the whole run: continue to
            # integration with whatever was verified (2026-09-23 real-corpus
            # acceptance: 175s of 300s, verified evidence, zero final calls).
            # Cancellation and the absolute deadline still propagate.
            if not _is_call_timeout(exc):
                raise
            note("time_budget", detail="モデル呼出が時間内に終わらず探索を終了")

        def revalidate():
            invalid = {
                e["path"] for e in evidence if not _source_unchanged(sources[e["path"]], budget)
            }
            for path in invalid:
                note("source_changed", path)
            evidence[:] = [e for e in evidence if e["path"] not in invalid]
            return bool(invalid)

        revalidate()
        verified = [e for e in evidence if e["verification_status"] == "semantic_checked"]
        if verified:
            # Distinguishes "the model never returned an answer" (timeout,
            # connection error, prompt budget) from "an answer came back but
            # failed citation/support checks".
            candidate = None
            try:
                budget.check()
                available = budget.remaining() - DEEP_FINALIZATION_RESERVE_SECONDS
                if available <= 2:
                    raise DeepBudgetExpired("reserved time")
                _record_progress(
                    budget,
                    progress,
                    emit_progress,
                    stage="integrate",
                    message="回答案と原文の意味を照合しています",
                )
                # Include parent scope source text even if its structured extraction failed.
                # Unverified records contain no model-authored claims.
                prompt = build_deep_final_prompt(
                    query,
                    evidence,
                    _summarize_unconfirmed_for_prompt(unconfirmed, ledger),
                    "scope_processed",
                )
                _check_prompt(DEEP_FINAL_SYSTEM, prompt)
                candidate = _call_ollama_text(
                    model,
                    DEEP_FINAL_SYSTEM,
                    prompt,
                    timeout=min(60, available / 2),
                    cancel_check=budget.check,
                )
                budget.check()
                candidate, issues = _check_final_candidate(candidate, evidence, final_answer)
                retry_available = budget.remaining() - DEEP_FINALIZATION_RESERVE_SECONDS
                needs_rewrite = (
                    not _final_answer_passes(issues) or issues["limitation_lines"] or issues["enumeration_gaps"]
                )
                if needs_rewrite and retry_available >= DEEP_FINAL_RETRY_MIN_SECONDS:
                    # One rewrite with the failing line numbers, inside the same deadline.
                    retry_prompt = _final_retry_prompt(prompt, candidate, issues)
                    try:
                        _check_prompt(DEEP_FINAL_SYSTEM, retry_prompt)
                        rewritten = _call_ollama_text(
                            model,
                            DEEP_FINAL_SYSTEM,
                            retry_prompt,
                            timeout=min(60, retry_available / 2),
                            cancel_check=budget.check,
                        )
                    except Exception as exc:
                        # Cancellation still propagates; a rewrite that times out
                        # or errors leaves the first answer's citation failure.
                        if not _is_call_timeout(exc):
                            _raise_control(exc)
                    else:
                        budget.check()
                        candidate, issues = _check_final_candidate(
                            rewritten, evidence, final_answer
                        )
                if not _final_answer_passes(issues):
                    final_answer["result"] = "rejected_citations"
                    raise _FinalAnswerRejected("最終回答の根拠IDが不足または不正のため原文抜粋を表示")
                if issues["limitation_lines"]:
                    # Still dropping limits after the rewrite: put the cited source
                    # text back verbatim on those lines instead of discarding the answer.
                    findings = _limitation_findings(candidate, evidence)
                    repaired, repair_notes = _repair_limitations(candidate, findings)
                    if not _final_answer_passes(
                        _final_answer_issues(repaired, {e["evidence_id"] for e in evidence})
                    ) or _limitation_issues(repaired, evidence):
                        final_answer["result"] = "rejected_limitations"
                        missing = [text for _, texts in issues["limitation_lines"] for text in texts]
                        raise _FinalAnswerRejected(
                            "最終回答で根拠の限定（括弧書き・限定の文言・適用範囲）が欠けたため原文抜粋を表示（欠けた原文の限定: "
                            + "／".join(_shorten(text) for text in missing[:3])
                            + (f" ほか{len(missing) - 3}件" if len(missing) > 3 else "")
                            + "）"
                        )
                    candidate = repaired
                    final_answer["repaired_lines"] = len(findings)
                    for text in repair_notes:
                        note("limitation_repaired", detail=text)
                enumeration = _enumeration_findings(candidate, evidence)
                if enumeration:
                    # Still leaving out items of a list it uses: add them verbatim.
                    candidate, enumeration_notes = _repair_enumerations(candidate, enumeration)
                    final_answer["added_items"] = sum(len(f["missing"]) for f in enumeration)
                    for text in enumeration_notes:
                        note("enumeration_repaired", detail=text)
                candidate, duplicates = _drop_duplicate_lines(candidate)
                # Line-by-line support check with only each line's cited evidence,
                # using time the run would otherwise leave unused.
                candidate, line_notes = _check_lines_support(
                    candidate,
                    evidence,
                    query=query,
                    model=model,
                    budget=budget,
                    record=final_answer,
                )
                for text in line_notes:
                    note("line_unsupported", detail=text)
                # Replacing unsupported lines with source text can repeat a line too.
                candidate, more_duplicates = _drop_duplicate_lines(candidate)
                if duplicates + more_duplicates:
                    final_answer["duplicate_lines"] = duplicates + more_duplicates
                if not _final_answer_passes(
                    _final_answer_issues(candidate, {e["evidence_id"] for e in evidence})
                ):
                    final_answer["result"] = "rejected_support"
                    raise _FinalAnswerRejected(
                        "最終回答の各行が根拠に支持されないと照合されたため原文抜粋を表示"
                    )
                check_input = json.dumps(
                    {"question": query, "source": evidence, "claims": candidate},
                    ensure_ascii=False,
                )
                if not _json_stage(
                    model,
                    DEEP_VERIFY_SYSTEM,
                    check_input,
                    budget,
                    _verification,
                    reserve=DEEP_FINALIZATION_RESERVE_SECONDS,
                ):
                    final_answer["result"] = "rejected_support"
                    raise _FinalAnswerRejected("最終回答が根拠に支持されないと照合されたため原文抜粋を表示")
                answer = candidate
                answer_state = ANSWER_STATE_GENERATED
                final_answer["result"] = "accepted"
            except Exception as exc:
                _raise_control(exc)
                if candidate is None:
                    note(
                        _exception_stop_reason(exc),
                        detail="最終回答を生成できず原文抜粋を表示",
                    )
                    answer_state = ANSWER_STATE_NOT_GENERATED
                    final_answer["result"] = "not_generated"
                else:
                    final_answer.setdefault("result", "verification_incomplete")
                    note(
                        _exception_stop_reason(exc),
                        detail=exc.detail
                        if isinstance(exc, _FinalAnswerRejected)
                        else "最終回答の照合を完了できず原文抜粋を表示",
                    )
                    answer_state = ANSWER_STATE_VERIFICATION_FAILED
        if revalidate():
            answer = ""
            answer_state = ANSWER_STATE_VERIFICATION_FAILED
        budget.check()
    except Exception as exc:
        stop_reason = _exception_stop_reason(exc)
        note(stop_reason)
        # Never display an answer whose final validation was interrupted.
        answer = ""
        answer_state = ANSWER_STATE_NOT_GENERATED

    reasons = {
        e["reason"]
        for e in ledger
        if e["status"] != "irrelevant" and e["reason"] not in DEEP_FINAL_ANSWER_NOTE_REASONS
    }
    stop_reason = _select_stop_reason(stop_reason, reasons)
    status = (
        "cancelled"
        if stop_reason == "cancelled"
        else "partial"
        if evidence and (unconfirmed or not answer)
        else "completed"
        if evidence
        else "failed"
    )
    if not answer:
        answer = _fallback_answer(evidence, unconfirmed, ledger, stop_reason)
    # Entering "complete" charges the last working stage (usually integrate,
    # the longest one) to stage_seconds, so it must precede the snapshot below.
    _record_progress(
        budget,
        progress,
        emit_progress,
        stage="complete",
        message=f"調査状態: {status} / {stop_reason}",
    )
    diagnostics = {
        "schema_version": DEEP_SCHEMA_VERSION,
        "timeout_seconds": timeout_seconds,
        # Monotonic total elapsed time, independent of any single call's
        # timeout and of wall-clock UTC timestamps recorded elsewhere.
        "elapsed_seconds": round(budget.elapsed_seconds(), 3),
        "stage_seconds": {k: round(v, 3) for k, v in budget.stage_seconds.items()},
        "reserved_seconds": {
            "integration": DEEP_INTEGRATION_RESERVE_SECONDS,
            "finalization": DEEP_FINALIZATION_RESERVE_SECONDS,
        },
        "rounds": budget.rounds,
        "documents": len(budget.documents),
        "units": budget.units,
        "evidence_count": len(evidence),
        "evidence_chars": budget.evidence_chars,
        "retrieval_counts": retrieval_counts,
        "final_answer": final_answer,
        # Units the model judged unrelated to the question: read, but not used.
        # Paths and line ranges only (bounded by the unit limit), so a reader can
        # tell "not read" from "read and judged irrelevant".
        "irrelevant_ranges": [
            {"path": e["path"], "start_line": e["start_line"], "end_line": e["end_line"]}
            for e in ledger
            if e.get("status") == "irrelevant"
        ][:DEEP_MAX_UNITS],
        "ledger_counts": {
            r: sum(e["reason"] == r for e in ledger) for r in sorted({e["reason"] for e in ledger})
        },
    }
    return DeepResearchResult(
        status,
        stop_reason,
        answer,
        answer_state,
        evidence,
        unconfirmed,
        progress,
        ledger,
        diagnostics,
    )
