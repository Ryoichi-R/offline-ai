"""ローカル資料を節単位で読み直す deep 調査モード。

通常の bounded retrieval と契約を混ぜないため、このモジュールは候補の
取得・節範囲・調査台帳・最終統合を所有する。本文は診断 receipt へ出さず、
利用者向けの結果を確定するまでメモリ上だけで扱う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import time
import urllib.error
from pathlib import Path
from typing import Callable, Any

from config import OLLAMA_HOST
from deep_transport import run_worker
from prompt_templates import (
    DEEP_FINAL_SYSTEM,
    DEEP_SECTION_SYSTEM,
    DEEP_VERIFY_SYSTEM,
    DEEP_GAPS_SYSTEM,
    build_deep_final_prompt,
    build_deep_section_prompt,
)
from source_structure import parse_heading_tree, section_range_for_line, node_for_line


DEEP_SCHEMA_VERSION = 1
DEEP_TIMEOUT_MIN = 300
DEEP_TIMEOUT_MAX = 1800
DEEP_TIMEOUT_DEFAULT = 1800
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
        "model_error",
        "unresolved_scope",
        "cancelled",
    }
)


class DeepExplorationFinished(RuntimeError):
    """Exploration reservation reached; integration may still proceed."""


class DeepBudgetExpired(RuntimeError):
    """深掘りの全体期限を超えた。"""


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


@dataclass
class DeepResearchResult:
    status: str
    stop_reason: str
    answer: str = ""
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
    if isinstance(exc, DeepBudgetExpired):
        return "time_budget"
    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return "time_budget"
    return "model_error"


def _check_prompt(system: str, user: str) -> None:
    from search import compute_evidence_char_limit

    compute_evidence_char_limit(
        len(system) + len(user),
        num_ctx=32768,
        reserve_tokens=4096,
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


def _call_ollama_content(model, system, user, *, timeout, cancel_check=None):
    _check_prompt(system, user)
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
                "options": {"num_ctx": 32768, "num_predict": 4096, "temperature": 0},
            },
        },
        timeout=timeout,
        cancel_check=cancel_check,
    )


def _retrieve_candidates(query, chunks, *, timeout, cancel_check):
    return run_worker(
        {"operation": "retrieve", "query": query, "chunks": chunks},
        timeout=timeout,
        cancel_check=cancel_check,
    )


def _raise_control(exc):
    if isinstance(exc, (DeepExplorationFinished, DeepBudgetExpired, TimeoutError)) or getattr(
        exc, "code", ""
    ) in {"cancelled", "timeout", "stall"}:
        raise exc


def _json_stage(model, system, user, budget, validator, *, reserve):
    """One retry for malformed output, charged to the same absolute deadline."""
    _check_prompt(system, user)
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


def _gaps(data):
    return {key: _trim_strings(data.get(key)) for key in ("queries", "unresolved")}


def _ranges_with_context(text, start, end):
    """Separate provenance for parent introductions and the selected inner section."""
    nodes = parse_heading_tree(text)
    node = node_for_line(nodes, start)
    ranges = []
    while node is not None and node.parent_index is not None:
        parent = nodes[node.parent_index]
        stop = nodes[parent.children_indices[0]].heading_line - 1
        if stop >= parent.heading_line:
            ranges.append((parent.heading_line, stop, parent.heading_text))
        node = parent
    if nodes and nodes[0].heading_line > 1:
        ranges.append((1, nodes[0].heading_line - 1, ""))
    ranges.reverse()
    ranges.append(_section_range(text, start, end))
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


def _validate_final_answer(answer, evidence_ids):
    """Every non-heading paragraph must carry real citations; semantic check is separate."""
    if not answer.strip():
        return False
    lines = answer.splitlines()
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
            cited = set(re.findall(r"\[(E\d+)\]", stripped))
            if not cited or not cited <= evidence_ids:
                return False
            continue
        # A citation may follow sentence punctuation; an uncited trailing sentence cannot.
        claims = re.findall(r"[^。！？\n]+[。！？]?(?:\s*\[E\d+\])*", stripped)
        for claim in claims:
            if not claim.strip():
                continue
            cited = set(re.findall(r"\[(E\d+)\]", claim))
            if not cited or not cited <= evidence_ids:
                return False
    return bool(re.search(r"\[E\d+\]", answer))


def _fallback_answer(evidence, unconfirmed, stop_reason):
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
    lines.append(f"調査終了理由: `{stop_reason}`")
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
    stop_reason, answer = "scope_processed", ""
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
                batch = {}
                for viewpoint in dict.fromkeys(pending):
                    budget.check()
                    if viewpoint in searched:
                        continue
                    if not budget.can_start_work(reserve=reserve + 1):
                        raise DeepExplorationFinished("reserved time")
                    searched.add(viewpoint)
                    _record_progress(
                        budget,
                        progress,
                        emit_progress,
                        stage="search",
                        message=f"候補を検索しています（巡回{round_index}）",
                    )
                    found = _retrieve_candidates(
                        viewpoint,
                        chunks,
                        timeout=min(60, budget.remaining() - reserve),
                        cancel_check=budget.check,
                    )
                    counts, accepted = {}, 0
                    for candidate in found:
                        path = str(candidate.get("path") or "").replace("\\", "/")
                        if (
                            counts.get(path, 0) >= DEEP_MAX_CANDIDATES_PER_FILE
                            or accepted >= DEEP_MAX_CANDIDATES_TOTAL
                        ):
                            note("candidate_limit", path, candidate)
                            continue
                        counts[path] = counts.get(path, 0) + 1
                        accepted += 1
                        batch.setdefault(_candidate_key(candidate), candidate)
                pending = []
                # Headings in discovered documents supply additional exploration independent of the model.
                for path in {str(c.get("path") or "") for c in list(batch.values())}:
                    if path not in sources:
                        continue
                    for node in parse_heading_tree(sources[path][2]):
                        if re.search(
                            r"条件|例外|除外|適用|期限|時期|参照|定義|但書|付則",
                            node.heading_text,
                        ):
                            candidate = {
                                "path": path,
                                "start_line": node.heading_line,
                                "end_line": node.section_end_line,
                                "source_sha256": sources[path][1],
                            }
                            batch.setdefault(_candidate_key(candidate), candidate)
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
                prompt = build_deep_final_prompt(query, evidence, unconfirmed, "scope_processed")
                _check_prompt(DEEP_FINAL_SYSTEM, prompt)
                candidate = _call_ollama_text(
                    model,
                    DEEP_FINAL_SYSTEM,
                    prompt,
                    timeout=min(60, available / 2),
                    cancel_check=budget.check,
                )
                budget.check()
                if not _validate_final_answer(candidate, {e["evidence_id"] for e in evidence}):
                    raise ValueError("invalid final citations")
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
                    raise ValueError("unsupported final answer")
                answer = candidate
            except Exception as exc:
                _raise_control(exc)
                note(
                    _exception_stop_reason(exc),
                    detail="最終回答の支持を確認できず原文抜粋を表示",
                )
        if revalidate():
            answer = ""
        budget.check()
    except Exception as exc:
        stop_reason = _exception_stop_reason(exc)
        note(stop_reason)
        # Never display an answer whose final validation was interrupted.
        answer = ""

    reasons = {e["reason"] for e in ledger if e["status"] != "irrelevant"}
    if stop_reason == "cancelled":
        status = "cancelled"
    else:
        for reason in (
            "time_budget",
            "source_changed",
            "evidence_budget",
            "unit_limit",
            "document_limit",
            "round_limit",
            "model_error",
        ):
            if reason in reasons:
                stop_reason = reason
                break
        if reasons and stop_reason == "scope_processed":
            stop_reason = "unresolved_scope"
        status = (
            "partial"
            if evidence and (unconfirmed or not answer)
            else "completed"
            if evidence
            else "failed"
        )
    if not answer:
        answer = _fallback_answer(evidence, unconfirmed, stop_reason)
    diagnostics = {
        "schema_version": DEEP_SCHEMA_VERSION,
        "timeout_seconds": timeout_seconds,
        "rounds": budget.rounds,
        "documents": len(budget.documents),
        "units": budget.units,
        "evidence_count": len(evidence),
        "evidence_chars": budget.evidence_chars,
        "ledger_counts": {
            r: sum(e["reason"] == r for e in ledger) for r in sorted({e["reason"] for e in ledger})
        },
    }
    _record_progress(
        budget,
        progress,
        emit_progress,
        stage="complete",
        message=f"調査状態: {status} / {stop_reason}",
    )
    return DeepResearchResult(
        status,
        stop_reason,
        answer,
        evidence,
        unconfirmed,
        progress,
        ledger,
        diagnostics,
    )
