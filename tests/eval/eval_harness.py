"""offline-ai Phase 2 検索品質・根拠品質の評価ハーネス（core）。

マスタープラン Phase 2 の受入条件を、固定 corpus と固定質問で測定可能にする。

設計上の不変条件:

- 追加の外向き通信を行わない。Embedding/chat は既存の loopback Ollama 経路だけを使う。
- runtime Python 依存を増やさない（標準ライブラリのみ）。
- receipt へ資料本文・query 応答本文・絶対 path を保存しない（`redact_match` を経由する）。
- Ollama 未起動などで実行できない経路は `skipped` とし、PASS へ数えない。
- 製品側の embed cache（`_internal/embed_cache.json`）を書き換えない。

用語:

- route: `keyword`（keyword のみ）、`hybrid`（keyword + Embedding + RRF）、
  `agentic-lite`（bounded な再検索を含む既存 pipeline 相当）。
"""

from __future__ import annotations

import json
import io
import statistics
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import search
from prompt_templates import SYSTEM_PROMPT, build_user_prompt

SPEC_SCHEMA_VERSION = "1.0"

ROUTE_KEYWORD = "keyword"
ROUTE_HYBRID = "hybrid"
ROUTE_AGENTIC_LITE = "agentic-lite"
ROUTES = (ROUTE_KEYWORD, ROUTE_HYBRID, ROUTE_AGENTIC_LITE)

STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"

ANSWER_ABSTAIN_PHRASES = ("該当情報なし", "関連する資料が見つかりませんでした")

REQUIRED_QUESTION_KEYS = ("id", "category", "query", "answerable")
_ALLOWED_ACCEPTANCE_KEYS = {
    "min_retrieval_hit_rate",
    "min_evidence_coverage",
    "min_evidence_precision",
    "min_evidence_line_overlap",
    "max_forbidden_source_hits",
    "min_abstain_accuracy",
    "max_unsupported_claims",
}

# receipt へ残してよい match の field。本文（text / snippet）は含めない。
_MATCH_RECEIPT_KEYS = (
    "path",
    "chunk_id",
    "heading",
    "start_line",
    "end_line",
    "score",
    "rrf_score",
    "source",
    "page",
    "layout_type",
    "parser",
)


class SpecError(ValueError):
    """評価仕様の不備。fail-closed で実行を止める。"""


# ---------------------------------------------------------------------------
# 仕様の読み込みと検証
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedSource:
    path: str
    heading: str = ""
    line_start: int | None = None
    line_end: int | None = None


ABSTAIN_LAYER_RETRIEVAL = "retrieval"
ABSTAIN_LAYER_ANSWER = "answer"
ABSTAIN_LAYERS = (ABSTAIN_LAYER_RETRIEVAL, ABSTAIN_LAYER_ANSWER)


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    query: str
    answerable: bool
    expected_sources: tuple[ExpectedSource, ...] = ()
    forbidden_sources: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    abstain_layer: str = ABSTAIN_LAYER_RETRIEVAL
    notes: str = ""


@dataclass(frozen=True)
class EvalSpec:
    schema_version: str
    corpus_dir: Path
    acceptance: dict[str, float]
    questions: tuple[Question, ...]
    spec_path: Path


def _as_str_tuple(value: Any, field_name: str, question_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise SpecError(f"{question_id}: {field_name} は文字列配列でなければならない")
    return tuple(x.strip() for x in value if x.strip())


def _parse_expected_sources(value: Any, question_id: str) -> tuple[ExpectedSource, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SpecError(f"{question_id}: expected_sources は配列でなければならない")
    parsed: list[ExpectedSource] = []
    for entry in value:
        if not isinstance(entry, dict) or not str(entry.get("path", "")).strip():
            raise SpecError(f"{question_id}: expected_sources の各要素は path を持つ必要がある")
        line_start = entry.get("line_start")
        line_end = entry.get("line_end")
        for key, line_value in (("line_start", line_start), ("line_end", line_end)):
            if line_value is not None and (
                isinstance(line_value, bool) or not isinstance(line_value, int)
            ):
                raise SpecError(f"{question_id}: expected_sources.{key} は整数または null")
        if line_start is not None and line_end is not None and line_end < line_start:
            raise SpecError(f"{question_id}: expected_sources の line 範囲が逆転している")
        parsed.append(
            ExpectedSource(
                path=search.normalize_source_path(entry["path"]),
                heading=str(entry.get("heading", "")).strip(),
                line_start=line_start,
                line_end=line_end,
            )
        )
    return tuple(parsed)


def parse_spec(data: dict[str, Any], *, spec_path: Path) -> EvalSpec:
    """評価仕様 dict を検証済みの `EvalSpec` へ変換する。"""
    if not isinstance(data, dict):
        raise SpecError("評価仕様はオブジェクトでなければならない")
    if data.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise SpecError(f"未対応の schema_version: {data.get('schema_version')}")

    corpus_dir_raw = str(data.get("corpus_dir", "")).strip()
    if not corpus_dir_raw:
        raise SpecError("corpus_dir は必須")
    corpus_dir = (spec_path.parent / corpus_dir_raw).resolve()

    acceptance = data.get("acceptance", {})
    if not isinstance(acceptance, dict) or not acceptance:
        raise SpecError("acceptance は非空のオブジェクトでなければならない")
    unknown = set(acceptance) - _ALLOWED_ACCEPTANCE_KEYS
    if unknown:
        raise SpecError(f"acceptance に未知のキーがある: {sorted(unknown)}")
    for key, value in acceptance.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SpecError(f"acceptance.{key} は数値でなければならない")

    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise SpecError("questions は1件以上必要")

    questions: list[Question] = []
    seen_ids: set[str] = set()
    for raw in raw_questions:
        if not isinstance(raw, dict):
            raise SpecError("questions の各要素はオブジェクトでなければならない")
        for key in REQUIRED_QUESTION_KEYS:
            if key not in raw:
                raise SpecError(f"question に {key} がない: {raw.get('id', '<no-id>')}")
        question_id = str(raw["id"]).strip()
        if not question_id:
            raise SpecError("question.id が空")
        if question_id in seen_ids:
            raise SpecError(f"question.id が重複している: {question_id}")
        seen_ids.add(question_id)
        answerable = raw["answerable"]
        if not isinstance(answerable, bool):
            raise SpecError(f"{question_id}: answerable は真偽値")
        expected = _parse_expected_sources(raw.get("expected_sources"), question_id)
        if answerable and not expected:
            raise SpecError(f"{question_id}: answerable な質問は expected_sources を必須とする")
        if not answerable and expected:
            raise SpecError(f"{question_id}: 該当情報なしの質問へ expected_sources は指定できない")
        abstain_layer = str(raw.get("abstain_layer", ABSTAIN_LAYER_RETRIEVAL)).strip()
        if abstain_layer not in ABSTAIN_LAYERS:
            raise SpecError(f"{question_id}: abstain_layer は {ABSTAIN_LAYERS} のいずれか")
        if answerable and "abstain_layer" in raw:
            raise SpecError(f"{question_id}: answerable な質問へ abstain_layer は指定できない")
        questions.append(
            Question(
                id=question_id,
                category=str(raw["category"]).strip(),
                query=str(raw["query"]).strip(),
                answerable=answerable,
                expected_sources=expected,
                forbidden_sources=tuple(
                    search.normalize_source_path(x)
                    for x in _as_str_tuple(
                        raw.get("forbidden_sources"), "forbidden_sources", question_id
                    )
                ),
                required_facts=_as_str_tuple(
                    raw.get("required_facts"), "required_facts", question_id
                ),
                forbidden_facts=_as_str_tuple(
                    raw.get("forbidden_facts"), "forbidden_facts", question_id
                ),
                abstain_layer=abstain_layer,
                notes=str(raw.get("notes", "")).strip(),
            )
        )

    return EvalSpec(
        schema_version=SPEC_SCHEMA_VERSION,
        corpus_dir=corpus_dir,
        acceptance={k: float(v) for k, v in acceptance.items()},
        questions=tuple(questions),
        spec_path=spec_path,
    )


def load_spec(spec_path: Path) -> EvalSpec:
    """評価仕様 JSON を読み込み、検証する。"""
    spec_path = Path(spec_path).resolve()
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"評価仕様を読めない: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"評価仕様が不正なJSON: {exc}") from exc
    spec = parse_spec(data, spec_path=spec_path)
    if not spec.corpus_dir.is_dir():
        raise SpecError(f"corpus_dir が存在しない: {corpus_display(spec)}")
    return spec


def corpus_display(spec: EvalSpec) -> str:
    """絶対 path を receipt へ出さないための表示名。"""
    try:
        return spec.corpus_dir.relative_to(spec.spec_path.parent).as_posix()
    except ValueError:
        return spec.corpus_dir.name


# ---------------------------------------------------------------------------
# route 実行
# ---------------------------------------------------------------------------


@dataclass
class RouteOutcome:
    route: str
    status: str
    matches: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    evidence_status: str = "insufficient"
    latency_ms: float = 0.0
    reason: str = ""
    attempts: int = 0


def build_corpus_chunks(corpus_dir: Path) -> list[dict]:
    """固定 corpus をチャンク化する。製品と同じ chunking 実装を使う。"""
    chunks = search.build_source_chunks(Path(corpus_dir))
    if not chunks:
        raise SpecError(f"corpus からチャンクを1件も生成できない: {Path(corpus_dir).name}")
    return chunks


def _retrieval_candidate_limit() -> int:
    return max(1, search.RETRIEVAL_CANDIDATE_LIMIT)


def _expansion_kwargs(chunks: list[dict], corpus_dir: Path | None) -> dict:
    """親子展開が有効なときだけ finalize_ranked_matches へ渡す追加引数を組み立てる。

    評価は製品 skill-source を一切読まないため、``source_root`` は必ず固定
    corpus_dir を指す（既定 None のまま渡すと製品ディレクトリへ誤って
    アクセスし得るため、展開有効時は corpus_dir 必須とする）。
    """
    if not search._parent_child_expansion_enabled() or corpus_dir is None:
        return {}
    return {
        "source_chunks": chunks,
        "source_root": corpus_dir,
        "char_budget": search.PROMPT_EVIDENCE_CHAR_LIMIT,
    }


def run_keyword_route(
    question: Question, chunks: list[dict], *, corpus_dir: Path | None = None
) -> RouteOutcome:
    """keyword のみの決定的 route。Ollama を必要としない。"""
    started = time.perf_counter()
    candidate_limit = _retrieval_candidate_limit()
    keyword_matches = search.keyword_search_chunks(
        question.query,
        top_k=candidate_limit,
        chunks=chunks,
    )
    merged = search.merge_and_filter_matches([], keyword_matches, candidate_limit=candidate_limit)
    matches, confidence, evidence_status = search.finalize_ranked_matches(
        question.query, merged, **_expansion_kwargs(chunks, corpus_dir)
    )
    return RouteOutcome(
        route=ROUTE_KEYWORD,
        status=STATUS_COMPLETED,
        matches=search._limit_items_preserving_groups(
            matches, search.RETRIEVAL_PROMPT_MATCH_LIMIT
        ),
        confidence=confidence,
        evidence_status=evidence_status,
        latency_ms=(time.perf_counter() - started) * 1000,
        attempts=1,
    )


def build_isolated_embed_cache(embed_model: str, chunks: list[dict]) -> dict:
    """評価専用の in-memory embed cache を作る。

    製品の `_internal/embed_cache.json` を読み書きしない。embedding の取得自体は
    製品と同じ `search._get_embedding` を使う。
    """
    entries: dict[str, dict] = {}
    for chunk in chunks:
        embedding = search._get_embedding(chunk["text"], embed_model)
        if embedding is None:
            raise RuntimeError(f"embedding 取得に失敗: {chunk['chunk_id']}")
        entries[chunk["chunk_id"]] = {
            **search._strip_cache_metadata(chunk),
            "embedding": embedding,
        }
    return {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": embed_model,
        "chunking": {
            "max_chars": search.CHUNK_MAX_CHARS,
            "overlap_chars": search.CHUNK_OVERLAP_CHARS,
        },
        "entries": entries,
    }


def run_hybrid_route(
    question: Question,
    chunks: list[dict],
    *,
    embed_model: str | None,
    embed_cache: dict | None,
    corpus_dir: Path | None = None,
) -> RouteOutcome:
    """keyword + Embedding + RRF の route。Embedding model が要る。"""
    if not embed_model or not embed_cache:
        return RouteOutcome(
            route=ROUTE_HYBRID,
            status=STATUS_SKIPPED,
            reason="Embedding model が利用できないため未実行（PASSへ数えない）",
        )
    started = time.perf_counter()
    candidate_limit = _retrieval_candidate_limit()
    keyword_matches = search.keyword_search_chunks(
        question.query,
        top_k=candidate_limit,
        chunks=chunks,
    )
    embed_matches = search.embedding_search(
        question.query,
        embed_model,
        embed_cache,
        top_k=candidate_limit,
    )
    merged = search.merge_results(keyword_matches, embed_matches, max_results=candidate_limit * 2)
    merged = search.filter_by_rrf_score(merged, search.SEARCH_MIN_RRF_SCORE)
    matches, confidence, evidence_status = search.finalize_ranked_matches(
        question.query, merged, **_expansion_kwargs(chunks, corpus_dir)
    )
    return RouteOutcome(
        route=ROUTE_HYBRID,
        status=STATUS_COMPLETED,
        matches=search._limit_items_preserving_groups(
            matches, search.RETRIEVAL_PROMPT_MATCH_LIMIT
        ),
        confidence=confidence,
        evidence_status=evidence_status,
        latency_ms=(time.perf_counter() - started) * 1000,
        attempts=1,
    )


@contextmanager
def _isolated_agentic_inputs(
    chunks: list[dict],
    *,
    embed_model: str | None,
    embed_cache: dict | None,
    corpus_dir: Path | None = None,
):
    """評価 corpus と isolated cache を既存 pipeline へ一時注入する。

    ``run_retrieval_pipeline`` は通常 production の ``skill-source`` と
    ``.model_embed`` を検出する。評価時は固定 corpus と CLI 指定 model を
    使う必要があるため、pipeline の本体を複製せず入力境界だけを差し替える。
    全差し替えは route の終了時に必ず復元し、製品 cache へ書き込ませない。

    親子展開は資料の実bytesを読み直すため（``SKILL_SOURCE_DIR`` 固定 path）、
    ``corpus_dir`` 指定時は ``search.SKILL_SOURCE_DIR`` 自体も一時的に
    差し替え、製品 skill-source への誤アクセスを防ぐ。
    """
    original_build_source_chunks = search.build_source_chunks
    original_detect_embed_model = search.detect_embed_model
    original_build_or_update_embed_index = search.build_or_update_embed_index
    original_skill_source_dir = search.SKILL_SOURCE_DIR

    def build_eval_source_chunks(_source_dir: Path) -> list[dict]:
        return chunks

    def detect_eval_embed_model() -> str | None:
        return embed_model

    def build_eval_embed_index(
        _embed_model: str,
        _chunks: list[dict] | None = None,
        *,
        emit_progress=None,
        cancel_check=None,
    ) -> dict:
        # 評価は製品cache/checkpointへ書き込まず、productionのkeyword-only
        # callback契約だけを受け取る。
        return embed_cache or {}

    search.build_source_chunks = build_eval_source_chunks
    search.detect_embed_model = detect_eval_embed_model
    search.build_or_update_embed_index = build_eval_embed_index
    if corpus_dir is not None:
        search.SKILL_SOURCE_DIR = corpus_dir
        search._invalidate_structure_memo()
    try:
        yield
    finally:
        search.build_source_chunks = original_build_source_chunks
        search.detect_embed_model = original_detect_embed_model
        search.build_or_update_embed_index = original_build_or_update_embed_index
        if corpus_dir is not None:
            search.SKILL_SOURCE_DIR = original_skill_source_dir
            search._invalidate_structure_memo()


def run_agentic_lite_route(
    question: Question,
    *,
    chat_model: str | None,
    chunks: list[dict] | None = None,
    embed_model: str | None = None,
    embed_cache: dict | None = None,
    corpus_dir: Path | None = None,
) -> RouteOutcome:
    """既存 `run_retrieval_pipeline` 相当の bounded 再検索 route。

    chat model による search plan が必要なため、model 未使用時は skip する。
    評価 corpus が未指定の場合は production 資料へ誤って接続しないよう error とする。
    """
    if not chat_model:
        return RouteOutcome(
            route=ROUTE_AGENTIC_LITE,
            status=STATUS_SKIPPED,
            reason="chat model が利用できないため未実行（PASSへ数えない）",
        )
    if chunks is None:
        return RouteOutcome(
            route=ROUTE_AGENTIC_LITE,
            status=STATUS_ERROR,
            reason="評価 corpus が指定されていないため実行不可",
        )
    started = time.perf_counter()
    try:
        with _isolated_agentic_inputs(
            chunks,
            embed_model=embed_model,
            embed_cache=embed_cache,
            corpus_dir=corpus_dir,
        ):
            result = search.run_retrieval_pipeline(question.query, model=chat_model)
    except Exception as exc:  # noqa: BLE001 - route単位で失敗を封じ込め receipt へ記録する
        return RouteOutcome(
            route=ROUTE_AGENTIC_LITE,
            status=STATUS_ERROR,
            reason=f"{type(exc).__name__}",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    return RouteOutcome(
        route=ROUTE_AGENTIC_LITE,
        status=STATUS_COMPLETED,
        matches=list(result.matches),
        confidence=result.confidence,
        evidence_status=result.evidence_status,
        latency_ms=(time.perf_counter() - started) * 1000,
        attempts=len(result.attempts),
    )


def measure_answer_probe(
    question: Question,
    outcome: RouteOutcome,
    *,
    chat_model: str | None,
) -> dict[str, Any]:
    """回答層を本文保存なしで測る。対象は answer 層の no-answer 質問だけ。"""
    base = {
        "question_id": question.id,
        "status": "not_measured",
        "passed": None,
        "retrieval_status": outcome.evidence_status,
        "returned_count": len(outcome.matches),
    }
    if question.abstain_layer != ABSTAIN_LAYER_ANSWER:
        base["reason"] = "answer 層の対象外"
        return base
    if outcome.status != STATUS_COMPLETED:
        base["reason"] = f"retrieval {outcome.status}"
        return base
    if not chat_model:
        base["reason"] = "chat model が利用できない"
        return base

    prompt = build_user_prompt(
        question.query,
        outcome.matches,
        evidence_status=outcome.evidence_status,
        confidence=outcome.confidence,
    )
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            answer = search.stream_ollama_chat(chat_model, SYSTEM_PROMPT, prompt)
    except Exception as exc:  # noqa: BLE001 - type only is safe for receipt
        base.update({"status": STATUS_ERROR, "passed": False, "error_type": type(exc).__name__})
        return base

    answer_text = str(answer or "")
    transport_error = any(
        marker in answer_text or marker in captured_stderr.getvalue()
        for marker in ("接続エラー", "タイムアウト", "HTTPError")
    )
    abstain_phrase = any(phrase in answer_text for phrase in ANSWER_ABSTAIN_PHRASES)
    forbidden_fact = any(fact in answer_text for fact in question.forbidden_facts)
    base.update(
        {
            "status": "measured",
            "passed": bool(answer_text.strip())
            and abstain_phrase
            and not forbidden_fact
            and not transport_error,
            "answer_nonempty": bool(answer_text.strip()),
            "abstain_phrase": abstain_phrase,
            "forbidden_fact": forbidden_fact,
            "transport_error": transport_error,
        }
    )
    return base


def measure_answer_layer(
    spec: EvalSpec,
    *,
    routes: Iterable[str],
    chunks: list[dict],
    embed_model: str | None = None,
    embed_cache: dict | None = None,
    chat_model: str | None = None,
) -> dict[str, list[dict]]:
    """各 route の answer 層対象を1回だけ測る。回答本文は返さない。"""
    answer_questions = [q for q in spec.questions if q.abstain_layer == ABSTAIN_LAYER_ANSWER]
    probes: dict[str, list[dict]] = {}
    for route in routes:
        route_probes = []
        for question in answer_questions:
            outcome = run_route(
                route,
                question,
                chunks=chunks,
                embed_model=embed_model,
                embed_cache=embed_cache,
                chat_model=chat_model,
            )
            route_probes.append(measure_answer_probe(question, outcome, chat_model=chat_model))
        probes[route] = route_probes
    return probes


def apply_answer_probe_to_acceptance(report: dict[str, Any], probes: list[dict]) -> None:
    """測定済み answer probe を route 判定へ反映する。未測定は従来どおり残す。"""
    measured = [p for p in probes if p.get("status") == "measured"]
    if not measured:
        return
    passed = all(p.get("passed") is True for p in measured)
    for check in report["acceptance"]["checks"]:
        if check.get("criterion") != "answer_layer_abstain_verified":
            continue
        check["actual"] = 0 if passed else 1
        check["result"] = "PASS" if passed else "FAIL"
        break
    results = {c["result"] for c in report["acceptance"]["checks"]}
    report["acceptance"]["verdict"] = (
        "FAIL" if "FAIL" in results else "NOT_MEASURED" if "NOT_MEASURED" in results else "PASS"
    )


# ---------------------------------------------------------------------------
# 採点
# ---------------------------------------------------------------------------


@dataclass
class QuestionScore:
    question_id: str
    category: str
    route: str
    status: str
    passed: bool | None
    retrieval_hit: bool | None = None
    hit_at_1: bool | None = None
    evidence_coverage: float | None = None
    evidence_precision: float | None = None
    evidence_line_overlap: float | None = None
    forbidden_source_hits: int = 0
    abstain_correct: bool | None = None
    abstain_layer: str = ""
    evidence_status: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    attempts: int = 0
    returned_count: int = 0
    reason: str = ""
    matches: list[dict] = field(default_factory=list)


def _ranges_overlap(
    a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None
) -> bool:
    if None in (a_start, a_end, b_start, b_end):
        return False
    return int(a_start) <= int(b_end) and int(b_start) <= int(a_end)


def redact_match(match: dict) -> dict:
    """receipt 用に本文を落とした match を返す。"""
    redacted = {}
    for key in _MATCH_RECEIPT_KEYS:
        if key in match and match[key] is not None:
            value = match[key]
            if key == "path":
                value = search.normalize_source_path(value)
            if isinstance(value, float):
                value = round(value, 6)
            redacted[key] = value
    return redacted


def score_question(question: Question, outcome: RouteOutcome) -> QuestionScore:
    """1質問1 route の採点。skip / error は PASS でも FAIL でもなく `None` とする。"""
    if outcome.status != STATUS_COMPLETED:
        return QuestionScore(
            question_id=question.id,
            category=question.category,
            route=outcome.route,
            status=outcome.status,
            passed=None,
            reason=outcome.reason,
            latency_ms=round(outcome.latency_ms, 3),
            attempts=outcome.attempts,
        )

    returned_paths = [search.normalize_source_path(m.get("path", "")) for m in outcome.matches]
    returned_paths = [p for p in returned_paths if p]
    expected_paths = {e.path for e in question.expected_sources}
    forbidden = set(question.forbidden_sources)

    forbidden_hits = sum(1 for p in returned_paths if p in forbidden)

    coverage: float | None = None
    precision: float | None = None
    line_overlap: float | None = None
    retrieval_hit: bool | None = None
    hit_at_1: bool | None = None
    abstain_correct: bool | None = None

    if expected_paths:
        covered = {p for p in returned_paths if p in expected_paths}
        coverage = len(covered) / len(expected_paths)
        precision = (
            sum(1 for p in returned_paths if p in expected_paths) / len(returned_paths)
            if returned_paths
            else 0.0
        )
        retrieval_hit = bool(covered)
        hit_at_1 = bool(returned_paths) and returned_paths[0] in expected_paths

        ranged = [
            e
            for e in question.expected_sources
            if e.line_start is not None and e.line_end is not None
        ]
        if ranged:
            satisfied = 0
            for expected in ranged:
                for match in outcome.matches:
                    if search.normalize_source_path(match.get("path", "")) != expected.path:
                        continue
                    if _ranges_overlap(
                        match.get("start_line"),
                        match.get("end_line"),
                        expected.line_start,
                        expected.line_end,
                    ):
                        satisfied += 1
                        break
            line_overlap = satisfied / len(ranged)
    elif question.abstain_layer == ABSTAIN_LAYER_RETRIEVAL:
        # corpus に一切記載がないケース。retrieval が根拠なしを返せることが要件。
        abstain_correct = outcome.evidence_status == "insufficient"
    else:
        # 近接語ケース。関連文書自体は存在するため根拠は取得される。retrieval 層の要件は
        # 「sufficient と宣言しないこと」に限られ、「該当情報なし」の最終判定は生成
        # プロンプト契約（prompt_templates）側にある。回答生成なしでは確定できない。
        abstain_correct = outcome.evidence_status != "sufficient"

    if question.answerable:
        passed = bool(retrieval_hit) and forbidden_hits == 0
    else:
        passed = bool(abstain_correct) and forbidden_hits == 0

    return QuestionScore(
        question_id=question.id,
        category=question.category,
        route=outcome.route,
        status=outcome.status,
        passed=passed,
        retrieval_hit=retrieval_hit,
        hit_at_1=hit_at_1,
        evidence_coverage=None if coverage is None else round(coverage, 4),
        evidence_precision=None if precision is None else round(precision, 4),
        evidence_line_overlap=None if line_overlap is None else round(line_overlap, 4),
        forbidden_source_hits=forbidden_hits,
        abstain_correct=abstain_correct,
        abstain_layer="" if question.answerable else question.abstain_layer,
        evidence_status=outcome.evidence_status,
        confidence=round(outcome.confidence, 4),
        latency_ms=round(outcome.latency_ms, 3),
        attempts=outcome.attempts,
        returned_count=len(returned_paths),
        matches=[redact_match(m) for m in outcome.matches],
    )


def _mean(values: Iterable[float]) -> float | None:
    collected = [v for v in values if v is not None]
    if not collected:
        return None
    return round(statistics.fmean(collected), 4)


def aggregate_route(scores: list[QuestionScore]) -> dict[str, Any]:
    """1 route の集計。skip/error は分母から除外し、別カウントで残す。"""
    completed = [s for s in scores if s.status == STATUS_COMPLETED]
    skipped = [s for s in scores if s.status == STATUS_SKIPPED]
    errored = [s for s in scores if s.status == STATUS_ERROR]
    answerable = [s for s in completed if s.retrieval_hit is not None]
    unanswerable = [s for s in completed if s.abstain_correct is not None]
    latencies = [s.latency_ms for s in completed]

    return {
        "questions_total": len(scores),
        "completed": len(completed),
        "skipped": len(skipped),
        "errored": len(errored),
        "passed": sum(1 for s in completed if s.passed),
        "failed": sum(1 for s in completed if s.passed is False),
        "retrieval_hit_rate": (
            round(sum(1 for s in answerable if s.retrieval_hit) / len(answerable), 4)
            if answerable
            else None
        ),
        "hit_at_1_rate": (
            round(sum(1 for s in answerable if s.hit_at_1) / len(answerable), 4)
            if answerable
            else None
        ),
        "evidence_coverage_mean": _mean(s.evidence_coverage for s in answerable),
        "evidence_precision_mean": _mean(s.evidence_precision for s in answerable),
        "evidence_line_overlap_mean": _mean(s.evidence_line_overlap for s in answerable),
        "forbidden_source_hits": sum(s.forbidden_source_hits for s in completed),
        "abstain_accuracy": (
            round(sum(1 for s in unanswerable if s.abstain_correct) / len(unanswerable), 4)
            if unanswerable
            else None
        ),
        "answer_layer_abstain_pending": sum(
            1 for s in unanswerable if s.abstain_layer == ABSTAIN_LAYER_ANSWER
        ),
        "attempts_mean": _mean(s.attempts for s in completed),
        "attempts_median": round(statistics.median(s.attempts for s in completed), 3)
        if completed
        else None,
        "attempts_max": max((s.attempts for s in completed), default=None),
        "latency_ms_median": round(statistics.median(latencies), 3) if latencies else None,
        "latency_ms_max": round(max(latencies), 3) if latencies else None,
    }


def evaluate_acceptance(summary: dict[str, Any], acceptance: dict[str, float]) -> dict[str, Any]:
    """受入基準の判定。測定できていない指標は PASS にしない。"""
    checks: list[dict[str, Any]] = []
    minimums = {
        "min_retrieval_hit_rate": "retrieval_hit_rate",
        "min_evidence_coverage": "evidence_coverage_mean",
        "min_evidence_precision": "evidence_precision_mean",
        "min_evidence_line_overlap": "evidence_line_overlap_mean",
        "min_abstain_accuracy": "abstain_accuracy",
    }
    maximums = {
        "max_forbidden_source_hits": "forbidden_source_hits",
    }
    for key, metric in minimums.items():
        if key not in acceptance:
            continue
        actual = summary.get(metric)
        checks.append(
            {
                "criterion": key,
                "metric": metric,
                "threshold": acceptance[key],
                "actual": actual,
                "result": "NOT_MEASURED"
                if actual is None
                else ("PASS" if actual >= acceptance[key] else "FAIL"),
            }
        )
    for key, metric in maximums.items():
        if key not in acceptance:
            continue
        actual = summary.get(metric)
        checks.append(
            {
                "criterion": key,
                "metric": metric,
                "threshold": acceptance[key],
                "actual": actual,
                "result": "NOT_MEASURED"
                if actual is None
                else ("PASS" if actual <= acceptance[key] else "FAIL"),
            }
        )
    if summary.get("answer_layer_abstain_pending"):
        checks.append(
            {
                "criterion": "answer_layer_abstain_verified",
                "metric": "answer_layer_abstain_pending",
                "threshold": 0,
                "actual": summary["answer_layer_abstain_pending"],
                "result": "NOT_MEASURED",
            }
        )
    if summary.get("skipped"):
        checks.append(
            {
                "criterion": "route_executed",
                "metric": "skipped",
                "threshold": 0,
                "actual": summary["skipped"],
                "result": "NOT_MEASURED",
            }
        )
    results = {c["result"] for c in checks}
    if "FAIL" in results:
        verdict = "FAIL"
    elif "NOT_MEASURED" in results or not checks:
        verdict = "NOT_MEASURED"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "checks": checks}


def run_route(
    route: str,
    question: Question,
    *,
    chunks: list[dict],
    embed_model: str | None = None,
    embed_cache: dict | None = None,
    chat_model: str | None = None,
    corpus_dir: Path | None = None,
) -> RouteOutcome:
    """route 名から適切な実行関数へ振り分ける。"""
    if route == ROUTE_KEYWORD:
        return run_keyword_route(question, chunks, corpus_dir=corpus_dir)
    if route == ROUTE_HYBRID:
        return run_hybrid_route(
            question,
            chunks,
            embed_model=embed_model,
            embed_cache=embed_cache,
            corpus_dir=corpus_dir,
        )
    if route == ROUTE_AGENTIC_LITE:
        return run_agentic_lite_route(
            question,
            chat_model=chat_model,
            chunks=chunks,
            embed_model=embed_model,
            embed_cache=embed_cache,
            corpus_dir=corpus_dir,
        )
    raise SpecError(f"未知の route: {route}")


def run_evaluation(
    spec: EvalSpec,
    *,
    routes: Iterable[str],
    chunks: list[dict],
    embed_model: str | None = None,
    embed_cache: dict | None = None,
    chat_model: str | None = None,
    repeat: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """全 route × 全質問 × repeat を実行し、集計済みの構造体を返す。

    非決定的な route を1回だけ測って最良値を採らないため、`repeat` 回の測定結果を
    すべて保持し、集計は run ごとに算出したうえで min/median/max を残す。
    """
    if repeat < 1:
        raise SpecError("repeat は1以上でなければならない")
    routes = tuple(routes)
    for route in routes:
        if route not in ROUTES:
            raise SpecError(f"未知の route: {route}")

    route_reports: dict[str, Any] = {}
    for route in routes:
        runs: list[dict[str, Any]] = []
        for run_index in range(1, repeat + 1):
            if progress:
                progress(f"{route}: run {run_index}/{repeat}")
            scores = [
                score_question(
                    question,
                    run_route(
                        route,
                        question,
                        chunks=chunks,
                        embed_model=embed_model,
                        embed_cache=embed_cache,
                        chat_model=chat_model,
                        corpus_dir=spec.corpus_dir,
                    ),
                )
                for question in spec.questions
            ]
            runs.append({"run": run_index, "summary": aggregate_route(scores), "scores": scores})

        summaries = [r["summary"] for r in runs]
        representative = _representative_summary(summaries)
        route_reports[route] = {
            "runs": runs,
            "summary": representative,
            "stability": _stability(summaries),
            "acceptance": evaluate_acceptance(representative, spec.acceptance),
        }
    return route_reports


_STABILITY_METRICS = (
    "retrieval_hit_rate",
    "hit_at_1_rate",
    "evidence_coverage_mean",
    "evidence_precision_mean",
    "evidence_line_overlap_mean",
    "abstain_accuracy",
    "latency_ms_median",
)


def _representative_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """複数 run の代表値。最良値ではなく中央値を採る。"""
    if len(summaries) == 1:
        return dict(summaries[0])
    representative = dict(summaries[0])
    for metric in _STABILITY_METRICS:
        values = [s.get(metric) for s in summaries if s.get(metric) is not None]
        representative[metric] = round(statistics.median(values), 4) if values else None
    for metric in ("passed", "failed", "forbidden_source_hits"):
        values = [s.get(metric, 0) for s in summaries]
        representative[metric] = int(statistics.median(values))
    return representative


def _stability(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """run 間のばらつき。1 run のみの場合も構造を維持する。"""
    stability: dict[str, Any] = {"runs": len(summaries)}
    for metric in _STABILITY_METRICS:
        values = [s.get(metric) for s in summaries if s.get(metric) is not None]
        stability[metric] = (
            {
                "min": round(min(values), 4),
                "median": round(statistics.median(values), 4),
                "max": round(max(values), 4),
            }
            if values
            else None
        )
    return stability


def overall_verdict(route_reports: dict[str, Any]) -> str:
    """全 route の受入判定を合成する。1つでも FAIL があれば FAIL。"""
    verdicts = {report["acceptance"]["verdict"] for report in route_reports.values()}
    if "FAIL" in verdicts:
        return "FAIL"
    if "NOT_MEASURED" in verdicts or not verdicts:
        return "NOT_MEASURED"
    return "PASS"
