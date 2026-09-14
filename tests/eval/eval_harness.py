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
EXPECTED_EXPANSIONS = ("", "complete", "partial")


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
    # True の場合、行範囲付き expected_sources が最終根拠に残り、required_facts が
    # その範囲と重なる抜粋に含まれていなければ質問単位で FAIL とする。
    require_expected_lines: bool = False
    # 親子展開の期待。"complete"（配下を完全に展開）/ "partial"（予算・範囲数で部分展開し、
    # sufficient を宣言しない）/ ""（検査しない）。
    expected_expansion: str = ""
    # True の質問は上限・重みの調整に使わない保留質問として別集計する。
    holdout: bool = False
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
        expected_expansion = str(raw.get("expected_expansion", "")).strip()
        if expected_expansion not in EXPECTED_EXPANSIONS:
            raise SpecError(f"{question_id}: expected_expansion は {EXPECTED_EXPANSIONS} のいずれか")
        if expected_expansion and not answerable:
            raise SpecError(f"{question_id}: expected_expansion は answerable な質問だけに指定できる")
        holdout = raw.get("holdout", False)
        if not isinstance(holdout, bool):
            raise SpecError(f"{question_id}: holdout は真偽値")
        require_expected_lines = raw.get("require_expected_lines", False)
        if not isinstance(require_expected_lines, bool):
            raise SpecError(f"{question_id}: require_expected_lines は真偽値")
        if require_expected_lines and not any(
            e.line_start is not None and e.line_end is not None for e in expected
        ):
            raise SpecError(
                f"{question_id}: require_expected_lines には行範囲付きの expected_sources が必要"
            )
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
                require_expected_lines=require_expected_lines,
                expected_expansion=expected_expansion,
                holdout=holdout,
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
    # search の段階追跡（試行ごと、本文を含まない）。候補段階の recall 算出に使う。
    trace: list[dict] = field(default_factory=list)
    # 検索計画の必須語（agentic-lite のみ）。展開を除いた通常根拠の再判定に使う。
    must_find_terms: tuple[str, ...] = ()


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


def _finalize_route_evidence(
    query: str,
    merged: list[dict],
    chunks: list[dict],
    corpus_dir: Path | None,
    trace: dict | None = None,
) -> tuple[list[dict], float, str]:
    """製品 pipeline と同じ後段（展開 → 件数枠の配分 → 状態の再判定）を適用する。"""
    ranked, _, _ = search.finalize_ranked_matches(
        query, merged, trace=trace, **_expansion_kwargs(chunks, corpus_dir)
    )
    return search.select_final_evidence(
        query, ranked, limit=search.final_evidence_match_limit(), trace=trace
    )


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
    trace: dict = {"attempt": 1}
    matches, confidence, evidence_status = _finalize_route_evidence(
        question.query, merged, chunks, corpus_dir, trace
    )
    return RouteOutcome(
        route=ROUTE_KEYWORD,
        status=STATUS_COMPLETED,
        matches=matches,
        confidence=confidence,
        evidence_status=evidence_status,
        latency_ms=(time.perf_counter() - started) * 1000,
        attempts=1,
        trace=[trace],
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
    trace: dict = {"attempt": 1}
    matches, confidence, evidence_status = _finalize_route_evidence(
        question.query, merged, chunks, corpus_dir, trace
    )
    return RouteOutcome(
        route=ROUTE_HYBRID,
        status=STATUS_COMPLETED,
        matches=matches,
        confidence=confidence,
        evidence_status=evidence_status,
        latency_ms=(time.perf_counter() - started) * 1000,
        attempts=1,
        trace=[trace],
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
        trace=list(getattr(result, "trace", {}).get("attempts", [])),
        must_find_terms=tuple(
            str(term) for term in (getattr(result, "plan", {}) or {}).get("must_find_terms", []) or []
        ),
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
                corpus_dir=spec.corpus_dir,
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


def measure_answer_quality_probe(
    question: Question,
    outcome: RouteOutcome,
    *,
    chat_model: str | None,
) -> dict[str, Any]:
    """回答可能な質問の回答を本文保存なしで検査する。

    検査は機械的な信号に限る: 空回答でない、``required_facts`` を全て含む、
    ``forbidden_facts`` を含まない、期待 source の path を引用する、該当情報なしと
    答えない、接続エラーでない。意味内容の正しさの証明ではなく、特定の例示文言を
    回答へ強制する目的にも使わない（``required_facts`` は短い必要事項に留める）。
    """
    base = {
        "question_id": question.id,
        "status": "not_measured",
        "passed": None,
        "retrieval_status": outcome.evidence_status,
        "returned_count": len(outcome.matches),
    }
    if not question.answerable or not question.required_facts:
        base["reason"] = "回答品質の対象外（answerable かつ required_facts あり）"
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
    missing_facts = sum(1 for fact in question.required_facts if fact not in answer_text)
    forbidden_fact = any(fact in answer_text for fact in question.forbidden_facts)
    cites_expected = any(e.path in answer_text for e in question.expected_sources)
    abstained = any(phrase in answer_text for phrase in ANSWER_ABSTAIN_PHRASES)
    nonempty = bool(answer_text.strip())
    base.update(
        {
            "status": "measured",
            "passed": nonempty
            and missing_facts == 0
            and not forbidden_fact
            and cites_expected
            and not abstained
            and not transport_error,
            "answer_nonempty": nonempty,
            "missing_required_facts": missing_facts,
            "forbidden_fact": forbidden_fact,
            "cites_expected_source": cites_expected,
            "abstained": abstained,
            "transport_error": transport_error,
        }
    )
    return base


def measure_answer_quality(
    spec: EvalSpec,
    *,
    routes: Iterable[str],
    chunks: list[dict],
    embed_model: str | None = None,
    embed_cache: dict | None = None,
    chat_model: str | None = None,
) -> dict[str, list[dict]]:
    """各 route の回答可能な質問を1回ずつ回答生成して検査する。回答本文は返さない。"""
    targets = [q for q in spec.questions if q.answerable and q.required_facts]
    probes: dict[str, list[dict]] = {}
    for route in routes:
        probes[route] = [
            measure_answer_quality_probe(
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
                chat_model=chat_model,
            )
            for question in targets
        ]
    return probes


def apply_answer_quality_to_acceptance(report: dict[str, Any], probes: list[dict]) -> None:
    """回答品質 probe を route 判定へ反映する。未測定だけなら NOT_MEASURED を残す。"""
    if not probes:
        return
    measured = [p for p in probes if p.get("status") in {"measured", STATUS_ERROR}]
    failed = sum(1 for p in measured if p.get("passed") is not True)
    report["acceptance"]["checks"].append(
        {
            "criterion": "answer_quality_verified",
            "metric": "answer_quality_failed",
            "threshold": 0,
            "actual": failed if measured else None,
            "result": "NOT_MEASURED"
            if len(measured) < len(probes)
            else ("PASS" if failed == 0 else "FAIL"),
        }
    )
    results = {c["result"] for c in report["acceptance"]["checks"]}
    report["acceptance"]["verdict"] = (
        "FAIL" if "FAIL" in results else "NOT_MEASURED" if "NOT_MEASURED" in results else "PASS"
    )


_STATUS_RANK = {"insufficient": 0, "partial": 1, "sufficient": 2}


def compare_expansion_reports(
    off_reports: dict[str, Any], on_reports: dict[str, Any]
) -> dict[str, Any]:
    """親子展開 OFF/ON の評価結果を質問ごとに比較する（各 route の run 1 同士）。

    status 遷移（特に sufficient→partial の低下）、合否の変化、再検索の発生率と
    試行回数を残す。低下の理由の特定は receipt の段階追跡と併せて行う。
    """
    comparison: dict[str, Any] = {}
    for route in sorted(set(off_reports) & set(on_reports)):
        off_scores = {s.question_id: s for s in off_reports[route]["runs"][0]["scores"]}
        on_scores = {s.question_id: s for s in on_reports[route]["runs"][0]["scores"]}
        questions = []
        for question_id in off_scores:
            off, on = off_scores[question_id], on_scores.get(question_id)
            if on is None:
                continue
            off_rank = _STATUS_RANK.get(off.evidence_status)
            on_rank = _STATUS_RANK.get(on.evidence_status)
            transition = "same"
            if off_rank is not None and on_rank is not None and off_rank != on_rank:
                transition = "up" if on_rank > off_rank else "down"
            questions.append(
                {
                    "question_id": question_id,
                    "off_status": off.evidence_status,
                    "on_status": on.evidence_status,
                    "transition": transition,
                    "off_passed": off.passed,
                    "on_passed": on.passed,
                    "off_attempts": off.attempts,
                    "on_attempts": on.attempts,
                    "off_line_overlap": off.evidence_line_overlap,
                    "on_line_overlap": on.evidence_line_overlap,
                }
            )

        def retry_rate(scores: dict[str, QuestionScore]) -> float | None:
            completed = [s for s in scores.values() if s.status == STATUS_COMPLETED]
            if not completed:
                return None
            return round(sum(1 for s in completed if s.attempts > 1) / len(completed), 4)

        comparison[route] = {
            "questions": questions,
            "status_down": [q["question_id"] for q in questions if q["transition"] == "down"],
            "status_up": [q["question_id"] for q in questions if q["transition"] == "up"],
            "newly_failed": [
                q["question_id"] for q in questions if q["off_passed"] and q["on_passed"] is False
            ],
            "newly_passed": [
                q["question_id"] for q in questions if q["off_passed"] is False and q["on_passed"]
            ],
            "off_retry_rate": retry_rate(off_scores),
            "on_retry_rate": retry_rate(on_scores),
        }
    return comparison


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
    # require_expected_lines の質問だけ True/False。対象外は None。
    expected_lines_retained: bool | None = None
    # 期待行範囲のうち、支持判定前の検索候補（trace）に入っていた割合。trace が無ければ None。
    candidate_line_recall: float | None = None
    # 期待行範囲と重ならない展開itemの件数（無関係な子の採用数）。
    expanded_irrelevant_ranges: int | None = None
    # expected_expansion の質問だけ True/False。対象外は None。
    expansion_expectation_met: bool | None = None
    holdout: bool = False
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


def _expected_lines_retained(
    question: Question, ranged: list[ExpectedSource], matches: list[dict]
) -> bool:
    """行範囲付きの期待根拠がすべて最終根拠に残り、必要事項が抜粋にあるか。

    各期待範囲と重なる返却根拠が1件以上必要。``required_facts`` がある場合は、
    期待範囲と重なる根拠の抜粋（最終的に渡した本文）に全て含まれる必要がある。
    行範囲が重なるだけで本文が切り詰められて必要記述が消えたケースを落とすため。
    """
    overlapping_snippets: list[str] = []
    for expected in ranged:
        hits = [
            match
            for match in matches
            if search.normalize_source_path(match.get("path", "")) == expected.path
            and _ranges_overlap(
                match.get("start_line"),
                match.get("end_line"),
                expected.line_start,
                expected.line_end,
            )
        ]
        if not hits:
            return False
        overlapping_snippets.extend(str(m.get("snippet") or "") for m in hits)
    text = "\n".join(overlapping_snippets)
    return all(fact in text for fact in question.required_facts)


def _candidate_line_recall(ranged: list[ExpectedSource], trace: list[dict]) -> float | None:
    """期待行範囲が、いずれかの試行の検索候補（支持判定・件数制限の前）に入った割合。

    失敗の分類（候補に入らない / 候補にはあるが選別で落ちる）に使う。
    """
    candidates = [ref for attempt in trace for ref in attempt.get("candidates", [])]
    if not trace:
        return None
    satisfied = sum(
        1
        for expected in ranged
        if any(
            search.normalize_source_path(ref.get("path", "")) == expected.path
            and _ranges_overlap(
                ref.get("start_line"), ref.get("end_line"), expected.line_start, expected.line_end
            )
            for ref in candidates
        )
    )
    return satisfied / len(ranged)


def _expansion_expectation_met(question: Question, outcome: RouteOutcome) -> bool:
    """親子展開の期待（完全展開 / 部分展開）が最終根拠で満たされたか。

    ``partial`` は、部分展開の item が採用され、かつ展開だけを理由に sufficient を
    宣言しないこと。計画§4に合わせ、展開itemを除いた通常根拠だけで sufficient が
    成立する場合の sufficient は許容する（通常根拠の判定を展開で妨げない）。
    ``complete`` は、展開 item が採用され、どれも部分展開でないこと。
    """
    expanded = [m for m in outcome.matches if m.get("source") == "expanded"]
    if not expanded:
        return False
    if question.expected_expansion == "partial":
        if not any(m.get("group_partial") for m in expanded):
            return False
        if outcome.evidence_status != "sufficient":
            return True
        direct_only = [m for m in outcome.matches if m.get("source") != "expanded"]
        _, direct_status = search._calculate_confidence(
            question.query, direct_only, list(outcome.must_find_terms)
        )
        return direct_status == "sufficient"
    return not any(m.get("group_partial") for m in expanded)


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
    expected_lines_retained: bool | None = None
    candidate_line_recall: float | None = None
    expanded_irrelevant_ranges: int | None = None
    expansion_expectation_met: bool | None = None
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
            if question.require_expected_lines:
                expected_lines_retained = _expected_lines_retained(question, ranged, outcome.matches)
            candidate_line_recall = _candidate_line_recall(ranged, outcome.trace)
            expanded_irrelevant_ranges = sum(
                1
                for match in outcome.matches
                if match.get("source") == "expanded"
                and not any(
                    search.normalize_source_path(match.get("path", "")) == expected.path
                    and _ranges_overlap(
                        match.get("start_line"),
                        match.get("end_line"),
                        expected.line_start,
                        expected.line_end,
                    )
                    for expected in ranged
                )
            )
        if question.expected_expansion:
            expansion_expectation_met = _expansion_expectation_met(question, outcome)
    elif question.abstain_layer == ABSTAIN_LAYER_RETRIEVAL:
        # corpus に一切記載がないケース。retrieval が根拠なしを返せることが要件。
        abstain_correct = outcome.evidence_status == "insufficient"
    else:
        # 近接語ケース。関連文書自体は存在するため根拠は取得される。retrieval 層の要件は
        # 「sufficient と宣言しないこと」に限られ、「該当情報なし」の最終判定は生成
        # プロンプト契約（prompt_templates）側にある。回答生成なしでは確定できない。
        abstain_correct = outcome.evidence_status != "sufficient"

    if question.answerable:
        passed = (
            bool(retrieval_hit)
            and forbidden_hits == 0
            and expected_lines_retained is not False
            and expansion_expectation_met is not False
        )
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
        expected_lines_retained=expected_lines_retained,
        candidate_line_recall=None
        if candidate_line_recall is None
        else round(candidate_line_recall, 4),
        expanded_irrelevant_ranges=expanded_irrelevant_ranges,
        expansion_expectation_met=expansion_expectation_met,
        holdout=question.holdout,
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
        "expected_lines_required": sum(
            1 for s in answerable if s.expected_lines_retained is not None
        ),
        "expected_lines_missing": sum(
            1 for s in answerable if s.expected_lines_retained is False
        ),
        "candidate_line_recall_mean": _mean(s.candidate_line_recall for s in answerable),
        "expanded_irrelevant_ranges": sum(s.expanded_irrelevant_ranges or 0 for s in answerable),
        "expansion_expectation_required": sum(
            1 for s in answerable if s.expansion_expectation_met is not None
        ),
        "expansion_expectation_failed": sum(
            1 for s in answerable if s.expansion_expectation_met is False
        ),
        "holdout_total": sum(1 for s in completed if s.holdout),
        "holdout_failed": sum(1 for s in completed if s.holdout and s.passed is False),
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
    if summary.get("expected_lines_required"):
        # 平均 line 一致が閾値を上回っても、指定質問の必要範囲が落ちたら FAIL にする。
        missing = summary.get("expected_lines_missing", 0)
        checks.append(
            {
                "criterion": "expected_lines_retained",
                "metric": "expected_lines_missing",
                "threshold": 0,
                "actual": missing,
                "result": "PASS" if missing == 0 else "FAIL",
            }
        )
    if summary.get("expansion_expectation_required"):
        failed = summary.get("expansion_expectation_failed", 0)
        checks.append(
            {
                "criterion": "expansion_expectation_met",
                "metric": "expansion_expectation_failed",
                "threshold": 0,
                "actual": failed,
                "result": "PASS" if failed == 0 else "FAIL",
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
    "candidate_line_recall_mean",
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
    # 必要範囲の欠落・展開期待の不一致・保留質問の失敗・無関係な展開は、1回でも
    # 起きれば代表値へ残す（中央値で相殺しない）。
    for metric in (
        "expected_lines_missing",
        "expansion_expectation_failed",
        "holdout_failed",
        "expanded_irrelevant_ranges",
    ):
        representative[metric] = max(s.get(metric, 0) for s in summaries)
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
