"""比較用合成corpusでMac固定chat候補を隔離評価する。

receiptには自由文回答・資料本文・絶対pathを保存しない。デフォルトは準備receiptのみ。
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tempfile
import types
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_internal"))
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import coverage_evaluator  # noqa: E402
import eval_harness  # noqa: E402
import macos_runtime  # noqa: E402
import search  # noqa: E402
from deep_research import DEEP_NUM_CTX, run_deep_research  # noqa: E402
from model_config import (  # noqa: E402
    MAC_MODEL_DIGESTS,
    canonical_model_reference,
    resolve_model_reasoning,
)

HOLDOUT_ROOT = EVAL_ROOT / "macos-large-model-holdout"
NORMAL_SPEC = HOLDOUT_ROOT / "eval-spec-normal.json"
DEEP_SPEC = HOLDOUT_ROOT / "eval-spec-deep.json"
MODES = ("answer", "deep")
CODE_FILES = (
    "_internal/model_config.py",
    "_internal/search.py",
    "_internal/deep_research.py",
    "_internal/deep_transport.py",
    "_internal/web_services.py",
    "_internal/web_server.py",
    "_internal/prompt_templates.py",
    "scripts/macos_runtime.py",
    "scripts/macos-runtime.example.json",
    "scripts/start-macos.sh",
    "scripts/start-ollama-macos.sh",
    "tests/eval/eval_harness.py",
    "tests/eval/coverage_evaluator.py",
    "tests/eval/macos_large_model_compare.py",
    "tests/eval/macos-large-model-holdout/eval-spec-normal.json",
    "tests/eval/macos-large-model-holdout/eval-spec-deep.json",
)
RECEIPT_SCHEMA = "1.0"
HOLDOUT_GROUP_ID = "MACOS-LARGE-MODEL-20260927-A"


def snapshot_files(root: Path) -> list[str]:
    """Copy only executable source and the explicit synthetic evaluation inputs."""
    names = set(CODE_FILES) | {"VERSION", "_internal/scripts/collect_skill_source.ps1"}
    names.update(
        p.relative_to(root).as_posix() for p in (root / "_internal").glob("*.py")
    )
    names.update(
        p.relative_to(root).as_posix() for p in (root / "tests/eval").glob("*.py")
    )
    names.update(
        p.relative_to(root).as_posix()
        for p in (root / "tests/eval/macos-large-model-holdout/corpus").rglob("*")
        if p.is_file()
    )
    return sorted(names)


def verify_snapshot(receipt: dict, root: Path = ROOT) -> None:
    for name, digest in receipt["product"]["files"].items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root.resolve())
            or sha256_file(path) != digest
        ):
            raise CompareConfigError("評価snapshotのsourceまたはcorpusが変更されました")


def copy_snapshot(receipt: dict, destination: Path, *, source: Path = ROOT) -> None:
    """Bind the exact copied bytes, including transitive worker modules, to the plan."""
    for name, digest in receipt["product"]["files"].items():
        origin = source / name
        target = destination / name
        if not origin.resolve().is_relative_to(
            source.resolve()
        ) or not target.resolve().is_relative_to(destination.resolve()):
            raise CompareConfigError("snapshot pathが不正です")
        data = origin.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise CompareConfigError("準備後に評価対象が変更されました")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    verify_snapshot(receipt, destination)


def question_spec(spec: dict, question_id: str) -> dict:
    """Bind a question-scoped rubric to both its parent spec and item identity."""
    items = [item for item in spec["items"] if item["id"] == question_id]
    if len(items) != 1:
        raise CompareConfigError("採点対象のquestion IDが一意ではありません")
    binding = {
        "parentSpecSha256": spec["spec_sha256"],
        "questionId": question_id,
        "items": items,
        "acceptance": spec["acceptance"],
    }
    digest = hashlib.sha256(
        json.dumps(
            binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {**spec, "items": items, "spec_sha256": digest}


class CompareConfigError(ValueError):
    """比較条件が不正、または安全に評価できない。"""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_snapshot(corpus_root: Path) -> list[dict]:
    if not corpus_root.is_dir():
        raise CompareConfigError("合成評価corpusがありません")
    result = []
    for path in sorted(corpus_root.rglob("*")):
        if path.is_file():
            result.append(
                {
                    "path": path.relative_to(corpus_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not result:
        raise CompareConfigError("合成評価corpusが空です")
    return result


def validate_run_id(value: str) -> str:
    if (
        value in {".", ".."}
        or value.startswith(".")
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value)
    ):
        raise CompareConfigError(
            "run-idは英数字で始まる64文字以内の安全な名前にしてください"
        )
    return value


def validate_local_ollama_host(host: str) -> None:
    parsed = urlsplit(host)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise CompareConfigError("評価はlocalhostのOllamaだけに接続できます")


def validate_owned_runtime(runtime_config_path: Path | None = None) -> None:
    """Refuse an Ollama API that is not owned by the configured Mac launcher."""
    configured_path = runtime_config_path or Path(
        os.environ.get(
            "OFFLINE_AI_MACOS_RUNTIME_CONFIG",
            ROOT / ".test-results" / "macos-large-model" / "host-runtime.json",
        )
    )
    try:
        config = macos_runtime.load_runtime_config(configured_path)
    except (OSError, ValueError) as exc:
        raise CompareConfigError("Mac runtime configを検証できません") from exc
    args = types.SimpleNamespace(
        models_directory=config["models_directory"],
        volume_uuid=config["volume_uuid"],
        ollama_executable=config["ollama_executable"],
    )
    if not macos_runtime._owner_matches(args):
        raise CompareConfigError(
            "Ollamaが設定済みMac launcherの所有プロセスではありません"
        )


def planned_case_runs(models: list[str], modes: list[str], repeat: int) -> dict:
    normal = eval_harness.load_spec(NORMAL_SPEC)
    deep = coverage_evaluator.load_spec(DEEP_SPEC)
    normal_cases = len(normal.questions)
    deep_cases = len(deep["items"])
    normal_ids = {question.id for question in normal.questions}
    deep_ids = {item["id"] for item in deep["items"]}
    if normal_ids != deep_ids or any(
        not question.holdout for question in normal.questions
    ):
        raise CompareConfigError("normal/deepの固定holdout IDが一致しません")
    deep_queries = {item["id"]: item.get("query", "") for item in deep["items"]}
    if any(
        not deep_queries[question.id] or deep_queries[question.id] != question.query
        for question in normal.questions
    ):
        raise CompareConfigError("normal/deepの固定holdout queryが一致しません")
    if set(modes) - set(MODES) or not modes:
        raise CompareConfigError("評価modeが不正です")
    if repeat < 1 or repeat > 3:
        raise CompareConfigError("合成比較のrepeatは1〜3です")
    normal_total = normal_cases * ("answer" in modes) * len(models) * repeat
    deep_total = deep_cases * ("deep" in modes) * len(models) * repeat
    normal_max_seconds = normal_total * 600
    deep_max_seconds = deep_total * 1800
    warmup_max_seconds = len(models) * repeat * 300
    return {
        "normal_case_count_per_model_run": normal_cases if "answer" in modes else 0,
        "deep_case_count_per_model_run": deep_cases if "deep" in modes else 0,
        "models": len(models),
        "repeat": repeat,
        "warmupAttemptsUpperBound": len(models) * repeat,
        "normalMaxSeconds": normal_max_seconds,
        "deepMaxSeconds": deep_max_seconds,
        "warmupMaxSeconds": warmup_max_seconds,
        "totalMaxSeconds": normal_max_seconds + deep_max_seconds + warmup_max_seconds,
        "total_case_runs": normal_total + deep_total,
    }


def build_plan(
    *,
    run_id: str,
    models: list[str],
    modes: list[str],
    repeat: int,
    reasoning: str,
    embed_model: str | None,
) -> dict:
    run_id = validate_run_id(run_id)
    if (
        len(set(models)) != len(models)
        or not models
        or set(models) - set(MAC_MODEL_DIGESTS)
    ):
        raise CompareConfigError("比較対象は登録済みの固定chat候補から選んでください")
    if embed_model and canonical_model_reference(embed_model) != "bge-m3:latest":
        raise CompareConfigError("評価Embeddingは固定候補bge-m3だけを指定できます")
    counts = planned_case_runs(models, modes, repeat)
    normal = eval_harness.load_spec(NORMAL_SPEC)
    deep = coverage_evaluator.load_spec(DEEP_SPEC)
    normal_ids = {question.id for question in normal.questions}
    corpus_files = corpus_snapshot(Path(deep["corpus_root"]))
    identities = []
    for model in models:
        normal_contract = resolve_model_reasoning(model, reasoning)
        deep_contract = resolve_model_reasoning(model, reasoning)
        identities.append(
            {
                "tag": model,
                "expectedDigest": MAC_MODEL_DIGESTS[model],
                "answerReasoning": {
                    "requested": normal_contract["requested"],
                    "apiThinkValue": normal_contract["think"],
                    "effective": normal_contract["effective"],
                    "thinkingVisible": normal_contract["thinking_visible"],
                },
                "deepReasoning": {
                    "requested": deep_contract["requested"],
                    "apiThinkValue": deep_contract["think"],
                    "effective": deep_contract["effective"],
                    "thinkingEnabled": deep_contract["effective"] != "off",
                    "thinkingVisible": False,
                },
            }
        )
    return {
        "schemaVersion": RECEIPT_SCHEMA,
        "runId": run_id,
        "status": "PREPARED_NOT_RUN",
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "product": {
            "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
            "files": {name: sha256_file(ROOT / name) for name in snapshot_files(ROOT)},
            "executionSource": "isolated-copy-verified-before-each-case-and-after-run",
        },
        "models": identities,
        "modes": list(modes),
        "evaluationSplit": "synthetic-calibration-only",
        "embedding": {
            "requestedModel": embed_model,
            "comparisonMode": "hybrid-with-shared-in-memory-cache"
            if embed_model
            else "keyword-only",
            "acceptanceNote": "Embedding未指定時の結果はkeyword-only wiringでありhybrid受入と解釈しない",
        },
        "acceptanceEligible": False,
        "acceptanceIneligibleReason": "evaluation-group-used-for-remediation",
        "reasoningStage": "stage1-cost" if reasoning == "off" else "stage2-precision",
        "requestedReasoning": reasoning,
        "budgets": {
            "normalPipelineSecondsPerQuestion": 300,
            "normalAnswerGenerationSecondsPerQuestion": 300,
            "normalContextTokens": search.NUM_CTX,
            "deepContextTokens": DEEP_NUM_CTX,
            "deepJsonNumPredict": 8192,
            "deepTextNumPredict": 4096,
            "deepTimeoutSeconds": 1800,
            "integrationReserveSeconds": 120,
            "finalizationReserveSeconds": 5,
            "deepReadStageSeparatesExtractionAndVerification": False,
            "maximumExecutionSecondsIncludesWarmups": True,
        },
        "syntheticCorpus": {
            "scope": "tests/eval/macos-large-model-holdout/corpus only; production skill-source and other evaluation corpora are never opened",
            "files": corpus_files,
            "sourceSpecSha256": sha256_file(DEEP_SPEC),
            "normalSpecSha256": sha256_file(NORMAL_SPEC),
        },
        "holdout": {
            "status": "CALIBRATION_USED_FOR_REMEDIATION",
            "groupId": HOLDOUT_GROUP_ID,
            "itemIds": sorted(normal_ids),
            "categories": sorted({question.category for question in normal.questions}),
            "note": "修正に使用済みの合成corpus。独立保留評価・実資料の採用判断には使わない。",
        },
        "plannedCaseRuns": counts,
        "receiptDataPolicy": {
            "answerText": "not-stored",
            "sourceText": "not-stored",
            "absolutePaths": "not-stored",
            "thinkingText": "not-stored",
            "storedOutput": "hashes, enumerated states, counts, and timings only",
        },
        "runs": [],
    }


def _http_json(path: str, timeout: float = 3.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        f"{search.OLLAMA_HOST.rstrip('/')}{path}", method="GET"
    )
    with opener.open(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def loaded_state(model: str) -> str:
    try:
        data = _http_json("/api/ps")
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return "unknown"
    names = {
        str(item.get("name", ""))
        for item in data.get("models", [])
        if isinstance(item, dict)
    }
    return "warm" if model in names else "not-loaded-before-run"


def compact_normal_report(
    route_report: dict,
    answer_quality: list[dict],
    answer_layer: list[dict],
    *,
    embed_model: str | None,
) -> dict:
    summary = route_report["summary"]
    scores = route_report["runs"][0]["scores"] if route_report.get("runs") else []
    return {
        "route": "agentic-lite",
        "embeddingUsed": embed_model is not None,
        "retrieval": {
            key: summary.get(key)
            for key in (
                "completed",
                "skipped",
                "retrieval_hit_rate",
                "hit_at_1_rate",
                "evidence_coverage_mean",
                "evidence_precision_mean",
                "evidence_line_overlap_mean",
                "abstain_accuracy",
                "forbidden_source_hits",
                "latency_ms_median",
            )
        },
        "acceptanceVerdict": route_report["acceptance"].get("verdict"),
        "questions": [
            {
                "id": score.question_id,
                "status": score.status,
                "passed": score.passed,
                "retrievalHit": score.retrieval_hit,
                "evidenceCoverage": score.evidence_coverage,
                "evidencePrecision": score.evidence_precision,
                "lineOverlap": score.evidence_line_overlap,
                "latencyMs": score.latency_ms,
                "attempts": score.attempts,
            }
            for score in scores
        ],
        "answerQuality": [
            {
                "id": item.get("question_id"),
                "status": item.get("status"),
                "passed": item.get("passed"),
                "missingRequiredFacts": item.get("missing_required_facts"),
                "forbiddenFact": item.get("forbidden_fact"),
                "citesExpectedSource": item.get("cites_expected_source"),
                "latencyMs": item.get("answer_latency_ms"),
            }
            for item in answer_quality
        ],
        "answerLayer": [
            {
                "id": item.get("question_id"),
                "status": item.get("status"),
                "passed": item.get("passed"),
                "abstainPhrase": item.get("abstain_phrase"),
                "forbiddenFact": item.get("forbidden_fact"),
                "latencyMs": item.get("answer_latency_ms"),
            }
            for item in answer_layer
        ],
    }


def run_normal_case(
    model: str,
    reasoning: str,
    repeat_index: int,
    spec,
    chunks: list[dict],
    embed_model: str | None,
    embed_cache: dict | None,
    snapshot_receipt: dict | None = None,
) -> dict:
    before = loaded_state(model)
    original_stream = search.stream_ollama_chat
    original_pipeline = search.run_retrieval_pipeline

    def configured_stream(chat_model, system, prompt, **kwargs):
        deadline = time.monotonic() + 300
        return original_stream(
            chat_model,
            system,
            prompt,
            reasoning=reasoning,
            remaining_seconds=lambda: max(0.001, deadline - time.monotonic()),
            **kwargs,
        )

    def bounded_pipeline(query, **kwargs):
        deadline = time.monotonic() + 300
        existing_remaining = kwargs.pop("remaining_seconds", None)

        def remaining_seconds():
            budget = max(0.0, deadline - time.monotonic())
            if existing_remaining is not None:
                budget = min(budget, max(0.0, float(existing_remaining())))
            return budget

        return original_pipeline(
            query,
            reasoning=reasoning,
            remaining_seconds=remaining_seconds,
            **kwargs,
        )

    search.stream_ollama_chat = configured_stream
    search.run_retrieval_pipeline = bounded_pipeline
    started = time.monotonic()
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            scores = []
            answer_quality = []
            answer_layer = []
            for question in spec.questions:
                if snapshot_receipt is not None:
                    verify_snapshot(snapshot_receipt)
                outcome = eval_harness.run_route(
                    eval_harness.ROUTE_AGENTIC_LITE,
                    question,
                    chunks=chunks,
                    embed_model=embed_model,
                    embed_cache=embed_cache,
                    chat_model=model,
                    corpus_dir=spec.corpus_dir,
                )
                scores.append(
                    eval_harness.score_question(
                        question, outcome, run_index=repeat_index
                    )
                )
                if question.answerable and question.required_facts:
                    answer_quality.append(
                        eval_harness.measure_answer_quality_probe(
                            question,
                            outcome,
                            chat_model=model,
                            route=eval_harness.ROUTE_AGENTIC_LITE,
                            run_index=repeat_index,
                        )
                    )
                elif question.abstain_layer == eval_harness.ABSTAIN_LAYER_ANSWER:
                    answer_layer.append(
                        eval_harness.measure_answer_probe(
                            question,
                            outcome,
                            chat_model=model,
                            route=eval_harness.ROUTE_AGENTIC_LITE,
                            run_index=repeat_index,
                        )
                    )
            summary = eval_harness.aggregate_route(scores)
            route_report = {
                "summary": summary,
                "runs": [{"run": repeat_index, "summary": summary, "scores": scores}],
                "acceptance": eval_harness.evaluate_acceptance(
                    summary, spec.acceptance
                ),
            }
            eval_harness.apply_answer_probe_to_acceptance(route_report, answer_layer)
            eval_harness.apply_answer_quality_to_acceptance(
                route_report, answer_quality
            )
    finally:
        search.stream_ollama_chat = original_stream
        search.run_retrieval_pipeline = original_pipeline
    return {
        "mode": "answer",
        "repeat": repeat_index,
        "model": model,
        "stateBefore": before,
        "stateAfter": loaded_state(model),
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "loadDuration": "not-exposed-by-current-search-contract",
        **compact_normal_report(
            route_report,
            answer_quality,
            answer_layer,
            embed_model=embed_model,
        ),
    }


def run_deep_case(
    model: str,
    item: dict,
    repeat_index: int,
    spec: dict,
    *,
    reasoning: str,
    embed_model: str | None,
    embed_cache_path: Path | None,
    embed_cache_sha256: str | None,
    timeout_seconds: int = 1800,
) -> dict:
    query_id = item["id"]
    query = item["query"]
    before = loaded_state(model)
    started = time.monotonic()
    result = run_deep_research(
        query,
        model=model,
        reasoning=reasoning,
        source_root=Path(spec["corpus_root"]),
        timeout_seconds=timeout_seconds,
        isolate_retrieval_embedding=True,
        retrieval_embedding_model=embed_model,
        retrieval_embedding_cache_path=embed_cache_path,
        retrieval_embedding_cache_sha256=embed_cache_sha256,
    )
    elapsed = time.monotonic() - started
    scoped_spec = question_spec(spec, query_id)
    evaluation = coverage_evaluator.evaluate_result(
        scoped_spec, result.to_public_dict()
    )
    diagnostics = result.diagnostics
    return {
        "mode": "deep",
        "questionId": query_id,
        "repeat": repeat_index,
        "model": model,
        "stateBefore": before,
        "stateAfter": loaded_state(model),
        "elapsedSeconds": round(elapsed, 3),
        "status": result.status,
        "stopReason": result.stop_reason,
        "answerState": result.answer_state,
        "outputSha256": evaluation["result_sha256"],
        "questionSpecSha256": scoped_spec["spec_sha256"],
        "parentSpecSha256": spec["spec_sha256"],
        "machineReview": {
            "status": evaluation["status"],
            "semanticReviewValid": evaluation["semantic_review_valid"],
            "itemCoverage": evaluation["item_coverage"],
            "criticalMissIds": evaluation["critical_misses"],
            "unsupportedCitationIds": evaluation["unsupported_citations"],
            "invalidEvidenceIds": evaluation["invalid_evidence"],
            "sourceHashMismatchIds": evaluation["source_hash_mismatches"],
            "itemChecks": [
                {
                    "id": row["id"],
                    "missingRequiredTermCount": len(row["missing_terms"]),
                    "missingEvidenceTermCount": len(row["missing_evidence_terms"]),
                    "sourceFound": row["source_found"],
                    "contradiction": row["contradiction"],
                    "abstentionReported": row["abstention_reported"],
                }
                for row in evaluation["items"]
            ],
        },
        "diagnostics": {
            "reportedElapsedSeconds": diagnostics.get("elapsed_seconds"),
            "stageSeconds": diagnostics.get("stage_seconds", {}),
            "rounds": diagnostics.get("rounds"),
            "documents": diagnostics.get("documents"),
            "units": diagnostics.get("units"),
            "evidenceCount": diagnostics.get("evidence_count"),
            "evidenceChars": diagnostics.get("evidence_chars"),
            "embeddingModel": diagnostics.get("embedding_model"),
            "requestedReasoning": diagnostics.get("requested_reasoning"),
            "effectiveReasoning": diagnostics.get("effective_reasoning"),
            "apiThinkValue": diagnostics.get("api_think_value"),
            "thinkingEnabled": diagnostics.get("thinking_enabled"),
            "thinkingVisible": diagnostics.get("thinking_visible"),
            "sourceRepair": diagnostics.get("final_answer", {}).get("source_repair"),
            "readStageAggregatesExtractionAndVerification": True,
            "loadDuration": "not-exposed-by-current-worker-contract",
        },
    }


def _atomic_write(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ensure_model_warm(model: str, reasoning: str) -> dict:
    before = loaded_state(model)
    started = time.monotonic()
    if before != "warm":
        search.stream_ollama_chat(
            model,
            "この応答は合成ベンチマークのウォームアップです。",
            "READY の一語だけを返してください。",
            reasoning=reasoning,
            remaining_seconds=lambda: max(0.001, 300 - (time.monotonic() - started)),
        )
    after = loaded_state(model)
    record = {
        "model": model,
        "stateBefore": before,
        "stateAfter": after,
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "result": "already-warm" if before == "warm" else "warmed",
        "coldLoadDuration": "not-isolated; elapsed includes prompt generation",
    }
    if after != "warm":
        raise CompareConfigError("Ollama /api/psでwarm状態を確認できません")
    return record


def mark_calibration(receipt: dict) -> None:
    """Never upgrade this already-used question group, including old plans."""
    receipt["acceptanceEligible"] = False
    receipt["acceptanceIneligibleReason"] = "evaluation-group-used-for-remediation"
    receipt["evaluationSplit"] = "synthetic-calibration-only"
    receipt["holdout"]["status"] = "CALIBRATION_USED_FOR_REMEDIATION"
    receipt["holdout"]["note"] = "修正に使用済みの合成corpus。独立保留評価・実資料の採用判断には使わない。"


def execute(
    receipt: dict,
    output_path: Path,
    *,
    confirm_case_runs: int,
    confirm_max_seconds: int,
    reasoning: str,
    embed_model: str | None,
    runtime_config_path: Path | None = None,
) -> None:
    expected_runs = receipt["plannedCaseRuns"]["total_case_runs"]
    expected_seconds = receipt["plannedCaseRuns"]["totalMaxSeconds"]
    if confirm_case_runs != expected_runs or confirm_max_seconds != expected_seconds:
        raise CompareConfigError("実行件数または最大時間の確認が計画と一致しません")
    validate_local_ollama_host(search.OLLAMA_HOST)
    verify_snapshot(receipt)
    validate_owned_runtime(runtime_config_path)
    for model_entry in receipt["models"]:
        model = model_entry["tag"]
        identity = search.get_model_identity(model)
        if identity["digest"] != model_entry["expectedDigest"]:
            raise CompareConfigError("固定モデルdigestが比較計画と一致しません")
        model_entry["observedDigest"] = identity["digest"]
    receipt["status"] = "RUNNING"
    receipt["startedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    receipt["warmups"] = []
    mark_calibration(receipt)
    receipt["preparation"] = {"status": "RUNNING"}
    _atomic_write(output_path, receipt)

    preparation_started = time.monotonic()
    try:
        normal_spec = eval_harness.load_spec(NORMAL_SPEC)
        chunks = eval_harness.build_corpus_chunks(normal_spec.corpus_dir)
        embed_cache = None
        embed_cache_path = None
        embed_cache_sha256 = None
        if embed_model:
            embed_identity = search.get_embed_model_identity(embed_model)
            if not embed_identity:
                raise CompareConfigError("Embedding modelがOllamaへ導入されていません")
            receipt["embedding"]["observedDigest"] = embed_identity["digest"]
            embed_cache = eval_harness.build_isolated_embed_cache(embed_model, chunks)
            first_entry = next(iter(embed_cache.get("entries", {}).values()), None)
            dimension = (
                len(first_entry.get("embedding", []))
                if isinstance(first_entry, dict)
                else None
            )
            embed_cache["compatibility"] = search._new_embed_compatibility(
                embed_model, embed_identity, dimension
            )
            receipt["embedding"]["fileCount"] = len(embed_cache.get("entries", {}))
        deep_spec = coverage_evaluator.load_spec(DEEP_SPEC)
        with tempfile.TemporaryDirectory(
            prefix="offline-ai-macos-eval-"
        ) as runtime_dir:
            if embed_cache is not None:
                embed_cache_path = Path(runtime_dir) / "embed-cache.json"
                cache_bytes = (
                    json.dumps(embed_cache, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                embed_cache_path.write_bytes(cache_bytes)
                embed_cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
                receipt["embedding"]["runtimeCacheSha256"] = embed_cache_sha256
            receipt["preparation"] = {
                "status": "COMPLETED",
                "elapsedSeconds": round(time.monotonic() - preparation_started, 3),
                "corpusFiles": len(chunks),
                "embeddingCacheFiles": len(embed_cache.get("entries", {}))
                if embed_cache
                else 0,
                "embeddingCacheStorage": "memory plus per-run temporary worker transfer",
            }
            _atomic_write(output_path, receipt)
            for model_entry in receipt["models"]:
                model = model_entry["tag"]
                for repeat_index in range(1, receipt["plannedCaseRuns"]["repeat"] + 1):
                    verify_snapshot(receipt)
                    receipt["warmups"].append(ensure_model_warm(model, reasoning))
                    _atomic_write(output_path, receipt)
                    if "answer" in receipt["modes"]:
                        receipt["runs"].append(
                            run_normal_case(
                                model,
                                reasoning,
                                repeat_index,
                                normal_spec,
                                chunks,
                                embed_model,
                                embed_cache,
                                snapshot_receipt=receipt,
                            )
                        )
                        _atomic_write(output_path, receipt)
                    if "deep" in receipt["modes"]:
                        for item in deep_spec["items"]:
                            verify_snapshot(receipt)
                            receipt["runs"].append(
                                run_deep_case(
                                    model,
                                    item,
                                    repeat_index,
                                    deep_spec,
                                    reasoning=reasoning,
                                    embed_model=embed_model,
                                    embed_cache_path=embed_cache_path,
                                    embed_cache_sha256=embed_cache_sha256,
                                )
                            )
                            _atomic_write(output_path, receipt)
    except KeyboardInterrupt:
        if receipt["preparation"]["status"] == "RUNNING":
            receipt["preparation"]["status"] = "INTERRUPTED"
            receipt["preparation"]["elapsedSeconds"] = round(
                time.monotonic() - preparation_started, 3
            )
        receipt["status"] = "INTERRUPTED"
        receipt["finishedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write(output_path, receipt)
        raise
    except Exception as exc:
        if receipt["preparation"]["status"] == "RUNNING":
            receipt["preparation"]["status"] = "FAILED"
            receipt["preparation"]["elapsedSeconds"] = round(
                time.monotonic() - preparation_started, 3
            )
        receipt["status"] = "FAILED"
        receipt["failureType"] = type(exc).__name__
        receipt["finishedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write(output_path, receipt)
        raise
    try:
        verify_snapshot(receipt)
    except (OSError, CompareConfigError):
        receipt["status"] = "INVALIDATED"
        receipt["failureType"] = "SnapshotChanged"
        _atomic_write(output_path, receipt)
        raise
    receipt["status"] = "COMPLETED"
    receipt["finishedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    receipt["humanAcceptance"] = (
        "NOT_RUN: generated answers require item-by-item source review"
    )
    receipt["qualitySelection"] = (
        "NOT_DECIDED: synthetic holdout cannot establish quality on user material"
    )
    _atomic_write(output_path, receipt)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mac大型chat model隔離比較runner")
    parser.add_argument(
        "--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument(
        "--model", action="append", choices=tuple(MAC_MODEL_DIGESTS), dest="models"
    )
    parser.add_argument("--mode", choices=("answer", "deep", "both"), default="both")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--reasoning", choices=("off", "low", "medium", "high"), default="off"
    )
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--runtime-config", type=Path, default=None)
    parser.add_argument("--snapshot-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--execute", action="store_true", help="run explicit local Ollama evaluations"
    )
    parser.add_argument(
        "--confirm-case-runs", type=int, default=0, help="必須: 計画されたcase run件数"
    )
    parser.add_argument(
        "--confirm-max-seconds",
        type=int,
        default=0,
        help="必須: receiptに示された最大実行秒数を確認する",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    models = args.models or list(MAC_MODEL_DIGESTS)
    modes = ["answer", "deep"] if args.mode == "both" else [args.mode]
    try:
        if args.snapshot_plan is not None:
            if args.snapshot_output is None:
                raise CompareConfigError("snapshot outputが必要です")
            receipt = json.loads(args.snapshot_plan.read_text(encoding="utf-8"))
            execute(
                receipt,
                args.snapshot_output,
                confirm_case_runs=args.confirm_case_runs,
                confirm_max_seconds=args.confirm_max_seconds,
                reasoning=receipt["requestedReasoning"],
                embed_model=receipt["embedding"]["requestedModel"],
                runtime_config_path=args.runtime_config,
            )
            return 0
        receipt = build_plan(
            run_id=args.run_id,
            models=models,
            modes=modes,
            repeat=args.repeat,
            reasoning=args.reasoning,
            embed_model=args.embed_model,
        )
        output_path = (
            ROOT
            / ".test-results"
            / "macos-large-model"
            / receipt["runId"]
            / "receipt.json"
        )
        if output_path.exists():
            raise CompareConfigError(
                "同じrun-idのreceiptが既にあります。新しいrun-idを使ってください"
            )
        if args.execute and (
            args.confirm_case_runs != receipt["plannedCaseRuns"]["total_case_runs"]
            or args.confirm_max_seconds != receipt["plannedCaseRuns"]["totalMaxSeconds"]
        ):
            raise CompareConfigError(
                "--confirm-case-runsと--confirm-max-secondsが計画件数/時間と一致しません。receiptは作成していません"
            )
        _atomic_write(output_path, receipt)
        if args.execute:
            runtime_config = args.runtime_config or Path(
                os.environ.get(
                    "OFFLINE_AI_MACOS_RUNTIME_CONFIG",
                    ROOT / ".test-results/macos-large-model/host-runtime.json",
                )
            )
            with tempfile.TemporaryDirectory(
                prefix="offline-ai-eval-source-"
            ) as temporary:
                isolated = Path(temporary)
                copy_snapshot(receipt, isolated)
                plan_path = isolated / "evaluation-plan.json"
                _atomic_write(plan_path, receipt)
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(isolated / "tests/eval/macos_large_model_compare.py"),
                        "--snapshot-plan",
                        str(plan_path),
                        "--snapshot-output",
                        str(output_path),
                        "--runtime-config",
                        str(runtime_config.resolve()),
                        "--confirm-case-runs",
                        str(args.confirm_case_runs),
                        "--confirm-max-seconds",
                        str(args.confirm_max_seconds),
                    ],
                    cwd=isolated,
                    check=False,
                )
                if completed.returncode:
                    return completed.returncode
            receipt = json.loads(output_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "runId": receipt["runId"],
                    "plannedCaseRuns": receipt["plannedCaseRuns"],
                    "acceptanceEligible": receipt["acceptanceEligible"],
                    "holdout": receipt["holdout"],
                    "receipt": output_path.relative_to(ROOT).as_posix(),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (
        CompareConfigError,
        eval_harness.SpecError,
        coverage_evaluator.CoverageSpecError,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "reason": type(exc).__name__}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
