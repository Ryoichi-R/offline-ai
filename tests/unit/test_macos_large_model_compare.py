import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "macos_large_model_compare",
    ROOT / "tests" / "eval" / "macos_large_model_compare.py",
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def test_prepare_plan_binds_independent_four_case_holdout_and_no_answer_text():
    plan = compare.build_plan(
        run_id="unit-check",
        models=list(compare.MAC_MODEL_DIGESTS),
        modes=["answer", "deep"],
        repeat=3,
        reasoning="off",
        embed_model=None,
    )

    assert plan["status"] == "PREPARED_NOT_RUN"
    assert plan["plannedCaseRuns"]["normal_case_count_per_model_run"] == 4
    assert plan["plannedCaseRuns"]["deep_case_count_per_model_run"] == 4
    assert plan["plannedCaseRuns"]["total_case_runs"] == 48
    assert plan["plannedCaseRuns"]["normalMaxSeconds"] == 14_400
    assert plan["plannedCaseRuns"]["deepMaxSeconds"] == 43_200
    assert plan["plannedCaseRuns"]["warmupMaxSeconds"] == 1_800
    assert plan["plannedCaseRuns"]["totalMaxSeconds"] == 59_400
    assert plan["receiptDataPolicy"]["answerText"] == "not-stored"
    assert plan["syntheticCorpus"]["files"]
    assert plan["evaluationSplit"] == "synthetic-calibration-only"
    assert plan["holdout"]["status"] == "CALIBRATION_USED_FOR_REMEDIATION"
    assert len(plan["holdout"]["itemIds"]) == 4
    assert plan["acceptanceEligible"] is False
    assert "answer" not in plan


def test_runner_rejects_remote_ollama_and_unsafe_run_ids():
    with pytest.raises(compare.CompareConfigError, match="localhost"):
        compare.validate_local_ollama_host("https://ollama.example")
    with pytest.raises(compare.CompareConfigError):
        compare.validate_run_id("../escape")


def test_runner_requires_the_configured_launcher_owner(tmp_path, monkeypatch):
    config_path = tmp_path / "host-runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "modelsDirectory": "/Volumes/AI/ollama-models",
                "volumeUuid": "AABB-CCDD",
                "ollamaExecutable": "/opt/ollama",
                "pythonExecutable": "/opt/python",
                "ollamaVersion": "0.34.4",
                "minimumFreeBytes": 1024,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(compare.macos_runtime, "_owner_matches", lambda _args: False)

    with pytest.raises(compare.CompareConfigError, match="所有プロセス"):
        compare.validate_owned_runtime(config_path)


def test_single_model_run_cannot_claim_comparison_acceptance_eligibility():
    plan = compare.build_plan(
        run_id="single-model",
        models=["qwen3.5:122b"],
        modes=["answer", "deep"],
        repeat=3,
        reasoning="off",
        embed_model="bge-m3",
    )

    assert plan["acceptanceEligible"] is False


@pytest.mark.parametrize(
    ("confirm_case_runs", "confirm_max_seconds"),
    [(47, 59_400), (48, 59_399)],
)
def test_execute_requires_exact_run_and_time_confirmation_before_runtime_access(
    tmp_path, confirm_case_runs, confirm_max_seconds
):
    receipt = compare.build_plan(
        run_id="confirmation-check",
        models=list(compare.MAC_MODEL_DIGESTS),
        modes=["answer", "deep"],
        repeat=3,
        reasoning="off",
        embed_model=None,
    )
    output_path = tmp_path / "receipt.json"

    with pytest.raises(compare.CompareConfigError, match="最大時間"):
        compare.execute(
            receipt,
            output_path,
            confirm_case_runs=confirm_case_runs,
            confirm_max_seconds=confirm_max_seconds,
            reasoning="off",
            embed_model=None,
        )

    assert not output_path.exists()


def test_deep_runner_grades_only_the_requested_question(monkeypatch):
    import types

    spec = compare.coverage_evaluator.load_spec(compare.DEEP_SPEC)
    monkeypatch.setattr(compare, "loaded_state", lambda _model: "warm")
    result = types.SimpleNamespace(
        status="completed",
        stop_reason="scope_processed",
        answer_state="generated",
        diagnostics={},
        to_public_dict=lambda: {
            "status": "completed",
            "answer": "提供資料には記載がありません。判断できません。",
            "evidence": [],
        },
    )
    monkeypatch.setattr(compare, "run_deep_research", lambda *a, **kw: result)
    report = compare.run_deep_case(
        "qwen3.5:122b",
        spec["items"][3],
        1,
        spec,
        reasoning="off",
        embed_model=None,
        embed_cache_path=None,
        embed_cache_sha256=None,
    )
    assert report["machineReview"]["itemCoverage"] == 1.0
    assert report["machineReview"]["criticalMissIds"] == []
    assert report["machineReview"]["status"] == "NEEDS_REVIEW"
    checks = report["machineReview"]["itemChecks"][0]
    assert checks["missingRequiredTermCount"] == 0
    assert checks["abstentionReported"] is True
    assert "missing_terms" not in checks
    assert "answer" not in checks

    assert report["questionSpecSha256"] != report["parentSpecSha256"]
    assert compare.question_spec(spec, "MH03")["spec_sha256"] != report["questionSpecSha256"]


def test_question_scoped_review_is_bound_to_the_exact_question():
    import copy

    spec = compare.coverage_evaluator.load_spec(compare.DEEP_SPEC)
    scoped = compare.question_spec(spec, "MH04")
    changed = copy.deepcopy(spec)
    changed["items"][3]["abstention_patterns"] = ["changed"]
    assert compare.question_spec(changed, "MH04")["spec_sha256"] != scoped["spec_sha256"]
    with pytest.raises(compare.CompareConfigError):
        compare.question_spec(spec, "unknown")


def test_snapshot_copies_transitive_source_and_preserves_corpus(tmp_path):
    import subprocess
    import sys

    plan = compare.build_plan(
        run_id="snapshot",
        models=list(compare.MAC_MODEL_DIGESTS),
        modes=["answer", "deep"],
        repeat=1,
        reasoning="off",
        embed_model=None,
    )
    assert "_internal/config.py" in plan["product"]["files"]
    assert "tests/eval/diversion_evaluator.py" in plan["product"]["files"]
    compare.copy_snapshot(plan, tmp_path)
    compare.verify_snapshot(plan, tmp_path)
    assert not (tmp_path / "_internal/.model").exists()
    assert not (tmp_path / "skill-source").exists()
    # A fresh interpreter must resolve all modules from the copied tree.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            "-B",
            str(tmp_path / "tests/eval/macos_large_model_compare.py"),
            "--help",
        ],
        capture_output=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    (tmp_path / "_internal/config.py").write_text("# altered\n")
    with pytest.raises(compare.CompareConfigError, match="変更"):
        compare.verify_snapshot(plan, tmp_path)


def test_snapshot_refuses_changed_source_and_out_of_root_paths(tmp_path):
    import hashlib

    source = tmp_path / "source"
    source.mkdir()
    (source / "file.py").write_bytes(b"before")
    plan = {"product": {"files": {"file.py": hashlib.sha256(b"before").hexdigest()}}}
    (source / "file.py").write_bytes(b"after")
    with pytest.raises(compare.CompareConfigError, match="変更"):
        compare.copy_snapshot(plan, tmp_path / "copy", source=source)
    plan["product"]["files"] = {"../outside.py": "0" * 64}
    with pytest.raises(compare.CompareConfigError, match="path"):
        compare.copy_snapshot(plan, tmp_path / "copy", source=source)


def test_full_hybrid_plan_for_used_group_is_not_independent_acceptance():
    plan = compare.build_plan(
        run_id="calibration",
        models=list(compare.MAC_MODEL_DIGESTS),
        modes=["answer", "deep"],
        repeat=3,
        reasoning="off",
        embed_model="bge-m3",
    )
    assert plan["acceptanceEligible"] is False
    assert plan["acceptanceIneligibleReason"] == "evaluation-group-used-for-remediation"
    plan["acceptanceEligible"] = True
    plan["evaluationSplit"] = "independent-synthetic-holdout-only"
    plan["holdout"]["status"] = "FIXED_SYNTHETIC_NOT_RUN"
    compare.mark_calibration(plan)
    assert plan["acceptanceEligible"] is False
    assert plan["holdout"]["status"] == "CALIBRATION_USED_FOR_REMEDIATION"
