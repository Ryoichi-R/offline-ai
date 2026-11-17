import copy
import hashlib
import json

import pytest
import material_review as review


@pytest.fixture
def sample(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = "# 条件\n原文限定 SENTINEL_SOURCE\n".encode()
    (source / "a.md").write_bytes(data)
    result = {
        "status": "partial",
        "stop_reason": "unresolved_scope",
        "answer_state": "generated",
        "answer": "SENTINEL_ANSWER [E1]",
        "unconfirmed": ["SENTINEL_NOTE"],
        "evidence": [
            {
                "evidence_id": "E1",
                "path": "a.md",
                "source_sha256": hashlib.sha256(data).hexdigest(),
                "start_line": 2,
                "end_line": 2,
                "excerpt": "原文限定 SENTINEL_SOURCE",
                "verification_status": "raw_unverified",
            }
        ],
    }
    return source, tmp_path / "output", result


def export(sample):
    source, output, result = sample
    return review.export_review_bundle(
        query="SENTINEL_QUERY",
        result=result,
        source_root=source,
        destination=output,
        model_digest="a" * 64,
    )


def test_bundle_preserves_review_material_without_receipt_text(sample):
    source, output, result = sample
    before = (source / "a.md").read_bytes()
    receipt = export(sample)
    assert receipt["semanticReview"] == "NOT_REVIEWED"
    assert receipt["acceptanceEligible"] is False
    assert receipt["sourceVerification"] == "MATCH"
    assert "SENTINEL" not in json.dumps(receipt)
    assert "SENTINEL_ANSWER" in (output / "answer.md").read_text(encoding="utf-8")
    assert "SENTINEL_SOURCE" in (output / "review.md").read_text(encoding="utf-8")
    assert "SENTINEL_NOTE" in (output / "review.md").read_text(encoding="utf-8")
    assert "raw_unverified" in (output / "review.md").read_text(encoding="utf-8")
    assert (source / "a.md").read_bytes() == before
    assert (
        receipt["answerMarkdownSha256"]
        == hashlib.sha256((output / "answer.md").read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        export(sample)


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "../outside.md"),
        ("path", "/tmp/outside.md"),
        ("excerpt", "forged"),
        ("source_sha256", "0" * 64),
        ("end_line", 999),
        ("start_line", True),
    ],
)
def test_bad_source_reference_never_gets_a_completed_bundle(sample, field, value):
    sample[2]["evidence"][0][field] = value
    with pytest.raises(ValueError):
        export(sample)
    assert not sample[1].exists()


def test_symlink_cannot_read_outside_source(sample, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("原文限定 SENTINEL_SOURCE", encoding="utf-8")
    link = sample[0] / "link.md"
    try:
        link.symlink_to(outside)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows runner cannot create symbolic links")
        raise
    sample[2]["evidence"][0]["path"] = "link.md"
    with pytest.raises(ValueError, match="escaped"):
        export(sample)


def test_output_inside_source_is_rejected(sample):
    with pytest.raises(ValueError, match="separate"):
        export((sample[0], sample[0] / "output", sample[2]))


def test_character_slice_is_verified_and_bad_boundaries_rejected(sample):
    row = sample[2]["evidence"][0]
    row.update(char_start=1, char_end=3, excerpt=row["excerpt"][1:4])
    assert export(sample)["evidenceCount"] == 1
    row["char_end"] = 100
    with pytest.raises(ValueError):
        export((sample[0], sample[1].with_name("other"), sample[2]))


def test_unknown_citations_are_preserved_for_human_review_not_passed(sample):
    sample[2]["answer"] += " [E99]"
    receipt = export(sample)
    assert receipt["unknownCitationCount"] == 1
    assert receipt["acceptanceEligible"] is False


def test_duplicate_ids_are_rejected_before_writing(sample):
    sample[2]["evidence"].append(copy.deepcopy(sample[2]["evidence"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        export(sample)
    assert not sample[1].exists()


def test_context_is_verified_and_exported(sample):
    row = sample[2]["evidence"][0]
    row["context"] = [{"path": "a.md", "start_line": 1, "end_line": 1, "text": "# 条件"}]
    export(sample)
    assert "補助文脈 L1-1" in (sample[1] / "review.md").read_text(encoding="utf-8")
    row["context"][0]["text"] = "forged parent condition"
    with pytest.raises(ValueError, match="mismatch"):
        export((sample[0], sample[1].with_name("other"), sample[2]))


@pytest.mark.parametrize("path", [".env", "secrets.json", "hidden.md"])
def test_sensitive_reference_is_rejected_before_read(sample, path):
    row = sample[2]["evidence"][0]
    row["path"] = path
    if path == "hidden.md":
        (sample[0] / "hidden.md").symlink_to(sample[0] / "secrets.json")
        (sample[0] / "secrets.json").write_text("fixture only")
    with pytest.raises(ValueError, match="excluded"):
        export(sample)


# --- 回答品質の4軸採点（回答方針v2） -------------------------------------------------

import material_review as _review  # noqa: E402


def axes(a=2, b=2, c=2, d=2):
    return dict(zip(_review.QUALITY_AXES, (a, b, c, d)))


def test_axis_scores_reject_wrong_keys_bools_and_out_of_range():
    for bad in (
        None,
        {},
        {**axes(), "speed": 2},
        axes(a=3),
        axes(a=True),
        axes(a=1.0),
        axes(a="2"),
    ):
        with pytest.raises(ValueError):
            _review.validate_axis_scores(bad)
    assert _review.validate_axis_scores(axes()) == axes()


def test_acceptance_verdict_never_passes_without_the_human_column_and_never_averages():
    assert _review.acceptance_verdict(None) == "NOT_SCORED"
    assert _review.acceptance_verdict(axes()) == "PASS_PENDING_HUMAN"
    assert _review.acceptance_verdict(axes(), axes()) == "PASS"
    # 一軸でも0は不合格。他の軸が満点でも、速度など別の良さでも相殺しない。
    assert _review.acceptance_verdict(axes(a=0)) == "FAIL"
    assert _review.acceptance_verdict(axes(), axes(d=0)) == "FAIL"
    assert _review.acceptance_verdict(axes(b=1)) == "NEEDS_REVISION"
    assert _review.acceptance_verdict(axes(), axes(c=1)) == "NEEDS_REVISION"
    # 引用・意味の重大な誤りは、全軸2でも不合格。
    assert _review.acceptance_verdict(axes(), axes(), hard_conditions_met=False) == "FAIL"


def test_review_bundle_records_quality_states_and_leaves_axes_unscored(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.md").write_text("原文の一行目\n原文の二行目", encoding="utf-8")
    import hashlib as _hashlib

    sha = _hashlib.sha256((source / "a.md").read_bytes()).hexdigest()
    result = {
        "status": "partial",
        "stop_reason": "scope_processed",
        "answer_state": "generated",
        "answer": "回答。[E1]",
        "unconfirmed": [],
        "evidence": [
            {
                "evidence_id": "E1",
                "path": "a.md",
                "source_sha256": sha,
                "start_line": 1,
                "end_line": 2,
                "excerpt": "原文の一行目\n原文の二行目",
                "verification_status": "semantic_checked",
            }
        ],
        "answer_quality": {
            "state": "needs_review",
            "removedClaims": 1,
            "unconfirmedAspects": ["観点A"],
            "gates": {g: {"status": "passed"} for g in ("G1", "G2", "G3")}
            | {"G4": {"status": "not_run"}},
        },
    }
    receipt = _review.export_review_bundle(
        query="質問",
        result=result,
        source_root=source,
        destination=tmp_path / "out",
        model_digest="a" * 64,
    )
    assert receipt["qualityAxes"] == "NOT_SCORED" and receipt["acceptanceEligible"] is False
    assert receipt["answerQuality"] == {
        "state": "needs_review",
        "gates": {"G1": "passed", "G2": "passed", "G3": "passed", "G4": "not_run"},
    }
    review = (tmp_path / "out" / "review.md").read_text(encoding="utf-8")
    assert "自動品質検査（暫定" in review and "| 意味支持 | 未採点 | 未確認 |" in review
    # 回答本文は受領票へ入れない。
    assert "回答。" not in json.dumps(receipt, ensure_ascii=False)
