"""P2 集計スクリプト（``p2_aggregate.py``）の単体テスト。"""

from __future__ import annotations

import p2_aggregate as agg


def _receipt(verify: str = "on") -> dict:
    def q(qid, category, layer, pre, post, vstatus, vlat):
        return {
            "question_id": qid,
            "category": category,
            "abstain_layer": layer,
            "status": "completed",
            "evidence_status": post,
            "pre_verification_status": pre,
            "verification_status": vstatus,
            "verification_latency_ms": vlat,
        }

    def probe(qid, run, pre, post, vstatus, **extra):
        base = {
            "question_id": qid,
            "run_id": f"{qid}:agentic-lite:{run}",
            "status": "measured",
            "retrieval_status": post,
            "pre_verification_status": pre,
            "verification_status": vstatus,
            "verification_latency_ms": 2000.0,
            "retrieval_latency_ms": 10000.0,
            "answer_latency_ms": 5000.0,
        }
        base.update(extra)
        return base

    return {
        "environment": {
            "chat_model": "gpt-oss:20b",
            "repeat": 2,
            "routes_requested": ["agentic-lite"],
            "evidence_verify": verify,
            "use_q1": True,
        },
        "spec": {"sha256": "abcdef1234567890"},
        "routes": {
            "agentic-lite": {
                "runs": [
                    {
                        "questions": [
                            q("T01", "near-miss", "answer", "sufficient", "partial", "verified", 3000.0),
                            q("T05", "synonym", "", "partial", "partial", "failed", 15000.0),
                        ]
                    },
                    {
                        "questions": [
                            q("T01", "near-miss", "answer", "sufficient", "sufficient", "skipped_budget", 0.0),
                            q("T05", "synonym", "", "insufficient", "insufficient", "skipped_not_applicable", 0.0),
                        ]
                    },
                ]
            }
        },
        "answer_probes": {
            "agentic-lite": [
                probe("T01", 1, "sufficient", "partial", "verified", passed=False,
                      diversion_verdict="FAIL", forbidden_fact=True),
                probe("T01", 2, "sufficient", "partial", "verified", passed=True,
                      diversion_verdict="PASS", forbidden_fact=False),
            ]
        },
        "answer_probes_same_evidence": {
            "agentic-lite": [
                probe("T01", 1, "sufficient", "sufficient", "verified", passed=True),
                probe("T01", 2, "sufficient", "sufficient", "verified", passed=True),
            ]
        },
    }


def test_retrieval_samples_count_same_evidence_retrieval_once():
    samples = agg.retrieval_samples(_receipt())
    sources = [s["source"] for s in samples]
    assert sources.count("retrieval_run") == 4
    assert sources.count("answer_probes") == 2
    assert sources.count("answer_probes_same_evidence") == 1


def test_verification_summary_by_stratum_excludes_unexecuted_latency():
    summary = agg.verification_summary(agg.retrieval_samples(_receipt()))
    suff = summary["sufficient"]
    # retrieval_run 2 + 通し 2 + 同一根拠 1
    assert suff["targets"] == 5
    assert suff["counts"] == {"verified": 4, "skipped_budget": 1}
    assert suff["completion_rate"] == 0.8
    # skipped_budget の 0ms は遅延に含めない
    assert suff["latency_median_ms"] == 2000.0
    assert suff["downgraded"] == 3
    part = summary["partial"]
    assert part["targets"] == 1
    assert part["completion_rate"] == 0.0
    assert part["latency_p95_ms"] == 15000.0


def test_false_sufficient_only_for_no_answer_questions():
    receipt = _receipt()
    result = agg.false_sufficient(agg.retrieval_samples(receipt), agg._question_meta(receipt))
    assert set(result) == {"T01"}
    assert result["T01"]["trials"] == 5
    assert result["T01"]["sufficient"] == 2


def test_probe_table_counts_diversion_and_total_latency():
    table = agg.probe_table(_receipt(), "answer_probes")
    entry = table["T01"]
    assert entry["measured"] == 2
    assert entry["passed"] == 1
    assert (entry["diversion_fail"], entry["diversion_measured"]) == (1, 2)
    assert entry["forbidden_fact"] == 1
    assert entry["total_latency_ms"] == [15000.0, 15000.0]


def test_build_markdown_labels_patterns():
    markdown = agg.build_markdown([("on.json", _receipt("on")), ("off.json", _receipt("off"))])
    assert "agentic-lite / 案AON / Q-1後" in markdown
    assert "agentic-lite / 案AOFF / Q-1後" in markdown
    assert "## 4. 検証完了率" in markdown


def test_percentile_nearest_rank():
    assert agg._percentile([], 95) is None
    assert agg._percentile([1.0], 95) == 1.0
    assert agg._percentile([float(v) for v in range(1, 21)], 95) == 19.0


def test_verification_summary_breaks_down_failure_reasons():
    samples = [
        {"pre": "partial", "post": "partial", "verification_status": "failed",
         "verification_latency_ms": 900.0, "failure_reason": "contradiction"},
        {"pre": "partial", "post": "partial", "verification_status": "failed",
         "verification_latency_ms": 15000.0, "failure_reason": "timeout"},
        {"pre": "partial", "post": "partial", "verification_status": "verified",
         "verification_latency_ms": 4000.0, "failure_reason": ""},
    ]
    summary = agg.verification_summary(samples)["partial"]
    assert summary["counts"] == {
        "failed/contradiction": 1,
        "failed/timeout": 1,
        "verified": 1,
    }
    assert summary["completion_rate"] == 1 / 3
