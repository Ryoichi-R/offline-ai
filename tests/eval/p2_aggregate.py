"""P2: 案A（LLM根拠検証）× Q-1 の複数 receipt を受入基準の観点で集計する。

``run_eval.py`` が出力した receipt JSON（パターンごとに1件）を読み、計画の
受入基準（誤った sufficient 率、流用率、正例の新規不合格、検証完了率、遅延）
を質問単位・型別・検索時点の層別に Markdown で出す。判定そのもの（許容件数の
決定など）は利用者判断であり、このスクリプトは数値と一覧だけを出す。

使い方::

    python tests/eval/p2_aggregate.py <receipt.json> [<receipt.json> ...] [--out summary.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

VERIFY_TARGET_STRATA = ("sufficient", "partial")
LATENCY_STATUSES = ("verified", "failed")


def pattern_label(receipt: dict) -> str:
    env = receipt.get("environment", {})
    routes = "+".join(env.get("routes_requested", []))
    verify = "案A" + ("ON" if env.get("evidence_verify") == "on" else "OFF")
    q1 = "Q-1後" if env.get("use_q1", True) else "Q-1前"
    return f"{routes} / {verify} / {q1}"


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # 最近傍順位法（小標本で補間値を作らない）。
    rank = max(1, -(-len(ordered) * pct // 100))
    return ordered[int(rank) - 1]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt_seconds(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.2f}s"


def _question_meta(receipt: dict) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for route in receipt.get("routes", {}).values():
        for run in route.get("runs", []):
            for q in run.get("questions", []):
                meta.setdefault(
                    q["question_id"],
                    {"category": q.get("category", ""), "abstain_layer": q.get("abstain_layer", "")},
                )
    return meta


def retrieval_samples(receipt: dict) -> list[dict]:
    """検索を実際に行った試行（検証の対象になり得る）を重複なく返す。

    - 検索評価の run（routes.*.runs）
    - 通しの比較の probe（検索を毎回やり直す）
    - 同一根拠比較の probe は検索1回を repeat 回使い回すため run 1 だけ数える
    """
    samples: list[dict] = []
    for route_name, route in receipt.get("routes", {}).items():
        for run in route.get("runs", []):
            for q in run.get("questions", []):
                if q.get("status") != "completed":
                    continue
                samples.append(
                    {
                        "source": "retrieval_run",
                        "route": route_name,
                        "question_id": q["question_id"],
                        "pre": q.get("pre_verification_status") or q.get("evidence_status", ""),
                        "post": q.get("evidence_status", ""),
                        "verification_status": q.get("verification_status", ""),
                        "verification_latency_ms": q.get("verification_latency_ms", 0.0),
                        "failure_reason": q.get("verification_failure_reason", ""),
                    }
                )
    for key, same in (
        ("answer_probes", False),
        ("answer_quality_probes", False),
        ("answer_probes_same_evidence", True),
        ("answer_quality_probes_same_evidence", True),
    ):
        for route_name, probes in (receipt.get(key) or {}).items():
            for p in probes:
                if same and not str(p.get("run_id", "")).endswith(":1"):
                    continue
                if "pre_verification_status" not in p:
                    continue
                samples.append(
                    {
                        "source": key,
                        "route": route_name,
                        "question_id": p["question_id"],
                        "pre": p.get("pre_verification_status", ""),
                        "post": p.get("retrieval_status", ""),
                        "verification_status": p.get("verification_status", ""),
                        "verification_latency_ms": p.get("verification_latency_ms", 0.0),
                        "failure_reason": p.get("verification_failure_reason", ""),
                    }
                )
    return samples


def verification_summary(samples: list[dict]) -> dict[str, dict]:
    """検索時点の層（sufficient/partial）別の検証完了率と検証単体遅延。"""
    result: dict[str, dict] = {}
    for stratum in VERIFY_TARGET_STRATA:
        targets = [s for s in samples if s["pre"] == stratum]
        counts: dict[str, int] = defaultdict(int)
        for s in targets:
            key = s["verification_status"] or "(none)"
            if key == "failed" and s.get("failure_reason"):
                key = f"failed/{s['failure_reason']}"
            counts[key] += 1
        latencies = [
            float(s["verification_latency_ms"])
            for s in targets
            if s["verification_status"] in LATENCY_STATUSES
        ]
        verified = counts.get("verified", 0)
        result[stratum] = {
            "targets": len(targets),
            "counts": dict(counts),
            "completion_rate": (verified / len(targets)) if targets else None,
            "latency_median_ms": _median(latencies),
            "latency_p95_ms": _percentile(latencies, 95),
            "downgraded": sum(1 for s in targets if s["pre"] != s["post"]),
        }
    return result


def false_sufficient(samples: list[dict], meta: dict[str, dict]) -> dict[str, dict]:
    """該当なし質問の試行のうち、外部の根拠ステータスが sufficient の割合（質問単位）。"""
    per_q: dict[str, dict] = {}
    for s in samples:
        info = meta.get(s["question_id"], {})
        if not info.get("abstain_layer"):
            continue
        entry = per_q.setdefault(
            s["question_id"], {"category": info.get("category", ""), "trials": 0, "sufficient": 0}
        )
        entry["trials"] += 1
        entry["sufficient"] += int(s["post"] == "sufficient")
    return per_q


def probe_table(receipt: dict, key: str) -> dict[str, dict]:
    """answer probe の質問単位集計（合格・流用・禁止事実・遅延）。"""
    per_q: dict[str, dict] = {}
    for probes in (receipt.get(key) or {}).values():
        for p in probes:
            if p.get("status") == "not_measured":
                continue
            entry = per_q.setdefault(
                p["question_id"],
                {
                    "measured": 0,
                    "passed": 0,
                    "diversion_fail": 0,
                    "diversion_measured": 0,
                    "forbidden_fact": 0,
                    "errors": 0,
                    "total_latency_ms": [],
                    "post_status": defaultdict(int),
                },
            )
            entry["measured"] += 1
            entry["passed"] += int(p.get("passed") is True)
            entry["errors"] += int(p.get("status") != "measured")
            entry["forbidden_fact"] += int(p.get("forbidden_fact") is True)
            if "diversion_verdict" in p:
                entry["diversion_measured"] += 1
                entry["diversion_fail"] += int(p["diversion_verdict"] == "FAIL")
            entry["post_status"][p.get("retrieval_status", "")] += 1
            if "answer_latency_ms" in p:
                entry["total_latency_ms"].append(
                    float(p.get("retrieval_latency_ms", 0.0)) + float(p["answer_latency_ms"])
                )
    return per_q


def _status_text(counter: dict[str, int]) -> str:
    return ", ".join(f"{k or '-'}:{v}" for k, v in sorted(counter.items()))


def build_markdown(receipts: list[tuple[str, dict]]) -> str:
    lines = ["# P2 集計（案A × Q-1）", ""]
    lines.append("| パターン | receipt | chat model | repeat | spec sha256 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, r in receipts:
        env = r.get("environment", {})
        lines.append(
            f"| {pattern_label(r)} | {name} | {env.get('chat_model', '')} | "
            f"{env.get('repeat', '')} | {r.get('spec', {}).get('sha256', '')[:12]} |"
        )
    lines.append("")

    lines += ["## 1. 誤った sufficient（該当なし質問、検索を行った全試行）", ""]
    lines.append("| パターン | 質問 | 型 | sufficient / 試行 |")
    lines.append("| --- | --- | --- | --- |")
    for _, r in receipts:
        fs = false_sufficient(retrieval_samples(r), _question_meta(r))
        for qid in sorted(fs):
            e = fs[qid]
            lines.append(f"| {pattern_label(r)} | {qid} | {e['category']} | {e['sufficient']}/{e['trials']} |")
    lines.append("")

    for key, heading in (
        ("answer_probes", "2a. 該当なし質問の回答（通しの比較）"),
        ("answer_probes_same_evidence", "2b. 該当なし質問の回答（同一根拠比較）"),
        ("answer_quality_probes", "3a. 正例の回答品質（通しの比較）"),
        ("answer_quality_probes_same_evidence", "3b. 正例の回答品質（同一根拠比較）"),
    ):
        lines += [f"## {heading}", ""]
        lines.append(
            "| パターン | 質問 | 型 | 合格 | 流用FAIL(評価器) | 禁止事実一致 | error | 回答時の状態 | 回答完了 中央値 | p95 |"
        )
        lines.append("|" + " --- |" * 10)
        for _, r in receipts:
            meta = _question_meta(r)
            table = probe_table(r, key)
            for qid in sorted(table):
                e = table[qid]
                div = (
                    f"{e['diversion_fail']}/{e['diversion_measured']}"
                    if e["diversion_measured"]
                    else "-"
                )
                lines.append(
                    f"| {pattern_label(r)} | {qid} | {meta.get(qid, {}).get('category', '')} | "
                    f"{e['passed']}/{e['measured']} | {div} | {e['forbidden_fact']}/{e['measured']} | "
                    f"{e['errors']} | {_status_text(e['post_status'])} | "
                    f"{_fmt_seconds(_median(e['total_latency_ms']))} | "
                    f"{_fmt_seconds(_percentile(e['total_latency_ms'], 95))} |"
                )
        lines.append("")

    lines += ["## 4. 検証完了率と検証単体の遅延（検索時点の層別）", ""]
    lines.append(
        "| パターン | 層 | 対象件数 | 内訳 | 完了率 | 格下げ件数 | 検証遅延 中央値 | p95 |"
    )
    lines.append("|" + " --- |" * 8)
    for _, r in receipts:
        summary = verification_summary(retrieval_samples(r))
        for stratum, s in summary.items():
            rate = "n/a" if s["completion_rate"] is None else f"{s['completion_rate']:.1%}"
            lines.append(
                f"| {pattern_label(r)} | {stratum} | {s['targets']} | {_status_text(s['counts'])} | "
                f"{rate} | {s['downgraded']} | {_fmt_seconds(s['latency_median_ms'])} | "
                f"{_fmt_seconds(s['latency_p95_ms'])} |"
            )
    lines.append("")

    lines += ["## 5. 回答完了までの遅延（通しの比較、検索＋回答生成）", ""]
    lines.append("| パターン | 試行数 | 中央値 | p95 |")
    lines.append("| --- | --- | --- | --- |")
    for _, r in receipts:
        totals: list[float] = []
        for key in ("answer_probes", "answer_quality_probes"):
            for e in probe_table(r, key).values():
                totals += e["total_latency_ms"]
        lines.append(
            f"| {pattern_label(r)} | {len(totals)} | {_fmt_seconds(_median(totals))} | "
            f"{_fmt_seconds(_percentile(totals, 95))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    receipts = [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in args.receipts]
    markdown = build_markdown(receipts)
    if args.out:
        args.out.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
