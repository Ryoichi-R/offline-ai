"""offline-ai Phase 2 評価ランナー。

固定 corpus と固定質問で keyword / hybrid / agentic-lite の3経路を測定し、
candidate へ束縛できる redacted receipt（JSON + Markdown）を出力する。

使用例:

    python tests/eval/run_eval.py                      # keyword のみ（Ollama不要）
    python tests/eval/run_eval.py --routes all         # Ollama がある環境で全経路
    python tests/eval/run_eval.py --routes all --repeat 3 --chat-model qwen3.5:9b

出力は既定で `<repository>/.test-results/offline-ai-eval/` へ書く。receipt には資料本文、
生成回答、絶対 path を含めない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

OFFLINE_AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(OFFLINE_AI_ROOT / "_internal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import search  # noqa: E402  - sys.path 設定後に読み込む
import eval_harness as harness  # noqa: E402

HARNESS_VERSION = "1.4.0"
DEFAULT_SPEC = Path(__file__).resolve().parent / "eval-spec.json"
DEFAULT_OUT_DIR = OFFLINE_AI_ROOT / ".test-results" / "offline-ai-eval"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _corpus_inventory(corpus_dir: Path) -> list[dict]:
    inventory = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        inventory.append(
            {
                "path": path.relative_to(corpus_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return inventory


def _product_baseline() -> dict:
    version_path = OFFLINE_AI_ROOT / "VERSION"
    return {
        "product_version": version_path.read_text(encoding="utf-8").strip()
        if version_path.exists()
        else "",
        "search_py_sha256": _sha256_file(OFFLINE_AI_ROOT / "_internal" / "search.py"),
        "prompt_templates_sha256": _sha256_file(
            OFFLINE_AI_ROOT / "_internal" / "prompt_templates.py"
        ),
        "retrieval_config": {
            "chunk_max_chars": search.CHUNK_MAX_CHARS,
            "chunk_overlap_chars": search.CHUNK_OVERLAP_CHARS,
            "retrieval_candidate_limit": search.RETRIEVAL_CANDIDATE_LIMIT,
            "retrieval_prompt_match_limit": search.RETRIEVAL_PROMPT_MATCH_LIMIT,
            "max_chunks_per_file": search.MAX_CHUNKS_PER_FILE,
            "max_query_variants": search.MAX_QUERY_VARIANTS,
            "max_retrieval_attempts": search.MAX_RETRIEVAL_ATTEMPTS,
            "min_rrf_score": search.SEARCH_MIN_RRF_SCORE,
            "embed_sim_threshold": search.EMBED_SIM_THRESHOLD,
            "rerank_mode": search.RERANK_CONFIG.mode,
        },
    }


def _resolve_models(args: argparse.Namespace) -> tuple[str | None, str | None, dict]:
    """chat / embed model の利用可否を判定する。到達不能なら None を返す。"""
    diagnostics: dict = {"chat": "disabled", "embed": "disabled"}
    if args.no_ollama:
        return None, None, diagnostics

    chat_model = args.chat_model or search.detect_model()
    if chat_model:
        status = search._is_model_available(chat_model)
        diagnostics["chat"] = status.value
        if status is not search.ModelStatus.AVAILABLE:
            chat_model = None
    else:
        diagnostics["chat"] = "not-configured"

    embed_model = args.embed_model or search.detect_embed_model()
    if embed_model:
        status = search._is_model_available(embed_model)
        diagnostics["embed"] = status.value
        if status is not search.ModelStatus.AVAILABLE:
            embed_model = None
    else:
        diagnostics["embed"] = "not-configured"

    return chat_model, embed_model, diagnostics


def _score_to_dict(score: harness.QuestionScore) -> dict:
    return {
        "question_id": score.question_id,
        "category": score.category,
        "status": score.status,
        "passed": score.passed,
        "retrieval_hit": score.retrieval_hit,
        "hit_at_1": score.hit_at_1,
        "evidence_coverage": score.evidence_coverage,
        "evidence_precision": score.evidence_precision,
        "evidence_line_overlap": score.evidence_line_overlap,
        "expected_lines_retained": score.expected_lines_retained,
        "candidate_line_recall": score.candidate_line_recall,
        "expanded_irrelevant_ranges": score.expanded_irrelevant_ranges,
        "expansion_expectation_met": score.expansion_expectation_met,
        "holdout": score.holdout,
        "forbidden_source_hits": score.forbidden_source_hits,
        "abstain_correct": score.abstain_correct,
        "abstain_layer": score.abstain_layer,
        "evidence_status": score.evidence_status,
        "confidence": score.confidence,
        "latency_ms": score.latency_ms,
        "attempts": score.attempts,
        "returned_count": score.returned_count,
        "reason": score.reason,
        "evidence": score.matches,
    }


def _routes_to_dict(route_reports: dict) -> dict:
    serialized = {}
    for route, report in route_reports.items():
        serialized[route] = {
            "summary": report["summary"],
            "stability": report["stability"],
            "acceptance": report["acceptance"],
            "runs": [
                {
                    "run": run["run"],
                    "summary": run["summary"],
                    "questions": [_score_to_dict(s) for s in run["scores"]],
                }
                for run in report["runs"]
            ],
        }
    return serialized


def _format_metric(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_markdown(receipt: dict) -> str:
    lines: list[str] = []
    lines.append("# offline-ai Phase 2 検索品質評価 receipt")
    lines.append("")
    lines.append(f"- 開始日時 (UTC): {receipt.get('started_at', '未記録')}")
    lines.append(f"- 終了日時 (UTC): {receipt['generated_at']}")
    lines.append(f"- harness version: {receipt['harness_version']}")
    lines.append(f"- 製品version: {receipt['product']['product_version']}")
    lines.append(f"- `search.py` SHA-256: `{receipt['product']['search_py_sha256']}`")
    lines.append(
        f"- `prompt_templates.py` SHA-256: `{receipt['product']['prompt_templates_sha256']}`"
    )
    lines.append(f"- 評価仕様 SHA-256: `{receipt['spec']['sha256']}`")
    lines.append(f"- corpus file数: {len(receipt['corpus']['files'])}")
    lines.append(f"- 総合判定: **{receipt['verdict']}**")
    lines.append("")
    lines.append("> receipt には資料本文・生成回答・絶対pathを含めない。")
    lines.append("> `skipped` / `NOT_MEASURED` はPASSではない。")
    lines.append("")

    lines.append("## 経路別サマリ")
    lines.append("")
    lines.append(
        "| route | 判定 | 実行/skip | hit率 | hit@1 | coverage | precision | line一致 | 該当情報なし正答 | 禁止source | attempts中央値 | 中央latency(ms) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for route, report in receipt["routes"].items():
        s = report["summary"]
        lines.append(
            "| {route} | {verdict} | {done}/{skip} | {hit} | {hit1} | {cov} | {prec} | {line} | {abstain} | {forb} | {attempts} | {lat} |".format(
                route=route,
                verdict=report["acceptance"]["verdict"],
                done=s["completed"],
                skip=s["skipped"],
                hit=_format_metric(s["retrieval_hit_rate"]),
                hit1=_format_metric(s["hit_at_1_rate"]),
                cov=_format_metric(s["evidence_coverage_mean"]),
                prec=_format_metric(s["evidence_precision_mean"]),
                line=_format_metric(s["evidence_line_overlap_mean"]),
                abstain=_format_metric(s["abstain_accuracy"]),
                forb=s["forbidden_source_hits"],
                attempts=_format_metric(s["attempts_median"]),
                lat=_format_metric(s["latency_ms_median"]),
            )
        )
    lines.append("")

    for route, report in receipt["routes"].items():
        lines.append(f"## route: {route}")
        lines.append("")
        lines.append("### 受入基準")
        lines.append("")
        lines.append("| 基準 | 閾値 | 実測 | 結果 |")
        lines.append("| --- | --- | --- | --- |")
        for check in report["acceptance"]["checks"]:
            lines.append(
                f"| {check['criterion']} | {check['threshold']} | {_format_metric(check['actual'])} | {check['result']} |"
            )
        lines.append("")
        summary = report["summary"]
        lines.append(
            "- 保留質問: {total}件中 FAIL {failed}件 / 候補recall平均: {recall} / 無関係な展開範囲: {irrelevant}件".format(
                total=summary.get("holdout_total", 0),
                failed=summary.get("holdout_failed", 0),
                recall=_format_metric(summary.get("candidate_line_recall_mean")),
                irrelevant=summary.get("expanded_irrelevant_ranges", 0),
            )
        )
        lines.append("")
        first_run = report["runs"][0]
        lines.append("### 質問別（run 1）")
        lines.append("")
        lines.append(
            "| ID | 区分 | 保留 | 状態 | 判定 | hit | coverage | precision | line一致 | 必要範囲 | 候補recall | 無関係展開 | 展開期待 | evidence_status | attempts | latency(ms) |"
        )
        lines.append("|" + " --- |" * 16)
        for question in first_run["questions"]:
            verdict = (
                "-" if question["passed"] is None else ("PASS" if question["passed"] else "FAIL")
            )
            lines.append(
                "| {qid} | {cat} | {holdout} | {st} | {v} | {hit} | {cov} | {prec} | {line} | {retained} | {recall} | {irrelevant} | {expansion} | {ev} | {attempts} | {lat} |".format(
                    qid=question["question_id"],
                    cat=question["category"],
                    holdout="yes" if question.get("holdout") else "-",
                    recall=_format_metric(question.get("candidate_line_recall")),
                    irrelevant=_format_metric(question.get("expanded_irrelevant_ranges")),
                    expansion=_format_metric(question.get("expansion_expectation_met")),
                    st=question["status"],
                    v=verdict,
                    hit=_format_metric(question["retrieval_hit"]),
                    cov=_format_metric(question["evidence_coverage"]),
                    prec=_format_metric(question["evidence_precision"]),
                    line=_format_metric(question["evidence_line_overlap"]),
                    retained=_format_metric(question.get("expected_lines_retained")),
                    ev=question["evidence_status"] or "-",
                    attempts=question["attempts"],
                    lat=_format_metric(question["latency_ms"]),
                )
            )
        lines.append("")

    lines.append("## 環境")
    lines.append("")
    env = receipt["environment"]
    lines.append(f"- Python: {env['python']}")
    lines.append(f"- Platform: {env['platform']}")
    lines.append(
        f"- chat model: {env['chat_model'] or '(未使用)'} / 判定: {env['model_diagnostics']['chat']}"
    )
    lines.append(
        f"- embed model: {env['embed_model'] or '(未使用)'} / 判定: {env['model_diagnostics']['embed']}"
    )
    lines.append(f"- repeat: {env['repeat']}")
    lines.append(
        f"- answer probe: {'実施' if receipt.get('answer_probes') is not None else '未実施'}"
    )
    lines.append(
        f"- answer quality probe: {'実施' if receipt.get('answer_quality_probes') is not None else '未実施'}"
    )
    lines.append(
        f"- 親子展開 ON/OFF 比較: {'実施' if receipt.get('expansion_comparison') is not None else '未実施'}"
        f"（本評価の展開設定: {receipt['environment'].get('parent_child_expansion', 'n/a')}）"
    )
    lines.append("")

    if receipt.get("answer_probes") is not None:
        lines.append("## answer 層 probe（本文非保存）")
        lines.append("")
        lines.append(
            "| route | question | 状態 | 判定 | retrieval status | abstain phrase | forbidden fact | transport error |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for route, probes in receipt["answer_probes"].items():
            for probe in probes:
                lines.append(
                    "| {route} | {qid} | {status} | {passed} | {retrieval} | {phrase} | {forbidden} | {transport} |".format(
                        route=route,
                        qid=probe["question_id"],
                        status=probe["status"],
                        passed="PASS"
                        if probe.get("passed") is True
                        else "FAIL"
                        if probe.get("passed") is False
                        else "NOT_MEASURED",
                        retrieval=probe.get("retrieval_status", ""),
                        phrase=probe.get("abstain_phrase", "-"),
                        forbidden=probe.get("forbidden_fact", "-"),
                        transport=probe.get("transport_error", "-"),
                    )
                )
        lines.append("")
    lines.extend(_answer_quality_markdown(receipt))
    lines.extend(_expansion_comparison_markdown(receipt))
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="offline-ai Phase 2 検索品質評価ランナー")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="評価仕様JSON")
    parser.add_argument(
        "--routes",
        default=harness.ROUTE_KEYWORD,
        help=f"カンマ区切りの route、または 'all'。既定: {harness.ROUTE_KEYWORD}（{', '.join(harness.ROUTES)}）",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="非決定性の確認用に各 route を繰り返す回数",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="receipt 出力先")
    parser.add_argument(
        "--chat-model", default="", help="chat model 名（未指定時は .model から検出）"
    )
    parser.add_argument(
        "--embed-model",
        default="",
        help="Embedding model 名（未指定時は .model_embed から検出）",
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Ollama へ一切接続しない（keyword のみ実行可）",
    )
    parser.add_argument("--no-write", action="store_true", help="receipt を書き出さず標準出力のみ")
    parser.add_argument(
        "--measure-answer-layer",
        action="store_true",
        help="answer 層の no-answer probe を本文非保存で実行",
    )
    parser.add_argument(
        "--measure-answer-quality",
        action="store_true",
        help="回答可能な質問（required_facts あり）の回答を本文非保存で検査",
    )
    parser.add_argument(
        "--compare-expansion",
        action="store_true",
        help="親子展開 OFF/ON の両方で評価し、質問ごとの status 遷移・再検索率を比較",
    )
    return parser.parse_args(argv)


def _verdict_label(passed) -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "NOT_MEASURED"


def _answer_quality_markdown(receipt: dict) -> list[str]:
    probes_by_route = receipt.get("answer_quality_probes")
    if probes_by_route is None:
        return []
    lines = ["## 回答品質 probe（回答可能な質問、本文非保存）", ""]
    lines.append(
        "| route | question | 状態 | 判定 | retrieval status | 必要事項欠落 | 期待source引用 | 該当なし回答 | forbidden fact | transport error |"
    )
    lines.append("|" + " --- |" * 10)
    for route, probes in probes_by_route.items():
        for probe in probes:
            lines.append(
                f"| {route} | {probe['question_id']} | {probe['status']} | {_verdict_label(probe.get('passed'))} | "
                f"{probe.get('retrieval_status', '')} | {probe.get('missing_required_facts', '-')} | "
                f"{probe.get('cites_expected_source', '-')} | {probe.get('abstained', '-')} | "
                f"{probe.get('forbidden_fact', '-')} | {probe.get('transport_error', '-')} |"
            )
    lines.append("")
    return lines


def _expansion_comparison_markdown(receipt: dict) -> list[str]:
    comparison = receipt.get("expansion_comparison")
    if comparison is None:
        return []
    lines = ["## 親子展開 OFF/ON 比較（run 1）", ""]
    for route, item in comparison.items():
        lines.append(f"### {route}")
        lines.append("")
        lines.append(
            f"- status低下: {', '.join(item['status_down']) or 'なし'} / "
            f"status向上: {', '.join(item['status_up']) or 'なし'}"
        )
        lines.append(
            f"- 新規FAIL: {', '.join(item['newly_failed']) or 'なし'} / "
            f"新規PASS: {', '.join(item['newly_passed']) or 'なし'}"
        )
        lines.append(
            f"- 再検索発生率: OFF {_format_metric(item['off_retry_rate'])} / "
            f"ON {_format_metric(item['on_retry_rate'])}"
        )
        lines.append("")
        lines.append(
            "| ID | OFF status | ON status | 遷移 | OFF判定 | ON判定 | OFF attempts | ON attempts | OFF line一致 | ON line一致 |"
        )
        lines.append("|" + " --- |" * 10)
        for q in item["questions"]:
            lines.append(
                f"| {q['question_id']} | {q['off_status']} | {q['on_status']} | {q['transition']} | "
                f"{_verdict_label(q['off_passed'])} | {_verdict_label(q['on_passed'])} | "
                f"{q['off_attempts']} | {q['on_attempts']} | "
                f"{_format_metric(q['off_line_overlap'])} | {_format_metric(q['on_line_overlap'])} |"
            )
        lines.append("")
    return lines


@contextmanager
def _parent_child_expansion_env(enabled: bool):
    """親子展開の設定を一時的に切り替え、終了時に元の環境変数へ戻す。"""
    name = "OFFLINE_AI_PARENT_CHILD_EXPANSION"
    original = os.environ.get(name)
    os.environ[name] = "true" if enabled else "false"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def _force_utf8_stdio() -> None:
    """cp932 コンソールで日本語 receipt が化けないよう標準出力をUTF-8にする。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    args = parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()
    routes = (
        harness.ROUTES
        if args.routes.strip().lower() == "all"
        else tuple(r.strip() for r in args.routes.split(",") if r.strip())
    )

    try:
        spec = harness.load_spec(args.spec)
    except harness.SpecError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    chat_model, embed_model, diagnostics = _resolve_models(args)

    try:
        chunks = harness.build_corpus_chunks(spec.corpus_dir)
    except harness.SpecError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    embed_cache = None
    embedding_routes = {harness.ROUTE_HYBRID, harness.ROUTE_AGENTIC_LITE}
    if embedding_routes.intersection(routes) and embed_model:
        print(
            f"[INFO] Embedding インデックスを構築しています（{len(chunks)} chunks）...",
            file=sys.stderr,
        )
        try:
            embed_cache = harness.build_isolated_embed_cache(embed_model, chunks)
        except Exception as exc:  # noqa: BLE001 - skip 理由として receipt へ残す
            print(
                f"[WARN] Embedding インデックス構築に失敗: {type(exc).__name__}",
                file=sys.stderr,
            )
            embed_model = None
            diagnostics["embed"] = "index-build-failed"

    try:
        route_reports = harness.run_evaluation(
            spec,
            routes=routes,
            chunks=chunks,
            embed_model=embed_model,
            embed_cache=embed_cache,
            chat_model=chat_model,
            repeat=args.repeat,
            progress=lambda text: print(f"[INFO] {text}", file=sys.stderr),
        )
    except harness.SpecError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    expansion_comparison = None
    if args.compare_expansion:
        current = search._parent_child_expansion_enabled()
        other_label = "OFF" if current else "ON"
        with _parent_child_expansion_env(not current):
            other_reports = harness.run_evaluation(
                spec,
                routes=routes,
                chunks=chunks,
                embed_model=embed_model,
                embed_cache=embed_cache,
                chat_model=chat_model,
                repeat=1,
                progress=lambda text: print(f"[INFO] (展開{other_label}) {text}", file=sys.stderr),
            )
        if current:
            off_reports, on_reports = other_reports, route_reports
        else:
            off_reports, on_reports = route_reports, other_reports
        expansion_comparison = harness.compare_expansion_reports(off_reports, on_reports)

    answer_probes = None
    if args.measure_answer_layer:
        answer_probes = harness.measure_answer_layer(
            spec,
            routes=routes,
            chunks=chunks,
            embed_model=embed_model,
            embed_cache=embed_cache,
            chat_model=chat_model,
        )
        for route, probes in answer_probes.items():
            harness.apply_answer_probe_to_acceptance(route_reports[route], probes)

    answer_quality_probes = None
    if args.measure_answer_quality:
        answer_quality_probes = harness.measure_answer_quality(
            spec,
            routes=routes,
            chunks=chunks,
            embed_model=embed_model,
            embed_cache=embed_cache,
            chat_model=chat_model,
        )
        for route, probes in answer_quality_probes.items():
            harness.apply_answer_quality_to_acceptance(route_reports[route], probes)

    receipt = {
        "schema_version": "1.0",
        "harness_version": HARNESS_VERSION,
        "started_at": started_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": harness.overall_verdict(route_reports),
        "spec": {
            "name": spec.spec_path.name,
            "sha256": _sha256_file(spec.spec_path),
            "question_count": len(spec.questions),
            "acceptance": spec.acceptance,
        },
        "corpus": {
            "dir": harness.corpus_display(spec),
            "chunk_count": len(chunks),
            "files": _corpus_inventory(spec.corpus_dir),
        },
        "product": _product_baseline(),
        "environment": {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "chat_model": chat_model or "",
            "embed_model": embed_model or "",
            "model_diagnostics": diagnostics,
            "repeat": args.repeat,
            "routes_requested": list(routes),
            "parent_child_expansion": "on" if search._parent_child_expansion_enabled() else "off",
        },
        "routes": _routes_to_dict(route_reports),
    }
    if answer_probes is not None:
        receipt["answer_probes"] = answer_probes
    if answer_quality_probes is not None:
        receipt["answer_quality_probes"] = answer_quality_probes
    if expansion_comparison is not None:
        receipt["expansion_comparison"] = expansion_comparison

    markdown = build_markdown(receipt)
    print(markdown)

    if args.no_write:
        return 0 if receipt["verdict"] == "PASS" else 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"eval-receipt-{stamp}.json"
    md_path = out_dir / f"eval-receipt-{stamp}.md"
    json_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[INFO] receipt: {json_path.name} / {md_path.name}", file=sys.stderr)

    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
