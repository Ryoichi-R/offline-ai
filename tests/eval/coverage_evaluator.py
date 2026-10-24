"""deep 調査の条件・例外・参照範囲を採点する標準ライブラリ評価器。

回答本文をreceiptへ保存する既存評価ハーネスとは別に、実行時の結果dictを
その場で採点する。評価仕様は合成資料のsnapshotにだけ結び付け、利用者資料
やモデルの自由文をファイルへ書き出さない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(E\d+)\]")


class CoverageSpecError(ValueError):
    """評価仕様が不正で、fail-closedで採点できない。"""


def _normalize(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _string_list(
    value: object, *, field: str, item_id: str, required: bool = False
) -> tuple[str, ...]:
    if value is None:
        result: tuple[str, ...] = ()
    elif isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value):
        result = tuple(item.strip() for item in value)
    else:
        raise CoverageSpecError(f"{item_id}: {field} は空でない文字列配列でなければならない")
    if required and not result:
        raise CoverageSpecError(f"{item_id}: {field} は1件以上必要")
    return result


def _sources(value: object, *, item_id: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise CoverageSpecError(f"{item_id}: expected_sources は1件以上必要")
    result: list[dict[str, Any]] = []
    for source in value:
        if not isinstance(source, dict) or not str(source.get("path", "")).strip():
            raise CoverageSpecError(f"{item_id}: expected_sources.path が必要")
        start = source.get("line_start")
        end = source.get("line_end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1
            or end < start
        ):
            raise CoverageSpecError(f"{item_id}: expected_sources の行範囲が不正")
        path = str(source["path"]).replace("\\", "/")
        if (
            ":" in path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise CoverageSpecError(f"{item_id}: expected_sources.path が相対安全pathでない")
        digest = source.get("sha256", "")
        if not _HASH_RE.fullmatch(digest):
            raise CoverageSpecError("fixed source hash required")
        result.append({"path": path, "line_start": start, "line_end": end, "sha256": digest})
    return tuple(result)


def load_spec(path: str | Path) -> dict[str, Any]:
    """coverage specを検証して正規化する。"""
    spec_path = Path(path).resolve()
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageSpecError(f"評価仕様を読めない: {spec_path.name}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise CoverageSpecError("未対応または不正なschema_version")
    acceptance = data.get("acceptance")
    if not isinstance(acceptance, dict):
        raise CoverageSpecError("acceptance はオブジェクトでなければならない")
    allowed_acceptance = {
        "min_item_coverage",
        "max_critical_misses",
        "max_unsupported_citations",
    }
    if set(acceptance) - allowed_acceptance:
        raise CoverageSpecError("acceptance に未知のキーがある")
    items_raw = data.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        raise CoverageSpecError("items は1件以上必要")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in items_raw:
        if not isinstance(raw, dict):
            raise CoverageSpecError("items の各要素はオブジェクトでなければならない")
        item_id = str(raw.get("id", "")).strip()
        if not item_id or item_id in seen:
            raise CoverageSpecError(f"item id が空または重複: {item_id}")
        seen.add(item_id)
        required_terms = _string_list(
            raw.get("required_terms"),
            field="required_terms",
            item_id=item_id,
            required=True,
        )
        required_any = _string_list(raw.get("required_any"), field="required_any", item_id=item_id)
        evidence_terms = _string_list(
            raw.get("required_evidence_terms"),
            field="required_evidence_terms",
            item_id=item_id,
        )
        statements = _string_list(
            raw.get("expected_statements"),
            field="expected_statements",
            item_id=item_id,
            required=True,
        )
        forbidden = _string_list(
            raw.get("forbidden_patterns"), field="forbidden_patterns", item_id=item_id
        )
        for pattern in forbidden:
            re.compile(pattern)
        items.append(
            {
                "id": item_id,
                "category": str(raw.get("category", "")).strip(),
                "required_terms": required_terms,
                "required_any": required_any,
                "required_evidence_terms": evidence_terms,
                "expected_statements": statements,
                "forbidden_patterns": forbidden,
                "expected_sources": _sources(raw.get("expected_sources"), item_id=item_id),
                "critical": bool(raw.get("critical", False)),
            }
        )
    corpus_dir = str(data.get("corpus_dir", "")).strip()
    if not corpus_dir:
        raise CoverageSpecError("corpus_dir は必須")
    corpus_root = (spec_path.parent / corpus_dir).resolve()
    if not corpus_root.is_dir():
        raise CoverageSpecError(f"corpus_dir が存在しない: {corpus_dir}")
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_dir": corpus_dir,
        "acceptance": {key: value for key, value in acceptance.items()},
        "items": items,
        "spec_path": str(spec_path),
        "corpus_root": str(corpus_root),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    }


def _overlap(start: int, end: int, expected_start: int, expected_end: int) -> bool:
    return start <= expected_end and expected_start <= end


def _evidence_text(item: dict[str, Any]) -> str:
    parts = [
        item.get(key, "")
        for key in (
            "excerpt",
            "subject",
            "scope",
            "conditions",
            "exceptions",
            "references",
        )
    ]
    return _normalize(" ".join(str(part) for part in parts))


def _matching_evidence(
    item: dict[str, Any], evidence: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for expected in item["expected_sources"]:
        for actual in evidence:
            if str(actual.get("path", "")).replace("\\", "/") != expected["path"]:
                continue
            start = actual.get("start_line")
            end = actual.get("end_line")
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and start <= expected["line_start"]
                and end >= expected["line_end"]
            ):
                matches.append(actual)
    return matches


def result_digest(result: dict[str, Any]) -> str:
    bound = {k: result.get(k) for k in ("status", "answer", "evidence")}
    return hashlib.sha256(
        json.dumps(bound, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def evaluate_result(spec: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """deep結果を採点し、本文を含まない構造化レポートを返す。"""
    evidence = result.get("evidence", [])
    if not isinstance(evidence, list):
        raise CoverageSpecError("result.evidence は配列でなければならない")
    evidence_by_id: dict[str, dict[str, Any]] = {}
    invalid_evidence: list[str] = []
    source_hash_mismatches: list[str] = []
    source_hashes: dict[str, str] = {}
    fixed_hashes = {
        s["path"]: s["sha256"] for item in spec["items"] for s in item["expected_sources"]
    }
    for entry in evidence:
        if not isinstance(entry, dict):
            invalid_evidence.append("non_object")
            continue
        evidence_id = str(entry.get("evidence_id", ""))
        if not re.fullmatch(r"E\d+", evidence_id) or evidence_id in evidence_by_id:
            invalid_evidence.append(evidence_id or "missing_id")
            continue
        if not _HASH_RE.fullmatch(str(entry.get("source_sha256", ""))):
            invalid_evidence.append(evidence_id)
            continue
        path = str(entry.get("path", "")).replace("\\", "/")
        if (
            ":" in path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            invalid_evidence.append(evidence_id)
            continue
        source_path = Path(spec["corpus_root"]) / Path(path)
        if not source_path.resolve().is_relative_to(Path(spec["corpus_root"]).resolve()):
            invalid_evidence.append(evidence_id)
            continue
        try:
            expected_hash = source_hashes.setdefault(
                path, hashlib.sha256(source_path.read_bytes()).hexdigest()
            )
        except OSError:
            invalid_evidence.append(evidence_id)
            continue
        if (
            str(entry["source_sha256"]).lower() != expected_hash
            or fixed_hashes.get(path, expected_hash) != expected_hash
        ):
            source_hash_mismatches.append(evidence_id)
            continue
        if (
            type(entry.get("start_line")) is not int
            or type(entry.get("end_line")) is not int
            or entry["start_line"] < 1
            or entry["end_line"] < entry["start_line"]
        ):
            invalid_evidence.append(evidence_id)
            continue
        lines = source_path.read_text(encoding="utf-8").splitlines()
        excerpt = "\n".join(lines[entry["start_line"] - 1 : entry["end_line"]])
        if entry["end_line"] > len(lines) or entry.get("excerpt") != excerpt:
            invalid_evidence.append(evidence_id)
            continue
        evidence_by_id[evidence_id] = entry

    answer = _normalize(result.get("answer", ""))
    cited = set(_CITATION_RE.findall(str(result.get("answer", ""))))
    unsupported_citations = sorted(cited - set(evidence_by_id))
    review = result.get("semantic_review", {})
    reviewed = (
        isinstance(review, dict)
        and review.get("kind") == "human"
        and bool(review.get("reviewer"))
        and review.get("approved") is True
        and review.get("spec_sha256") == spec["spec_sha256"]
        and review.get("result_sha256") == result_digest(result)
        and review.get("unsupported_claims") == 0
        and set(review.get("supported_item_ids", [])) == {i["id"] for i in spec["items"]}
    )
    item_reports: list[dict[str, Any]] = []
    for item in spec["items"]:
        matching = _matching_evidence(item, list(evidence_by_id.values()))
        matching_ids = {str(entry["evidence_id"]) for entry in matching}
        missing_terms = [term for term in item["required_terms"] if _normalize(term) not in answer]
        required_any = item["required_any"]
        any_ok = not required_any or any(_normalize(term) in answer for term in required_any)
        evidence_terms_missing = [
            term
            for term in item["required_evidence_terms"]
            if not any(_normalize(term) in _evidence_text(entry) for entry in matching)
        ]
        cited_support = sorted(cited & matching_ids)
        contradiction = any(re.search(p, answer) for p in item["forbidden_patterns"])
        covered = (
            not contradiction
            and not missing_terms
            and any_ok
            and not evidence_terms_missing
            and bool(matching)
            and bool(cited_support)
        )
        item_reports.append(
            {
                "id": item["id"],
                "category": item["category"],
                "critical": item["critical"],
                "covered": covered,
                "contradiction": contradiction,
                "matching_evidence": cited_support,
                "missing_terms": missing_terms,
                "missing_evidence_terms": evidence_terms_missing,
                "source_found": bool(matching),
            }
        )

    covered_count = sum(1 for item in item_reports if item["covered"])
    coverage = covered_count / len(item_reports)
    critical_misses = [
        item["id"] for item in item_reports if item["critical"] and not item["covered"]
    ]
    acceptance = spec["acceptance"]
    passed = (
        coverage >= float(acceptance.get("min_item_coverage", 1.0))
        and len(critical_misses) <= int(acceptance.get("max_critical_misses", 0))
        and len(unsupported_citations) <= int(acceptance.get("max_unsupported_citations", 0))
        and not invalid_evidence
        and not source_hash_mismatches
        and result.get("status") == "completed"
    )
    categories = sorted({item["category"] for item in item_reports})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed and reviewed else "NEEDS_REVIEW" if passed else "FAIL",
        "semantic_review_valid": reviewed,
        "spec_sha256": spec["spec_sha256"],
        "result_sha256": result_digest(result),
        "category_coverage": {
            c: sum(i["covered"] for i in item_reports if i["category"] == c)
            / sum(i["category"] == c for i in item_reports)
            for c in categories
        },
        "result_status": str(result.get("status", "")),
        "item_coverage": coverage,
        "covered_items": covered_count,
        "total_items": len(item_reports),
        "critical_misses": critical_misses,
        "unsupported_citations": unsupported_citations,
        "invalid_evidence": sorted(invalid_evidence),
        "source_hash_mismatches": sorted(source_hash_mismatches),
        "items": item_reports,
    }


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """複数回評価を保守的に集計する（平均で重大欠落を相殺しない）。"""
    if not reports:
        raise CoverageSpecError("集計対象のreportがない")
    all_ids = sorted({item["id"] for report in reports for item in report.get("items", [])})
    item_runs = {
        item_id: sum(
            1
            for report in reports
            for item in report.get("items", [])
            if item.get("id") == item_id and item.get("covered")
        )
        for item_id in all_ids
    }
    return {
        "runs": len(reports),
        "all_runs_passed": len(reports) >= 3
        and all(report.get("status") == "PASS" for report in reports),
        "min_item_coverage": min(float(report.get("item_coverage", 0.0)) for report in reports),
        "critical_misses": sorted(
            {item for report in reports for item in report.get("critical_misses", [])}
        ),
        "unsupported_citations": sorted(
            {item for report in reports for item in report.get("unsupported_citations", [])}
        ),
        "item_covered_runs": item_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="offline-ai deep coverage evaluator")
    parser.add_argument("spec", type=Path)
    parser.add_argument("result", type=Path, help="run_deep_research結果のJSON")
    args = parser.parse_args()
    try:
        spec = load_spec(args.spec)
        result = json.loads(args.result.read_text(encoding="utf-8"))
        report = evaluate_result(spec, result)
    except (OSError, json.JSONDecodeError, CoverageSpecError) as exc:
        print(json.dumps({"status": "SPEC_ERROR", "message": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
