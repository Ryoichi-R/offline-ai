"""Local human-review bundle for real-material deep results; never awards PASS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from search import is_allowed_source_path


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Answer-quality scoring (answer policy v2) -----------------------------------
# Four axes, each 0 (fail) / 1 (needs revision) / 2 (satisfied). The agent's
# primary evaluation and the human confirmation are separate columns; an
# unconfirmed column is never treated as confirmed, and speed is not an axis, so
# it cannot offset a quality failure.
QUALITY_AXES = {
    "semantic_support": "意味支持",
    "question_fit": "質問適合",
    "required_items": "必須事項充足",
    "prose_quality": "文章品質",
}
QUALITY_VERDICTS = ("NOT_SCORED", "FAIL", "NEEDS_REVISION", "PASS_PENDING_HUMAN", "PASS")


def validate_axis_scores(scores: object) -> dict[str, int]:
    """Exactly the four axes, each the integer 0, 1 or 2 (bool is rejected)."""
    if not isinstance(scores, dict) or set(scores) != set(QUALITY_AXES):
        raise ValueError("scores must contain exactly the four quality axes")
    for axis, value in scores.items():
        if type(value) is not int or value not in (0, 1, 2):
            raise ValueError(f"axis {axis} must be 0, 1 or 2")
    return dict(scores)


def acceptance_verdict(
    agent_scores: dict | None,
    human_scores: dict | None = None,
    *,
    hard_conditions_met: bool = True,
) -> str:
    """Independent-acceptance verdict for one case.

    hard_conditions_met covers the non-negotiable zero-tolerance items (no invalid
    citation, no material subject/negation/number/condition/exception/scope error,
    uncertain/not_run never promoted to pass). Without the agent's scores the case
    is NOT_SCORED. Any axis at 0 or a failed hard condition is FAIL; any axis at 1
    is NEEDS_REVISION; all 2 from the agent is only PASS_PENDING_HUMAN until the
    human column is also all 2.
    """
    if agent_scores is None:
        return "NOT_SCORED"
    agent = validate_axis_scores(agent_scores)
    human = validate_axis_scores(human_scores) if human_scores is not None else None
    columns = [agent] + ([human] if human is not None else [])
    if not hard_conditions_met or any(0 in column.values() for column in columns):
        return "FAIL"
    if any(1 in column.values() for column in columns):
        return "NEEDS_REVISION"
    return "PASS" if human is not None else "PASS_PENDING_HUMAN"


def _quality_section(quality: dict) -> list[str]:
    """Automatic (same-model, provisional) gate states for the review Markdown."""
    gates = quality.get("gates") if isinstance(quality.get("gates"), dict) else {}
    summary = {
        "state": quality.get("state"),
        "gates": {gid: (gates.get(gid) or {}).get("status") for gid in ("G1", "G2", "G3", "G4")},
        "removedClaims": quality.get("removedClaims"),
        "correctedQuotes": quality.get("correctedQuotes"),
        "unconfirmedAspects": quality.get("unconfirmedAspects", []),
    }
    return [
        "## 自動品質検査（暫定。同じモデルによる評価を含み、独立受入の代わりにしない）\n",
        _literal(json.dumps(summary, ensure_ascii=False)),
        "## 4軸採点（0=不合格 / 1=要修正 / 2=充足。エージェント評価と人手確認は別欄）\n",
        "| 軸 | エージェント一次評価 | 人手確認 |\n|---|---|---|\n"
        + "".join(f"| {label} | 未採点 | 未確認 |\n" for label in QUALITY_AXES.values()),
    ]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _literal(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}\n"


def _verify_evidence(root: Path, entry: dict) -> None:
    relative = str(entry.get("path", ""))
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
        or any(part in {".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("unsafe evidence path")
    if not is_allowed_source_path(relative):
        raise ValueError("excluded evidence path")
    source = (root / relative).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError("evidence escaped source root")
    if not is_allowed_source_path(source.relative_to(root).as_posix()):
        raise ValueError("excluded evidence target")
    data = source.read_bytes()
    if _digest(data) != entry.get("source_sha256"):
        raise ValueError("source changed")
    lines = data.decode("utf-8", errors="replace").splitlines()
    first, last = entry.get("start_line"), entry.get("end_line")
    if type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(lines):
        raise ValueError("invalid evidence range")
    text = "\n".join(lines[first - 1 : last])
    start, end = entry.get("char_start"), entry.get("char_end")
    if start is not None or end is not None:
        if (
            first != last
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end < len(text)
        ):
            raise ValueError("invalid character range")
        text = text[start : end + 1]
    if text != entry.get("excerpt"):
        raise ValueError("excerpt mismatch")


def export_review_bundle(
    *, query: str, result: dict, source_root: Path, destination: Path, model_digest: str
) -> dict:
    """Verify referenced bytes before saving explicit local answer/review Markdown.

    destination must be new and outside source_root. Receipt contains only
    digests, fixed states and counts; source text belongs to review.md only.
    """
    root = Path(source_root).resolve(strict=True)
    output = Path(destination).resolve()
    if not root.is_dir() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("review output must be separate from source")
    if output.exists():
        raise FileExistsError("review output already exists")
    if not query.strip() or not re.fullmatch(r"[0-9a-f]{64}", model_digest):
        raise ValueError("query and model digest required")
    evidence = result.get("evidence", [])
    ids = set()
    for entry in evidence:
        evidence_id = entry.get("evidence_id", "")
        if not re.fullmatch(r"E[1-9][0-9]*", evidence_id) or evidence_id in ids:
            raise ValueError("invalid or duplicate evidence ID")
        ids.add(evidence_id)
        _verify_evidence(root, entry)
        for context in entry.get("context", []):
            if context.get("path") != entry["path"]:
                raise ValueError("context must belong to the hashed source")
            _verify_evidence(
                root,
                {
                    **context,
                    "source_sha256": entry["source_sha256"],
                    "excerpt": context.get("text"),
                },
            )
    answer = str(result.get("answer", ""))
    unknown_citations = set(re.findall(r"\[(E\d+)\]", answer)) - ids
    answer_md = f"# 実資料評価の回答\n\n## 質問\n\n{query}\n\n## 回答\n\n{answer}\n"
    review = [
        "# 人手による意味確認\n",
        "**未採点。原文とのバイト一致は、回答の意味や網羅性を保証しません。**\n",
        "## 確認項目\n",
        "- [ ] 質問された項目に回答している\n- [ ] 対象・主体・適用範囲が一致する\n- [ ] 条件・例外・否定・期限を保持している\n- [ ] 資料外の推測や断定がない\n- [ ] 未確認事項を踏まえ、採用可否を判断した\n",
        "## 実行状態\n",
        _literal(
            json.dumps(
                {k: result.get(k) for k in ("status", "stop_reason", "answer_state")},
                ensure_ascii=False,
            )
        ),
        "## 未確認事項\n",
        _literal("\n".join(result.get("unconfirmed", [])) or "なし"),
        f"未知の引用ID: {len(unknown_citations)}件\n",
        *(
            _quality_section(result["answer_quality"])
            if isinstance(result.get("answer_quality"), dict) and result["answer_quality"]
            else []
        ),
        "## 引用原文（保存時に照合済み）\n",
    ]
    for entry in evidence:
        review.extend(
            [
                f"### {entry['evidence_id']}\n",
                _literal(
                    f"{entry['path']} L{entry['start_line']}-{entry['end_line']}\nSHA-256: {entry['source_sha256']}\n検証状態: {entry.get('verification_status', 'unknown')}"
                ),
                _literal(entry["excerpt"]),
            ]
        )
        for context in entry.get("context", []):
            review.extend(
                [
                    f"補助文脈 L{context['start_line']}-{context['end_line']}\n",
                    _literal(context["text"]),
                ]
            )
    review_md = "\n".join(review)
    receipt = {
        "schemaVersion": 1,
        "status": "EXPORTED_FOR_HUMAN_REVIEW",
        "semanticReview": "NOT_REVIEWED",
        "acceptanceEligible": False,
        "querySha256": _digest(query.encode()),
        "resultSha256": _digest(_json_bytes(result)),
        "modelDigest": model_digest,
        "sourceVerification": "MATCH",
        "evidenceCount": len(evidence),
        "unknownCitationCount": len(unknown_citations),
        "unconfirmedCount": len(result.get("unconfirmed", [])),
        "qualityAxes": "NOT_SCORED",
        "answerMarkdownSha256": _digest(answer_md.encode()),
        "reviewMarkdownSha256": _digest(review_md.encode()),
    }
    quality = result.get("answer_quality")
    if isinstance(quality, dict) and quality:
        # Fixed states only (no answer or source text).
        receipt["answerQuality"] = {
            "state": quality.get("state"),
            "gates": {
                gid: ((quality.get("gates") or {}).get(gid) or {}).get("status")
                for gid in ("G1", "G2", "G3", "G4")
            },
        }
    output.mkdir(parents=True, exist_ok=False)
    (output / "answer.md").write_text(answer_md, encoding="utf-8")
    (output / "review.md").write_text(review_md, encoding="utf-8")
    # Receipt is the completion marker, written only after both artifacts exist.
    (output / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt
