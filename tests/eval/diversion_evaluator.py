"""E0-4: Q16型（別対象の手順・条件・数値だけが資料にある該当なし近接語）の回答評価器。

計画（``plans/offline-ai-evidence-sufficiency-verification-plan.md`` E0-4）が
取り下げた「結論節だけの禁止事実照合」（Q-2a）に代わり、回答全体で、質問の対象
（例: 外部委託）へ根拠にある別対象の手順・承認・時間帯・数値を「適用すると断定
したか」を文単位で判定する。

設計:

- 回答を文へ分割する。
- 禁止事実（``forbidden_facts``）ごとに、助詞等を除いた主要語（``search.py`` の
  ``_QUERY_CONTENT_TERM_PATTERN`` と同じ抽出規則）を求め、その全てを含む文を
  「禁止事実に言及した文」とする。表記ゆれ（「所属長の承認」→「所属長承認」）を
  越えて検出するため、助詞を含む原文一致ではなく主要語の包含で判定する。
- 言及文に否定・区別マーカー（「記載がない」「適用され…か」「ではない」等）が
  無ければ、その文は質問の対象へ適用を断定したとみなす（=流用）。
- 1文でも断定と判定されれば、結論節に「該当情報なし」があっても不合格とする
  （「該当なしと誤手順の併記」を合格にしないため）。
- 禁止事実への言及が無い場合は、「該当情報なし」等の該当なしフレーズを含むかで
  判定する（含まなければ作話とみなし不合格）。

この判定は機械的な文パターン照合であり、意味内容の正しさそのものの証明では
ない。既存の「禁止事実の語の一致」判定（``forbidden_fact``、変更前の判定）は
比較のために別途保持し、置き換えない。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import search

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"

REASON_ANSWER_FOUND_OR_RELATED_ONLY = "answer_found_or_related_only"
REASON_DIVERSION = "diversion"
REASON_FABRICATION_OR_NO_ABSTAIN = "fabrication_or_no_abstain"

# 該当情報なしを示すフレーズ。eval_harness.ANSWER_ABSTAIN_PHRASES と揃える
# （evaluator は harness に依存しないよう文字列を直接持つ）。
ABSTAIN_PHRASES = ("該当情報なし", "関連する資料が見つかりませんでした")

# 禁止事実へ言及した文に、質問対象への断定ではなく区別・否定を示すマーカー。
# いずれか1つでも文中にあれば断定とはみなさない。
_NEGATION_MARKERS = (
    "記載がない",
    "記載はない",
    "記載が無い",
    "記載は無い",
    "記載されていない",
    "の記載ではない",
    "についての記載ではない",
    "適用されるかは",
    "適用されるか不明",
    "適用外",
    "対象外",
    "とは限らない",
    "ではない",
    "わからない",
    "不明である",
    "不明です",
    "確認できない",
    "断定できない",
    # 2026-09-19 追加（利用者判断）: gpt-oss:20b が区別に使う丁寧体・名詞止めを
    # 拾えず、正しく区別した回答を流用と誤判定していた（P2の再採点で確認）。
    "言及なし",
    "言及がない",
    "言及はない",
    "言及されていない",
    "言及していない",
    "記載なし",
    "記載されておらず",
    "記述はない",
    "記述がない",
    "触れていない",
    "触れられていない",
    "含まれていない",
    "明示されていない",
    "示されていない",
    "対するものではない",
    "言及されていません",
    "言及していません",
    "記載されていません",
    "記載はありません",
    "記載がありません",
    "触れていません",
    "含まれていません",
    "明示されていません",
    "示されていません",
    "ではありません",
)

_DASH_RE = re.compile("[‐‑‒–—―−－]")
_SENTENCE_SPLIT_RE = re.compile(r"[。\n]+")


def _normalize_text(text: str) -> str:
    """全角半角(NFKC)とハイフン類を統一する（eval_harness と同じ正規化）。"""
    return _DASH_RE.sub("-", unicodedata.normalize("NFKC", str(text or "")))


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]


def _fact_tokens(fact: str) -> list[str]:
    """禁止事実の主要語（助詞等を除いた2文字以上の連続）を抽出する。

    ``search._QUERY_CONTENT_TERM_PATTERN`` と同じ抽出規則を使い、
    「所属長の承認」→「所属長」「承認」のように表記ゆれへ頑健にする。
    トークンが1つも取れない場合は原文をそのまま1トークンとして使う
    （数値のみ・短い固有語などを取りこぼさないため）。
    """
    tokens = [t.lower() for t in search._QUERY_CONTENT_TERM_PATTERN.findall(fact or "")]
    if not tokens:
        cleaned = _normalize_text(fact).strip().lower()
        return [cleaned] if cleaned else []
    return list(dict.fromkeys(tokens))


def _sentence_mentions_fact(sentence: str, fact: str) -> bool:
    tokens = _fact_tokens(fact)
    if not tokens:
        return False
    normalized_sentence = sentence.lower()
    return all(token in normalized_sentence for token in tokens)


def _sentence_has_negation_marker(sentence: str) -> bool:
    return any(marker in sentence for marker in _NEGATION_MARKERS)


def _mentions_abstain_phrase(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(phrase in normalized for phrase in ABSTAIN_PHRASES)


@dataclass
class DiversionVerdict:
    verdict: str
    reason: str
    diverted_sentences: list[str] = field(default_factory=list)
    mentions_forbidden_fact: bool = False
    mentions_abstain_phrase: bool = False


def evaluate_diversion(answer_text: str, forbidden_facts: list[str] | tuple[str, ...]) -> DiversionVerdict:
    """回答全体を文単位で見て、別対象への流用・作話を判定する。"""
    sentences = _split_sentences(answer_text)
    diverted: list[str] = []
    mentions_forbidden_fact = False
    for sentence in sentences:
        for fact in forbidden_facts:
            if not _sentence_mentions_fact(sentence, fact):
                continue
            mentions_forbidden_fact = True
            if not _sentence_has_negation_marker(sentence):
                diverted.append(sentence)
            break

    abstain = _mentions_abstain_phrase(answer_text)

    if diverted:
        return DiversionVerdict(
            verdict=VERDICT_FAIL,
            reason=REASON_DIVERSION,
            diverted_sentences=diverted,
            mentions_forbidden_fact=mentions_forbidden_fact,
            mentions_abstain_phrase=abstain,
        )
    if not abstain:
        return DiversionVerdict(
            verdict=VERDICT_FAIL,
            reason=REASON_FABRICATION_OR_NO_ABSTAIN,
            mentions_forbidden_fact=mentions_forbidden_fact,
            mentions_abstain_phrase=abstain,
        )
    return DiversionVerdict(
        verdict=VERDICT_PASS,
        reason=REASON_ANSWER_FOUND_OR_RELATED_ONLY,
        mentions_forbidden_fact=mentions_forbidden_fact,
        mentions_abstain_phrase=abstain,
    )


def legacy_forbidden_fact_match(answer_text: str, forbidden_facts: list[str] | tuple[str, ...]) -> bool:
    """変更前の判定（禁止事実の原文部分一致）。比較のために保持する。

    ``eval_harness.measure_answer_probe`` / ``measure_answer_quality_probe`` の
    既存ロジックと同じ規則（NFKC + ハイフン正規化のうえでの単純部分文字列一致）。
    """
    normalized_answer = _normalize_text(answer_text)
    return any(_normalize_text(fact) in normalized_answer for fact in forbidden_facts)


# ---------------------------------------------------------------------------
# fixture 検証（E0-3: 評価器の採用条件）
# ---------------------------------------------------------------------------


class FixtureError(ValueError):
    """fixture の構造不備。"""


def load_fixtures(path: Path) -> dict[str, Any]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise FixtureError(f"fixture の構造が不正: {path}")
    return data


@dataclass
class FixtureCaseResult:
    case_id: str
    category: str
    expected_verdict: str
    new_verdict: DiversionVerdict
    legacy_matched: bool
    new_matches_expected: bool
    legacy_matches_expected: bool


def evaluate_fixture_cases(fixture: dict[str, Any]) -> list[FixtureCaseResult]:
    results: list[FixtureCaseResult] = []
    for case in fixture["cases"]:
        forbidden_facts = case.get("forbidden_facts", [])
        answer_text = case["answer_text"]
        expected = case["expected_verdict"]
        new_verdict = evaluate_diversion(answer_text, forbidden_facts)
        legacy_forbidden = legacy_forbidden_fact_match(answer_text, forbidden_facts)
        # 変更前の判定契約: forbidden_fact が一致すれば不合格、それ以外は合格
        # （measure_answer_probe.passed の forbidden_fact 項に相当）。
        legacy_verdict = VERDICT_FAIL if legacy_forbidden else VERDICT_PASS
        results.append(
            FixtureCaseResult(
                case_id=case["id"],
                category=case["category"],
                expected_verdict=expected,
                new_verdict=new_verdict,
                legacy_matched=legacy_forbidden,
                new_matches_expected=new_verdict.verdict == expected,
                legacy_matches_expected=legacy_verdict == expected,
            )
        )
    return results


def fixture_acceptance_report(fixture: dict[str, Any]) -> dict[str, Any]:
    """E0-3の採用条件（fixture全件で期待判定と一致）を判定するreceipt用レポート。"""
    results = evaluate_fixture_cases(fixture)
    new_failures = [r for r in results if not r.new_matches_expected]
    legacy_failures = [r for r in results if not r.legacy_matches_expected]
    return {
        "case_count": len(results),
        "new_evaluator": {
            "all_matched": not new_failures,
            "mismatched_case_ids": [r.case_id for r in new_failures],
        },
        "legacy_evaluator": {
            "all_matched": not legacy_failures,
            "mismatched_case_ids": [r.case_id for r in legacy_failures],
        },
        "adopted": "new" if not new_failures else ("legacy" if not legacy_failures else "none"),
        "cases": [
            {
                "id": r.case_id,
                "category": r.category,
                "expected_verdict": r.expected_verdict,
                "new_verdict": r.new_verdict.verdict,
                "new_reason": r.new_verdict.reason,
                "new_matches_expected": r.new_matches_expected,
                "legacy_verdict": VERDICT_FAIL if r.legacy_matched else VERDICT_PASS,
                "legacy_matches_expected": r.legacy_matches_expected,
            }
            for r in results
        ],
    }
