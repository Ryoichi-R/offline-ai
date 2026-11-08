"""回答品質(v2)試験用の合成規程と偽モデル。実資料の本文は含まない。

構造は、質問の語（審議）と資料の語（審査）が異なること、通知本文に設置の背景があること、
設置要綱本体に審査内容があること、別紙に設問形のチェック表があること、を再現する。
"""

from __future__ import annotations

import json

import answer_policy as ap

QUERY = (
    "提供資料において、設備更新委員会が審議する事項を教えてください。"
    "別の会議体の役割や設置の背景と区別し、出典を示してください"
)

SOURCE_TEXT = """\
# 設備更新委員会の設置について

今般、理事長から「設備の更新費用を２割削減する。」との指示があったところである。

ついては、費用削減を目的として、更新計画の審査を行うため、設備更新委員会を設置することとしたので通知する。

## 設備更新委員会設置要綱

### （目的）

第１条　設備の更新前に計画の審査を行うことにより、費用及び仕様の妥当性を確保するため、設備更新委員会を設置する。

### （組織）

第２条　委員会は、外部有識者及び職員で構成し、外部有識者は委員総数の半数以上とする。

### （審査案件及び審査内容）

第３条　委員会において審査する案件は、次のとおりとする。

- 一　概算費用１，０００万円以上の設備更新
- 二　概算費用３００万円以上の設備改修（緊急修繕を除く。）

２　前項の規定に関わらず、次の各号に該当する案件は審査案件としない。

- 一　同一仕様による部品の交換
- 二　災害復旧のため緊急に行う修繕

３　委員会において審査する内容は、前項の案件について、次のとおりとする。

- 一　費用削減の可否
- 二　仕様の妥当性
- 三　競争性の阻害要因の有無

## 別紙１　設備更新　設備更新委員会チェックリスト

| 確認項目 | 結果 |
|---|---|
| ①　図面が添付されているか。 | |
| ②　技術評価会は、原則、会議形式で開催することになっているか。 | |
| ③　保守契約の期間は適切か。 | |
| ④　見積書は複数の者から取得しているか。 | |
"""

SOURCE_LINES = SOURCE_TEXT.splitlines()
SOURCE_SHA = "a" * 64


def _find(prefix: str) -> int:
    for number, line in enumerate(SOURCE_LINES, 1):
        if line.startswith(prefix):
            return number
    raise KeyError(prefix)


def _slice(start: int, end: int) -> str:
    return "\n".join(SOURCE_LINES[start - 1 : end])


def make_evidence() -> list[dict]:
    """E1=設置の背景、E2=設置要綱 第3条、E3=別紙チェック表。"""
    background = (_find("今般"), _find("ついては"))
    article3 = (_find("### （審査案件及び審査内容）"), _find("- 三　競争性"))
    checklist = (_find("## 別紙１"), len(SOURCE_LINES))
    items = []
    for number, (start, end, heading) in enumerate(
        [
            (background[0], background[1], "設備更新委員会の設置について"),
            (article3[0], article3[1], "（審査案件及び審査内容）"),
            (checklist[0], checklist[1], "別紙１　設備更新　設備更新委員会チェックリスト"),
        ],
        1,
    ):
        items.append(
            {
                "evidence_id": f"E{number}",
                "path": "notice/sample-equipment.md",
                "source_sha256": SOURCE_SHA,
                "heading": heading,
                "start_line": start,
                "end_line": end,
                "excerpt": _slice(start, end),
                "subject": "",
                "scope": "",
                "conditions": [],
                "exceptions": [],
                "references": [],
                "verification_status": "semantic_checked",
            }
        )
    return items


def source_lines_for(path: str, sha: str):
    """G1用の原本行。ハッシュが違えば原本変更として取得不能を返す。"""
    if path != "notice/sample-equipment.md" or sha != SOURCE_SHA:
        return None
    return [line + "\n" for line in SOURCE_LINES]


def claim(text, evidence_id, quote, **extra):
    return {"text": text, "evidence": [{"id": evidence_id, "quote": quote}], **extra}


def good_draft() -> dict:
    """自然な概要回答（結論→審査内容→案件と除外→区別）。"""
    return {
        "conclusion": claim(
            "委員会は、設備の更新前に計画を審査し、費用削減の可否・仕様の妥当性・競争性の阻害要因の有無を確認します。",
            "E2",
            "委員会において審査する内容は、前項の案件について、次のとおりとする。",
            aspect="審査内容",
        ),
        "sections": [
            {
                "heading": "審査する内容",
                "claims": [
                    claim("費用削減が可能かどうかを審査します。", "E2", "費用削減の可否", aspect="審査内容"),
                    claim("仕様が妥当かどうかを審査します。", "E2", "仕様の妥当性"),
                    claim("競争性を阻害する要因がないかを審査します。", "E2", "競争性の阻害要因の有無"),
                ],
            },
            {
                "heading": "審査する案件と除外",
                "scope": claim(
                    "審査する案件は、金額などで次のとおり定められています。",
                    "E2",
                    "委員会において審査する案件は、次のとおりとする。",
                ),
                "claims": [
                    claim("概算費用が１，０００万円以上の設備更新が対象です。", "E2", "概算費用１，０００万円以上の設備更新"),
                    claim(
                        "概算費用が３００万円以上の設備改修が対象ですが、緊急修繕は除きます。",
                        "E2",
                        "概算費用３００万円以上の設備改修（緊急修繕を除く。）",
                    ),
                    claim(
                        "同一仕様による部品の交換と、災害復旧のための緊急修繕は、前項の規定にかかわらず審査案件になりません。",
                        "E2",
                        "次の各号に該当する案件は審査案件としない。",
                        exceptions=["同一仕様による部品の交換", "災害復旧のため緊急に行う修繕"],
                    ),
                ],
            },
            {
                "heading": "設置の背景との区別",
                "claims": [
                    claim(
                        "理事長の指示（更新費用の２割削減）は設置の背景であり、審査内容そのものではありません。",
                        "E1",
                        "設備の更新費用を２割削減する。",
                        aspect="設置の背景との区別",
                    ),
                ],
            },
        ],
        "unanswered_aspects": [],
    }


class FakeModel:
    """プロンプトの種類で応答を切り替える偽のcall_json。呼出し内容を記録する。"""

    def __init__(self, *, drafts=None, g2=None, quality=None, intent=None):
        self.drafts = list(drafts or [])
        self.g2 = g2 or (lambda claims: {c["claim_id"]: ("supported", "ok") for c in claims})
        self.quality = quality
        self.intent = intent
        self.calls: list[tuple[str, dict | str]] = []

    def __call__(self, system: str, user: str, *, timeout: float):
        if system.startswith("あなたは質問の意図"):
            self.calls.append(("intent", user))
            if self.intent is None:
                raise ValueError("no intent")
            return self.intent
        if system.startswith("あなたはローカル資料の調査結果を"):
            self.calls.append(("draft", user))
            value = self.drafts.pop(0) if self.drafts else ValueError("no more drafts")
            if callable(value):
                value = value(user)
            if isinstance(value, Exception):
                raise value
            return value
        if system.startswith("あなたは回答の主張と原文の照合者"):
            payload = json.loads(user)
            # ワイヤ形式の共有原文IDを解決し、既存の意味検査用偽モデルへ渡す。
            for row in payload["claims"]:
                if "source_ids" in row:
                    ranges = {ref["source_id"]: ref["lines"] for ref in row.get("source_ranges", [])}
                    row["source"] = [dict(payload["source_catalog"][sid], **({"lines": ranges[sid]} if sid in ranges else {})) for sid in row["source_ids"]]
            self.calls.append(("g2", payload))
            verdicts = self.g2(payload["claims"])
            return {
                "results": [
                    {"claim_id": cid, "verdict": verdict, "reason_code": reason}
                    for cid, (verdict, reason) in verdicts.items()
                ]
            }
        if system.startswith("あなたは回答の品質検査者"):
            self.calls.append(("g34", json.loads(user)))
            if self.quality is None:
                raise ValueError("no quality")
            return self.quality
        raise AssertionError("unexpected system prompt")

    def count(self, purpose: str) -> int:
        return sum(1 for name, _ in self.calls if name == purpose)


def make_env(model, *, mode="normal", total=300.0, **overrides):
    import answer_pipeline as pipeline

    clock = ap.StageClock(total, mode)
    env = pipeline.PipelineEnv(
        call_json=model,
        clock=clock,
        source_lines_for=source_lines_for,
        check_cap_seconds=ap.CHECK_CALL_CAP_SECONDS[mode],
        draft_cap_seconds=60.0,
    )
    for key, value in overrides.items():
        setattr(env, key, value)
    return env
