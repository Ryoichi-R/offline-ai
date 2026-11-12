"""資料に基づく確認案内と段階的な深掘り（plans/offline-ai-guided-follow-up-plan.md 段階2）。

既定は無効で、``OFFLINE_AI_GUIDED_FOLLOW_UP=1`` のときだけWeb画面の通常回答・深掘りで使う。
このモジュールはモデルを呼ばない。資料の照合・質問範囲の判定・概要と選択肢の抽出・
追質問の対象と論点の解決はすべて原文とローカル検索チャンクから決定的に行う。

確認情報はサーバーのメモリ内だけに置き、再起動で失われる。資料本文・質問履歴を
通常ログやファイルへ保存しない。
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import answer_policy as ap
from source_structure import (
    document_numbers,
    documents_from_chunks,
    match_documents,
    normalize_match_text,
    parse_heading_tree,
    primary_units,
    title_exclusions,
)
from source_view import EvidenceRegistry

GUIDED_ENV = "OFFLINE_AI_GUIDED_FOLLOW_UP"

# 新規の続きの操作を受け付ける期限（暫定）。閲覧・再表示では延長しない。
CONFIRMATION_IDLE_SECONDS = 30 * 60
CONFIRMATION_MAX_AGE_SECONDS = 2 * 60 * 60
CONFIRMATION_LIMIT = 64
FOLLOW_UP_MAX_CHARS = 1000

OVERVIEW_MAX_POINTS = 4
OVERVIEW_POINT_CHARS = 200
TOPIC_OPTION_MAX = 4
DOCUMENT_CANDIDATE_MAX = 5

ACTION_SELECT_DOCUMENT = "select_document"
ACTION_HINT = "hint"
ACTION_SELECT_TOPIC = "select_topic"
ACTION_FOLLOW_UP = "follow_up"
ACTIONS = frozenset({ACTION_SELECT_DOCUMENT, ACTION_HINT, ACTION_SELECT_TOPIC, ACTION_FOLLOW_UP})

STATE_DOCUMENT_PENDING = "document_pending"
STATE_DOCUMENT_FIXED = "document_fixed"

_CID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_NUMBER_ONLY_RE = re.compile(r"^\s*([0-9０-９]{1,2})\s*[.．。)）]?\s*$")
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*・●○]|\(?[0-9０-９]+[).）]|[（(][0-9０-９]+[）)]|[①-⑳]|[一二三四五六七八九十]+[、.．]|[ア-ン][.．、])"
)
_QUALIFIER_RE = re.compile(r"例外|ただし|但し|除く|除き|場合|条件|限り|適用しない")
# 概要で省いてはならない限定（省くと適用範囲を広く見せる）。
_OMIT_QUALIFIER_RE = re.compile(r"ただし|但し|除く|除き|除いて|限る|限り|適用しない|この限りでない|例外")
_DATE_RE = re.compile(r"(?:令和|平成|昭和)?\s*[0-9０-９元]+\s*年\s*[0-9０-９]+\s*月\s*[0-9０-９]+\s*日")
_KIND_RE = re.compile(r"事務連絡|別添|別紙|通知|規程|要領|要綱|通達")
_REPLACE_RE = re.compile(r"だけ|のみ")
_ADD_RE = re.compile(r"も(?:[、。？?！!\s]|$)|及び|および|並びに")
# 汎用語は対象（具体的な語）に数えない。
_GENERIC_TERMS = frozenset(
    {
        "内容", "詳細", "詳しく", "ポイント", "概要", "要点", "取扱い", "取り扱い", "取扱", "説明",
        "全体", "この", "通知", "資料", "文書", "事務連絡", "別添", "教えて", "知りたい", "について",
    }
)


class ConfirmationError(RuntimeError):
    """確認の続きを受け付けられない理由（コードは web_server が HTTP 応答へ写す）。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def guided_enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(GUIDED_ENV, "")).strip() == "1"


def valid_confirmation_id(value: object) -> bool:
    return isinstance(value, str) and bool(_CID_RE.fullmatch(value))


def normalize_follow_up_text(value: object) -> str:
    """自由入力を検証する（Unicodeのコードポイントで1000文字以内）。"""
    if not isinstance(value, str):
        raise ConfirmationError("invalid_query", "入力が不正です")
    text = unicodedata.normalize("NFC", value).strip()
    if not text:
        raise ConfirmationError("invalid_query", "入力が空です")
    if len(text) > FOLLOW_UP_MAX_CHARS:
        raise ConfirmationError("invalid_query", f"入力は{FOLLOW_UP_MAX_CHARS}文字以内にしてください")
    return text


# ---------------------------------------------------------------------------
# 資料の読取（検索時と同じハッシュの原文だけ）
# ---------------------------------------------------------------------------


def read_document(source_root: Path, relative_path: str, expected_sha256: str = "") -> tuple[str, str] | None:
    """資料を読み、``(text, sha256)`` を返す。境界外・不一致・読めない場合は None。"""
    from source_view import (
        MAX_SOURCE_FILE_BYTES,
        EvidenceViewError,
        _assert_path_boundary,
    )

    try:
        path = _assert_path_boundary(Path(source_root), relative_path)
        data = path.read_bytes()
    except (EvidenceViewError, OSError, ValueError):
        return None
    if len(data) > MAX_SOURCE_FILE_BYTES:
        return None
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256.lower():
        return None
    return data.decode("utf-8", errors="replace"), digest


# ---------------------------------------------------------------------------
# 概要と選択肢（原文から抽出し、文言を変えない）
# ---------------------------------------------------------------------------


def _body_start(unit: dict, lines: list[str]) -> int:
    start = unit["start_line"]
    if unit["heading"] and start <= len(lines) and lines[start - 1].lstrip().startswith("#"):
        return start + 1
    return start


def _point_for_range(lines: list[str], start: int, end: int, heading: str) -> dict | None:
    """節の最初の段落を抜き出し、直後の条件・例外の箇条書きまで含める。

    意味が通る範囲を短く収められない場合は、見出しと行範囲だけの案内にする。
    """
    line_no = start
    while line_no <= end and (not lines[line_no - 1].strip() or lines[line_no - 1].lstrip().startswith("#")):
        line_no += 1
    if line_no > end:
        return None
    first = line_no
    while line_no <= end and lines[line_no - 1].strip() and not lines[line_no - 1].lstrip().startswith("#"):
        line_no += 1
    last = line_no - 1
    # 原則の直後の箇条書きに条件・例外があれば、その範囲まで含める。
    probe = last + 1
    if probe <= end and not lines[probe - 1].strip():
        probe += 1
    extra_last = None
    while probe <= end and _LIST_ITEM_RE.match(lines[probe - 1]):
        extra_last = probe
        probe += 1
    if extra_last is not None and any(
        _QUALIFIER_RE.search(lines[n - 1]) for n in range(last + 1, extra_last + 1)
    ):
        last = extra_last
    text = "\n".join(lines[first - 1 : last])
    if len(text) > OVERVIEW_POINT_CHARS * 2:
        # 長い段落は先頭から文単位で原文のまま抜き出す。省く文に例外・限定があれば、
        # 原則だけを見せて適用範囲を広く見せないよう、案内（見出しと行範囲）にする。
        sentences = re.findall(r"[^。]*。|[^。]+$", text)
        kept, used = [], 0
        for sentence in sentences:
            if used + len(sentence) > OVERVIEW_POINT_CHARS * 2:
                break
            kept.append(sentence)
            used += len(sentence)
        omitted = "".join(sentences[len(kept):])
        if not kept or _OMIT_QUALIFIER_RE.search(omitted):
            return {"kind": "guide", "heading": heading, "startLine": first, "endLine": last}
        return {
            "kind": "excerpt", "heading": heading, "startLine": first, "endLine": last,
            "text": "".join(kept).strip(), "truncated": True,
        }
    return {"kind": "excerpt", "heading": heading, "startLine": first, "endLine": last, "text": text}


def _table_lines(lines: list[str], start: int, end: int) -> list[int]:
    return [n for n in range(start, min(end, len(lines)) + 1) if lines[n - 1].strip().startswith("|")]


def _appendix_point(lines: list[str], unit: dict) -> dict | None:
    """別紙・別表の子単位: 表の見出し行と先頭2行を原文のまま示す（R3）。"""
    rows = _table_lines(lines, unit["start_line"], unit["end_line"])
    if not rows:
        return _point_for_range(lines, unit["start_line"] + 1, unit["end_line"], unit["heading"])
    shown = rows[:4]  # 見出し行・区切り行・先頭2行
    text = "\n".join(lines[n - 1] for n in shown)
    point = {"kind": "excerpt", "heading": unit["heading"], "startLine": shown[0], "endLine": shown[-1], "text": text}
    if len(rows) > len(shown):
        point["truncated"] = True
    return point


def _appendix_label(lines: list[str], unit: dict) -> str:
    rows = _table_lines(lines, unit["start_line"], unit["end_line"])
    marker = unit["heading"].rsplit("／", 1)[-1]
    if not rows:
        return unit["heading"]
    columns = [cell.strip() for cell in lines[rows[0] - 1].strip().strip("|").split("|") if cell.strip()]
    return f"{marker}: " + " / ".join(columns[:3]) if columns else unit["heading"]


def build_overview(text: str) -> dict:
    """短い概要（2〜4点）と論点の選択肢を原文から作る。

    節の中の別紙・別表は表の見出し行と先頭行を点にし、直後の注記に除外・限定があれば
    同じ点に添える（原則と例外を切り離さない。R3）。
    """
    lines = text.splitlines()
    units = [unit for unit in primary_units(text) if not unit["ancillary"]]
    points: list[dict] = []
    for index, unit in enumerate(units):
        if unit.get("kind") == "note":
            # 直前の別紙の点に添える。別紙が点に無ければ独立した点にする。
            previous = units[index - 1] if index else None
            if previous and previous.get("kind") == "appendix" and points and points[-1].get("_unit") == id(previous):
                note = _point_for_range(lines, unit["start_line"], unit["end_line"], unit["heading"])
                if note is not None:
                    points[-1].setdefault("notes", []).append(
                        {key: note[key] for key in ("startLine", "endLine", "text") if key in note}
                    )
                continue
        if len(points) >= OVERVIEW_MAX_POINTS:
            break
        if unit.get("kind") == "appendix":
            point = _appendix_point(lines, unit)
        else:
            point = _point_for_range(lines, _body_start(unit, lines), unit["end_line"], unit["heading"])
        if point is not None:
            point["_unit"] = id(unit)
            points.append(point)
    if len(points) < 2 and units:
        # 節が少ない資料では、同じ範囲の後続の段落から補う。
        unit = units[0]
        cursor = points[-1]["endLine"] + 1 if points else _body_start(unit, lines)
        while len(points) < 2 and cursor <= unit["end_line"]:
            point = _point_for_range(lines, cursor, unit["end_line"], unit["heading"])
            if point is None:
                break
            points.append(point)
            cursor = point["endLine"] + 1
    for point in points:
        point.pop("_unit", None)
    sections = [unit for unit in units if unit["heading"] and unit.get("kind") != "note"]
    options = [
        {
            "kind": "section",
            "label": _appendix_label(lines, unit) if unit.get("kind") == "appendix" else unit["heading"],
            "startLine": unit["start_line"],
            "endLine": unit["end_line"],
        }
        for unit in sections[:TOPIC_OPTION_MAX]
    ]
    options.append({"kind": "overview", "label": "全体をまとめて"})
    remaining = [unit["heading"] for unit in sections[TOPIC_OPTION_MAX:]]
    return {"points": points, "options": options, "moreSections": remaining}


def _document_card(document: dict) -> dict:
    head = str(document.get("head_text") or "")
    date = _DATE_RE.search(unicodedata.normalize("NFKC", head))
    kind = _KIND_RE.search(str(document.get("title") or "") + " " + str(document.get("path") or ""))
    card = {"title": str(document.get("title") or document.get("path") or ""), "path": str(document.get("path") or "")}
    if date:
        card["date"] = date.group(0)
    if kind:
        card["kind"] = kind.group(0)
    if document.get("paths"):
        card["parts"] = len(document["paths"])
    return card


# ---------------------------------------------------------------------------
# 初回質問の振り分け
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """初回質問・資料選択・手掛かり入力の結果。

    kind: ``direct``（従来経路で回答）/ ``answer_fixed``（固定資料で回答）/
    ``topics``（概要と論点の選択肢）/ ``documents``（資料の候補）/ ``hints``（手掛かりの質問）。
    """

    kind: str
    document: dict | None = None
    candidates: list[dict] = field(default_factory=list)
    overview: dict | None = None
    scope: str = ""
    message: str = ""


def _documents_by_path(chunks: list[dict]) -> dict[str, dict]:
    return {doc["path"]: doc for doc in documents_from_chunks(chunks)}


def _fixed_document(source_root: Path, document: dict) -> dict | None:
    """資料（分割資料は全部分）を検索時と同じハッシュで読み、固定用の情報を作る。"""
    members = document.get("paths") or [{"path": document["path"], "file_sha256": document.get("file_sha256")}]
    parts = []
    for member in members:
        read = read_document(source_root, member["path"], str(member.get("file_sha256") or ""))
        if read is None:
            return None
        parts.append({"path": member["path"], "sha256": read[1], "text": read[0]})
    titles = list(document.get("titles") or [document["title"]])
    return {
        "path": parts[0]["path"],
        "title": document["title"],
        "titles": titles,
        "sha256": parts[0]["sha256"],
        "paths": [{"path": part["path"], "sha256": part["sha256"]} for part in parts],
        "sourceRoot": str(Path(source_root).resolve()),
        "text": parts[0]["text"],
        "parts": parts,
    }


def document_parts(document: dict) -> list[dict]:
    """固定資料の部分（``path``・``text``）。分割していない資料は1部分。"""
    return document.get("parts") or [{"path": document["path"], "text": document["text"]}]


RELATED_DOCUMENT_MAX = 2


def find_related(document: dict, chunks: list[dict]) -> list[dict]:
    """固定資料の冒頭の文書番号を本文で引用している、同じフォルダ（親フォルダ）の資料（R6）。

    接頭部のない番号（「第２号」だけ）は他の資料と取り違えやすいため使わない。
    戻り値は選択肢用の ``{"title", "kind", "paths": [{"path", "sha256"}]}``（最大2件）。
    """
    numbers = {n for n in document_numbers(document.get("head_text") or "") if n[0]}
    if not numbers:
        return []
    own = {m["path"] for m in document.get("paths") or [{"path": document["path"]}]}
    folder = document["path"].rsplit("/", 1)[0] if "/" in document["path"] else ""
    parent = folder.rsplit("/", 1)[0] if "/" in folder else ""
    citing: list[str] = []
    for chunk in chunks:
        path = str(chunk.get("path") or "").replace("\\", "/")
        if path in own or path in citing:
            continue
        chunk_folder = path.rsplit("/", 1)[0] if "/" in path else ""
        if chunk_folder not in {folder, parent} and not chunk_folder.startswith(folder + "/"):
            continue
        if numbers & document_numbers(str(chunk.get("text") or "")):
            citing.append(path)
    documents = {doc["path"]: doc for doc in documents_from_chunks(chunks)}
    by_member = {m["path"]: doc for doc in documents.values() for m in doc.get("paths") or []}
    related, seen = [], set()
    for path in citing:
        doc = documents.get(path) or by_member.get(path)
        if doc is None or doc["path"] in seen:
            continue
        seen.add(doc["path"])
        card = _document_card(doc)
        related.append({
            "title": card["title"],
            "kind": card.get("kind", ""),
            "paths": [
                {"path": m["path"], "sha256": m.get("file_sha256") or ""}
                for m in doc.get("paths") or [{"path": doc["path"], "file_sha256": doc.get("file_sha256")}]
            ],
        })
        if len(related) >= RELATED_DOCUMENT_MAX:
            break
    return related


def build_document_overview(document: dict, related: list[dict] | None = None) -> dict:
    """資料（分割資料は部分ごと）の概要と選択肢。点・選択肢には部分のパスを付ける。

    ``related`` は関連資料（番号を引用する事務連絡等）。本文との区別が分かるラベルで選択肢に加える。
    """
    parts = document_parts(document)
    points, options, more = [], [], []
    for index, part in enumerate(parts, start=1):
        overview = build_overview(part["text"])
        prefix = f"［{index}/{len(parts)}］" if len(parts) > 1 else ""
        for point in overview["points"]:
            points.append({**point, "path": part["path"], "heading": prefix + point.get("heading", "")})
        for option in overview["options"]:
            if option["kind"] == "section":
                options.append({**option, "path": part["path"], "label": prefix + option["label"]})
        more.extend(prefix + label for label in overview["moreSections"])
    if len(parts) > 1:
        more = [o["label"] for o in options[TOPIC_OPTION_MAX:]] + more
        options = options[:TOPIC_OPTION_MAX]
        points = points[:OVERVIEW_MAX_POINTS]
    for item in (related or [])[:RELATED_DOCUMENT_MAX]:
        kind = f"{item['kind']}・" if item.get("kind") else ""
        options.append({"kind": "related", "label": f"関連資料（{kind}{item['title']}）", "paths": item["paths"],
                        "title": item["title"]})
    options.append({"kind": "overview", "label": "全体をまとめて"})
    return {"points": points, "options": options, "moreSections": more}


def _exclusions(document: dict) -> tuple[str, ...]:
    return title_exclusions(document.get("titles") or [document.get("title") or ""])


def route_for_document(query: str, document: dict, related: list[dict] | None = None) -> Decision:
    """資料が確定した後、元の質問の範囲で回答・案内を決める。"""
    scope = ap.classify_guided_scope(query, exclude_texts=_exclusions(document))
    if scope.scope == ap.GUIDED_SCOPE_BROAD:
        return Decision("topics", document=document, overview=build_document_overview(document, related),
                        scope=scope.scope)
    return Decision("answer_fixed", document=document, scope=scope.scope)


def decide(query: str, chunks: list[dict], source_root: Path, *, initial: bool = True) -> Decision:
    """質問と資料を照合して初回の応答を決める（モデルを呼ばない）。"""
    documents = _documents_by_path(chunks)
    matched = match_documents(query, list(documents.values()))
    if matched["confirmed"]:
        fixed = _fixed_document(source_root, documents[matched["confirmed"]])
        if fixed is None:
            return Decision("direct", message="資料を読み出せませんでした")
        scope = ap.classify_guided_scope(query, exclude_texts=_exclusions(fixed))
        if initial and scope.scope != ap.GUIDED_SCOPE_BROAD:
            # 具体的な質問・全体概要の明示は従来の回答経路で直接回答する。
            return Decision("direct", scope=scope.scope)
        source = documents[matched["confirmed"]]
        return route_for_document(query, fixed, find_related({**fixed, "head_text": source.get("head_text")}, chunks))
    candidates = matched["candidates"]
    # 資料が確定しなくても、候補の表題に含まれる語（「範囲」等）は尋ねている項目に数えない。
    exclusions = tuple(dict.fromkeys(t for item in candidates[:DOCUMENT_CANDIDATE_MAX] for t in item["exclude"]))
    scope = ap.classify_guided_scope(query, exclude_texts=exclusions)
    if initial and scope.scope != ap.GUIDED_SCOPE_BROAD:
        # 資料が確定しない具体的な質問は、従来の資料横断の回答経路で直接回答する
        # （不要な確認を増やさない）。
        return Decision("direct", scope=scope.scope)
    if candidates:
        cards = [_document_card(documents[item["path"]]) for item in candidates[:DOCUMENT_CANDIDATE_MAX]]
        for card, item in zip(cards, candidates):
            card["fileSha256"] = str(documents[item["path"]].get("file_sha256") or "")
        return Decision("documents", candidates=cards, scope=scope.scope,
                        message="該当しそうな資料が複数あります。対象の資料を選んでください。"
                        if len(cards) > 1 else "この資料が対象か確認してください。")
    return Decision(
        "hints",
        scope=scope.scope,
        message=(
            "対象の資料を確定できませんでした。時期（発出年など）・発出者・文書番号・"
            "知りたい実務のうち、分かるものを入力してください。"
        ),
    )


# ---------------------------------------------------------------------------
# 追質問の対象と論点
# ---------------------------------------------------------------------------


@dataclass
class Topic:
    """現在の論点: 選んだ節、引き継ぐ対象、直前に尋ねた項目の手掛かり語。"""

    section: dict | None = None  # {"label", "startLine", "endLine"}
    targets: tuple[str, ...] = ()
    aspects: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "section": dict(self.section) if self.section else None,
            "targets": list(self.targets),
            "aspects": list(self.aspects),
        }


@dataclass
class FollowUpPlan:
    """追質問の解決結果。``ask`` があれば回答せず聞き返す。"""

    query: str
    topic: Topic
    focus_ranges: list[tuple[int, int, str]] = field(default_factory=list)  # (開始行, 終了行, パス)
    aspects: tuple[str, ...] = ()
    overview: bool = False
    ask: dict | None = None
    related: dict | None = None  # 関連資料の選択（固定資料に加えて検索・根拠の対象にする）


def _terms(text: str, exclude: tuple[str, ...]) -> list[str]:
    normalized_exclude = {normalize_match_text(item) for item in exclude if item}
    terms = []
    for term in ap._TERM_RE.findall(unicodedata.normalize("NFKC", text)):
        if term in ap._STOP_TERMS or term in _GENERIC_TERMS:
            continue
        if normalize_match_text(term) in normalized_exclude:
            continue
        if any(normalize_match_text(term) in item for item in normalized_exclude if len(term) >= 2):
            continue
        terms.append(term)
    return list(dict.fromkeys(terms))


def _aspect_cues(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    labels, cues = [], []
    for label, words in ap.GUIDED_ITEM_CUES:
        hit = [word for word in words if word in text]
        if hit:
            labels.append(label)
            cues.extend(hit)
    return tuple(labels), tuple(cues)


def _leaf_ranges(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    nodes = parse_heading_tree(text)
    if not nodes:
        return [(1, len(lines), "")]
    ranges = []
    if nodes[0].heading_line > 1:
        ranges.append((1, nodes[0].heading_line - 1, ""))
    for node in nodes:
        end = nodes[node.children_indices[0]].heading_line - 1 if node.children_indices else node.section_end_line
        if any(line.strip() for line in lines[node.heading_line : end]):
            ranges.append((node.heading_line, end, node.heading_text))
    return ranges


def _ranges_containing(parts: list[dict], words: tuple[str, ...], within: dict | None) -> list[tuple[int, int, str, str]]:
    """語をすべて含む節（葉）の ``(開始行, 終了行, 見出し, パス)``。``within`` は選んだ節。"""
    result = []
    for part in parts:
        if within and within.get("path", part["path"]) != part["path"]:
            continue
        lines = part["text"].splitlines()
        for start, end, heading in _leaf_ranges(part["text"]):
            if within and not (within["startLine"] <= start and end <= within["endLine"]):
                continue
            body = unicodedata.normalize("NFKC", "\n".join(lines[start - 1 : end]))
            if all(word in body for word in words):
                result.append((start, end, heading, part["path"]))
    return result


def _range_text(parts: list[dict], start: int, end: int, path: str) -> str:
    part = next((item for item in parts if item["path"] == path), parts[0])
    return "\n".join(part["text"].splitlines()[start - 1 : end])


def resolve_follow_up(
    document: dict,
    topic: Topic,
    options: list[dict],
    *,
    action: str,
    choice: int | None = None,
    text: str = "",
    drop_targets: tuple[str, ...] = (),
) -> FollowUpPlan:
    """論点選択・追質問から、回答の問い・資料内の焦点範囲・新しい論点を決める。

    規則（モデルを使わない）:
    - 選択肢から節を選ぶと、対象を空にしてその節を論点にする。番号だけの入力も同じ。
    - 固定資料の見出し・本文に現れる具体的な語を対象とし、振り分け用の手掛かり語は論点とする。
    - 新しい対象は直前の対象を置き換え、「も」「及び」で並べた場合は追加する。「だけ」「のみ」は置き換え。
    - 論点だけの入力は直前に確定した対象を引き継ぐ。
    - 対象が資料内の複数の箇所に残る場合は、同じ資料内の候補を示して聞き返す。
    """
    title = document["title"]
    parts = document_parts(document)
    current = Topic(topic.section, tuple(t for t in topic.targets if t not in drop_targets), topic.aspects)
    if action == ACTION_FOLLOW_UP:
        number = _NUMBER_ONLY_RE.match(text or "")
        if number:
            action, choice = ACTION_SELECT_TOPIC, int(unicodedata.normalize("NFKC", number.group(1))) - 1
    if action == ACTION_SELECT_TOPIC:
        if choice is None or not 0 <= choice < len(options):
            raise ConfirmationError("confirmation_invalid", "選択肢が無効です")
        option = options[choice]
        if option["kind"] == "overview":
            return FollowUpPlan(f"「{title}」の全体をまとめて説明してください。", current, overview=True)
        if option["kind"] == "related":
            return FollowUpPlan(
                f"「{title}」に関連する「{option['title']}」について、資料ごとに説明してください。",
                current,
                overview=True,
                related={"title": option["title"], "paths": list(option["paths"])},
            )
        path = option.get("path") or parts[0]["path"]
        section = {"label": option["label"], "startLine": option["startLine"], "endLine": option["endLine"], "path": path}
        return FollowUpPlan(
            f"「{title}」の「{option['label']}」について説明してください。",
            Topic(section, (), ()),
            focus_ranges=[(option["startLine"], option["endLine"], path)],
        )
    if action != ACTION_FOLLOW_UP:
        raise ConfirmationError("confirmation_invalid", "現在の状態に合わない操作です")

    labels, cues = _aspect_cues(text)
    cue_set = set(cues)
    body = unicodedata.normalize("NFKC", "\n".join(part["text"] for part in parts))
    candidates = [
        term
        for term in _terms(text, _exclusions(document))
        if term not in cue_set and not any(cue in term for cue in cue_set)
    ]
    present = [term for term in candidates if term in body]
    missing = [term for term in candidates if term not in body]
    if present:
        if _REPLACE_RE.search(text):
            targets = tuple(present)
        elif _ADD_RE.search(text) or len(present) > 1 and re.search(r"[とや]", text):
            targets = tuple(dict.fromkeys(current.targets + tuple(present)))
        else:
            targets = tuple(present)
    else:
        targets = current.targets
    # 項目の手掛かり語は新しく尋ねた場合に置き換え、無ければ直前のものを引き継ぐ。
    cues = cues or current.aspects
    new_topic = Topic(current.section, targets, tuple(dict.fromkeys(cues)))
    if missing and not present and not labels:
        return FollowUpPlan(
            text,
            current,
            ask={
                "message": "固定中の資料から「" + "」「".join(missing[:3]) + "」を確認できませんでした。",
                "options": [],
            },
        )
    within = current.section
    focus: list[tuple[int, int, str, str]] = []
    cue_groups = [words for label, words in ap.GUIDED_ITEM_CUES if any(w in cues for w in words)]

    def best_by_cues(ranges):
        """尋ねた項目の種類を最も多く含む範囲だけを残す（同数なら全部）。"""
        if not cue_groups or not ranges:
            return ranges, 0
        scored = [
            (sum(any(w in _range_text(parts, r[0], r[1], r[3]) for w in group) for group in cue_groups), r)
            for r in ranges
        ]
        top = max(score for score, _ in scored)
        return [r for score, r in scored if score == top and score > 0], top

    if targets:
        focus, local_score = best_by_cues(_ranges_containing(parts, targets, within))
        if within and cue_groups and local_score < len(cue_groups):
            # 選んだ節に尋ねた項目（例外・期限等）が揃わない場合は、同じ資料の他の箇所も探す
            # （「別紙の様式のうち除かれるもの」は別紙の注記にある等）。
            others, other_score = best_by_cues(_ranges_containing(parts, targets, None))
            if other_score > local_score:
                focus = others
        if not focus:
            focus = _ranges_containing(parts, targets, within) or _ranges_containing(parts, targets, None)
        if len(focus) > 1:
            return FollowUpPlan(
                text,
                current,
                aspects=labels,
                ask={
                    "message": "「" + "」「".join(targets) + "」は資料内の複数の箇所にあります。どの箇所について知りたいですか。",
                    "options": [
                        {"kind": "section", "label": heading or f"行 {start}-{end}", "startLine": start, "endLine": end,
                         "path": path}
                        for start, end, heading, path in focus[:TOPIC_OPTION_MAX]
                    ],
                },
            )
    elif within:
        focus = [(within["startLine"], within["endLine"], within["label"], within.get("path") or parts[0]["path"])]
    context = []
    if current.section:
        context.append(f"「{current.section['label']}」")
    if targets:
        context.append("対象: " + "、".join(targets))
    prefix = f"「{title}」" + ("（" + "、".join(context) + "）" if context else "")
    return FollowUpPlan(
        f"{prefix}について: {text}",
        new_topic,
        focus_ranges=[(start, end, path) for start, end, _, path in focus],
        aspects=labels,
    )


# ---------------------------------------------------------------------------
# 確認情報（サーバーのメモリ内）
# ---------------------------------------------------------------------------


@dataclass
class ConfirmationRecord:
    confirmation_id: str
    session_id: str
    initial_query: str
    state: str
    created_at: float
    updated_at: float
    last_completed_at: float
    candidates: list[dict] = field(default_factory=list)
    options: list[dict] = field(default_factory=list)
    options_version: int = 0
    document: dict | None = None  # path/title/sha256/sourceRoot（本文は保持しない）
    topic: Topic = field(default_factory=Topic)
    message: str = ""
    points: list[dict] = field(default_factory=list)
    more_sections: list[str] = field(default_factory=list)
    busy_request: str = ""

    def public(self, evidence_registry: EvidenceRegistry | None = None) -> dict:
        """画面へ返す確認情報（候補・選択肢・固定中の資料と論点）。"""
        data = {
            "confirmationId": self.confirmation_id,
            "state": self.state,
            "optionsVersion": self.options_version,
            "message": self.message,
        }
        if self.state == STATE_DOCUMENT_PENDING:
            data["candidates"] = [
                {key: value for key, value in card.items() if key != "fileSha256"} for card in self.candidates
            ]
        else:
            data["document"] = {
                key: self.document[key] for key in ("path", "title") if self.document and key in self.document
            }
            if self.document and len(self.document.get("paths") or []) > 1:
                data["document"]["parts"] = len(self.document["paths"])
            data["points"] = self._public_points(evidence_registry)
            data["options"] = [dict(option) for option in self.options]
            data["moreSections"] = list(self.more_sections)
            data["topic"] = self.topic.to_dict()
        return data

    def _public_points(self, evidence_registry: EvidenceRegistry | None) -> list[dict]:
        points = [dict(point) for point in self.points]
        if evidence_registry is None or self.document is None:
            return points
        hashes = {
            member["path"]: member["sha256"]
            for member in self.document.get("paths") or [
                {"path": self.document["path"], "sha256": self.document.get("sha256")}
            ]
        }
        for point in points:
            if "notes" in point:
                point["notes"] = [dict(note) for note in point["notes"]]
            path = point.get("path")
            for span in [point, *point.get("notes", [])]:
                evidence_id = evidence_registry.register(
                    session_id=self.session_id,
                    relative_path=path,
                    start_line=span.get("startLine"),
                    end_line=span.get("endLine"),
                    source_sha256=hashes.get(path),
                )
                if evidence_id:
                    span["evidenceId"] = evidence_id
        return points


class ConfirmationStore:
    """確認IDごとの状態。期限・件数上限・同時操作の直列化を管理する。"""

    def __init__(
        self,
        *,
        idle_seconds: float = CONFIRMATION_IDLE_SECONDS,
        max_age_seconds: float = CONFIRMATION_MAX_AGE_SECONDS,
        limit: int = CONFIRMATION_LIMIT,
        clock=time.monotonic,
        evidence_registry: EvidenceRegistry | None = None,
    ):
        self.idle_seconds = idle_seconds
        self.max_age_seconds = max_age_seconds
        self.limit = max(1, int(limit))
        self._clock = clock
        self._evidence_registry = evidence_registry
        self._records: dict[str, ConfirmationRecord] = {}
        self._lock = threading.Lock()

    def _expired(self, record: ConfirmationRecord, now: float) -> bool:
        return (
            now - record.last_completed_at > self.idle_seconds
            or now - record.created_at > self.max_age_seconds
        )

    def _purge_locked(self, now: float) -> None:
        for cid in [cid for cid, rec in self._records.items() if self._expired(rec, now)]:
            self._records.pop(cid, None)

    def create(self, session_id: str, initial_query: str, decision: Decision) -> ConfirmationRecord:
        with self._lock:
            now = self._clock()
            self._purge_locked(now)
            while len(self._records) >= self.limit:
                oldest = min(self._records.values(), key=lambda rec: rec.updated_at)
                self._records.pop(oldest.confirmation_id, None)
            cid = secrets.token_urlsafe(18)
            record = ConfirmationRecord(
                confirmation_id=cid,
                session_id=session_id,
                initial_query=initial_query,
                state=STATE_DOCUMENT_PENDING,
                created_at=now,
                updated_at=now,
                last_completed_at=now,
            )
            _apply_decision(record, decision)
            self._records[cid] = record
            return record

    def _get_locked(self, confirmation_id: object, session_id: str) -> ConfirmationRecord:
        if not valid_confirmation_id(confirmation_id):
            raise ConfirmationError("confirmation_invalid", "確認IDが不正です")
        now = self._clock()
        self._purge_locked(now)
        record = self._records.get(confirmation_id)
        if record is None:
            raise ConfirmationError(
                "confirmation_expired",
                "確認の有効期限が切れたか、サーバーが再起動しました。もう一度質問してください。",
            )
        if not secrets.compare_digest(record.session_id, session_id or ""):
            raise ConfirmationError("confirmation_invalid", "この確認を続ける権限がありません")
        return record

    def view(self, confirmation_id: object, session_id: str) -> dict:
        """パネル復元用の読み取り。期限を延長しない。"""
        with self._lock:
            return self._get_locked(confirmation_id, session_id).public(self._evidence_registry)

    def begin(
        self, confirmation_id: object, session_id: str, *, action: str, options_version: object, request_id: str
    ) -> ConfirmationRecord:
        """新しい続きの操作を受け付け、同じ確認IDの操作を直列化する。"""
        if action not in ACTIONS:
            raise ConfirmationError("confirmation_invalid", "操作種別が不正です")
        with self._lock:
            record = self._get_locked(confirmation_id, session_id)
            pending_actions = {ACTION_SELECT_DOCUMENT, ACTION_HINT}
            if (record.state == STATE_DOCUMENT_PENDING) != (action in pending_actions):
                raise ConfirmationError("confirmation_invalid", "現在の状態に合わない操作です")
            if action in {ACTION_SELECT_DOCUMENT, ACTION_SELECT_TOPIC} and options_version != record.options_version:
                raise ConfirmationError("confirmation_invalid", "選択肢が更新されています。表示を更新してから選び直してください。")
            if record.busy_request and record.busy_request != request_id:
                raise ConfirmationError("confirmation_busy", "この確認では別の質問を処理中です。終わってから送信してください。")
            record.busy_request = request_id
            return _snapshot(record)

    def finish(self, confirmation_id: str, request_id: str, *, decision: Decision | None = None,
               topic: Topic | None = None, options: list[dict] | None = None,
               add_paths: list[dict] | None = None, completed: bool) -> dict | None:
        """操作の終了。正常完了だけ期限と論点を更新する。失効・破棄済みなら復活させない。"""
        with self._lock:
            record = self._records.get(confirmation_id)
            if record is None or record.busy_request != request_id:
                return None
            record.busy_request = ""
            if not completed:
                return None
            now = self._clock()
            if self._expired(record, now):
                self._records.pop(confirmation_id, None)
                return None
            if decision is not None:
                _apply_decision(record, decision)
            if topic is not None:
                record.topic = Topic(topic.section, tuple(topic.targets), tuple(topic.aspects))
            if add_paths and record.document is not None:
                # 関連資料を以後の検索・根拠の対象に加える（固定の解除で消える）。
                current = list(record.document.get("paths") or [
                    {"path": record.document["path"], "sha256": record.document.get("sha256")}
                ])
                known = {m["path"] for m in current}
                record.document["paths"] = current + [dict(m) for m in add_paths if m["path"] not in known]
            if options is not None:
                # 聞き返しの候補を新しい選択肢一覧にする（番号入力・古い番号の照合に使う）。
                record.options = [dict(option) for option in options]
                record.options_version += 1
            record.updated_at = now
            record.last_completed_at = now
            return record.public(self._evidence_registry)

    def discard(self, confirmation_id: str) -> None:
        with self._lock:
            self._records.pop(confirmation_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def _snapshot(record: ConfirmationRecord) -> ConfirmationRecord:
    return ConfirmationRecord(
        confirmation_id=record.confirmation_id,
        session_id=record.session_id,
        initial_query=record.initial_query,
        state=record.state,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_completed_at=record.last_completed_at,
        candidates=[dict(card) for card in record.candidates],
        options=[dict(option) for option in record.options],
        options_version=record.options_version,
        document=dict(record.document) if record.document else None,
        topic=Topic(record.topic.section, tuple(record.topic.targets), tuple(record.topic.aspects)),
        message=record.message,
        points=[dict(point) for point in record.points],
        more_sections=list(record.more_sections),
        busy_request=record.busy_request,
    )


def _apply_decision(record: ConfirmationRecord, decision: Decision) -> None:
    record.message = decision.message
    if decision.kind in {"documents", "hints"}:
        record.state = STATE_DOCUMENT_PENDING
        record.candidates = list(decision.candidates)
        record.options_version += 1
        return
    if decision.document is not None:
        changed = not record.document or record.document.get("path") != decision.document["path"]
        record.state = STATE_DOCUMENT_FIXED
        record.document = {
            key: decision.document.get(key)
            for key in ("path", "title", "titles", "sha256", "paths", "sourceRoot", "indexMarker")
        }
        record.candidates = []
        if changed:
            record.topic = Topic()
    if decision.overview is not None:
        record.points = list(decision.overview["points"])
        record.options = list(decision.overview["options"])
        record.more_sections = list(decision.overview["moreSections"])
        record.options_version += 1


def clarification_event(record_public: dict, *, kind: str, ask: dict | None = None) -> dict:
    """SSEの確認結果イベント（``clarification``）。"""
    event = {"type": "clarification", "kind": kind, **record_public}
    if ask is not None:
        event["ask"] = ask
    return event
