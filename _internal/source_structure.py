"""資料の見出し階層を解析し、親候補から配下本文への展開範囲を計算する。

検索用チャンク分割（``search._line_chunks``）とは独立した派生構造である。
既存 chunk・chunk_id・text_sha256・Embedding 入力には一切影響しない。
同一資料 snapshot の bytes から生成し、ディスクへは保存しない
（ライフサイクルは呼び出し側の source chunk memo と揃える）。

対応範囲（対象外は曖昧なまま階層化しない）:
- ATX 見出し（``#`` 〜 ``######``）のみを見出しとして扱う。
- バッククォート/チルダのコードフェンス内は見出し判定から除外する。
- ATX のレベル飛び（例: H1 の直下に H3）を許容し、直近の上位見出しを親とする。
- 見出し末尾の閉じ ``#`` 列（CommonMark ATX closing sequence）を除去する。
- Setext 見出し・インデントコード・表・frontmatter は見出しとして解釈しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

# 構造 parser の契約バージョン。解析規則を変更したら上げる。ファイル
# SHA-256 と組み合わせて構造 memo の世代キーに使う想定（呼び出し側の責務）。
STRUCTURE_PARSER_VERSION = 1

_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:\s+(.*))?$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_TRAILING_CLOSING_HASHES_RE = re.compile(r"(?:^|\s)#+\s*$")


@dataclass
class HeadingNode:
    """1つの ATX 見出しとその節（配下の子孫を含む）を表す。"""

    index: int
    level: int
    heading_text: str
    heading_line: int  # 1-indexed
    parent_index: int | None
    children_indices: list[int] = field(default_factory=list)
    # 節の終端＝次の同階層・上位階層見出しの直前、または EOF（inclusive, 1-indexed）。
    section_end_line: int = 0
    # 見出し行の次の行（heading_line + 1）。節が空なら body_start_line > section_end_line。
    body_start_line: int = 0


def _strip_closing_hashes(text: str) -> str:
    """CommonMark ATX の末尾閉じ ``#`` 列を取り除く（例: ``Heading ##`` -> ``Heading``）。"""
    stripped = text.strip()
    if not stripped:
        return stripped
    match = _TRAILING_CLOSING_HASHES_RE.search(stripped)
    if not match:
        return stripped
    trimmed = stripped[: match.start()].rstrip()
    # 全体が閉じ記号だけ（例: "###"）だった場合は空文字列を返す。
    return trimmed


def _close_section(nodes: list[HeadingNode], index: int, end_line: int) -> None:
    node = nodes[index]
    node.section_end_line = max(end_line, node.heading_line)


def parse_heading_tree(text: str) -> list[HeadingNode]:
    """本文からATX見出しツリーを構築する。行番号はすべて1-indexed。"""
    lines = text.splitlines()
    nodes: list[HeadingNode] = []
    stack: list[int] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line_no, line in enumerate(lines, start=1):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            char = marker[0]
            length = len(marker)
            if not in_fence:
                in_fence = True
                fence_char = char
                fence_len = length
                continue
            if char == fence_char and length >= fence_len:
                in_fence = False
                continue
            # 別種・短い fence 記号は fence 本文として無視する。
            continue
        if in_fence:
            continue

        heading_match = _ATX_HEADING_RE.match(line)
        if not heading_match:
            continue
        level = len(heading_match.group(1))
        raw_text = (heading_match.group(2) or "").strip()
        heading_text = _strip_closing_hashes(raw_text)

        while stack and nodes[stack[-1]].level >= level:
            _close_section(nodes, stack.pop(), line_no - 1)

        parent_index = stack[-1] if stack else None
        node = HeadingNode(
            index=len(nodes),
            level=level,
            heading_text=heading_text,
            heading_line=line_no,
            parent_index=parent_index,
            body_start_line=line_no + 1,
        )
        nodes.append(node)
        if parent_index is not None:
            nodes[parent_index].children_indices.append(node.index)
        stack.append(node.index)

    total_lines = len(lines)
    while stack:
        _close_section(nodes, stack.pop(), total_lines)

    return nodes


def is_blank_child_gap(lines: list[str], parent: HeadingNode, first_child: HeadingNode) -> bool:
    """親見出し行の次行から最初の子見出しの直前までが空行だけかを判定する。

    見出し自体（親行・子行）は判定対象に含めない。間に行が無い（見出しが
    連続する）場合は空行判定を満たすとみなす。
    """
    start = parent.heading_line + 1
    end = first_child.heading_line - 1
    if start > end:
        return True
    for line_no in range(start, end + 1):
        if line_no - 1 >= len(lines):
            continue
        if lines[line_no - 1].strip():
            return False
    return True


def find_expandable_parent(
    nodes: list[HeadingNode], lines: list[str], *, start_line: int
) -> HeadingNode | None:
    """展開資格を持つ親候補を返す。

    資格条件（両方満たす場合のみ）:
    (a) 構造上の子見出しを持ち、親見出し行の次行から最初の子見出しの直前
        までが空行だけ
    (b) 候補の ``start_line`` が当該親見出し行に一致する

    条件を満たさない、または該当する見出しが存在しない場合は ``None``。
    """
    for node in nodes:
        if node.heading_line != start_line:
            continue
        if not node.children_indices:
            return None
        first_child = nodes[node.children_indices[0]]
        if not is_blank_child_gap(lines, node, first_child):
            return None
        return node
    return None


def expand_range(nodes: list[HeadingNode], node: HeadingNode) -> tuple[int, int]:
    """配下本文（子孫すべてを含む）の実在範囲 ``(start_line, end_line)`` を返す。

    親見出し行自体は範囲に含めない。子を持たない場合は親自身の本文範囲を返す
    （呼び出し側は ``find_expandable_parent`` で子の有無を確認済みの前提）。
    """
    if not node.children_indices:
        return (node.body_start_line, node.section_end_line)
    first_child = nodes[node.children_indices[0]]
    return (first_child.heading_line, node.section_end_line)


def node_by_heading_line(nodes: list[HeadingNode], heading_line: int) -> HeadingNode | None:
    for node in nodes:
        if node.heading_line == heading_line:
            return node
    return None


def node_for_line(nodes: list[HeadingNode], line_no: int) -> HeadingNode | None:
    """本文行を最も内側のATX節へ割り当てる。

    ``node_by_heading_line`` は見出し行の完全一致専用であるため、本文ヒット
    の所属節を求めるdeep調査や参照範囲の構築ではこちらを使う。行番号は
    既存APIと同じく1-indexedで、同じ行を含む候補のうち見出しが最も後ろの
    ノードを返す。
    """
    containing = [
        node
        for node in nodes
        if node.heading_line <= line_no <= max(node.section_end_line, node.heading_line)
    ]
    return max(containing, key=lambda node: node.heading_line) if containing else None


def section_range_for_line(
    nodes: list[HeadingNode],
    line_no: int,
    *,
    include_heading: bool = True,
    preserve_parent_intro: bool = True,
) -> tuple[int, int, str]:
    """本文ヒットを含む節の実在行範囲と見出し名を返す。

    子を持つ親節の導入本文にヒットした場合は、最初の子節の直前で範囲を
    閉じる。これにより親の但書・適用条件を残しながら、無関係な兄弟節を
    読み込まない。見出しより前の本文は ``(1, first_heading - 1, "")``、
    見出しがない資料は呼び出し側で本文全体を扱えるよう ``(1, 0, "")`` を
    返す。
    """
    if not nodes:
        return 1, 0, ""
    node = node_for_line(nodes, line_no)
    if node is None:
        first_heading = min(item.heading_line for item in nodes)
        return 1, first_heading - 1, ""

    start = node.heading_line if include_heading else node.body_start_line
    end = node.section_end_line
    if preserve_parent_intro and node.children_indices:
        first_child = nodes[node.children_indices[0]]
        if line_no < first_child.heading_line:
            end = first_child.heading_line - 1
    return start, end, node.heading_text


def direct_child_ranges(nodes: list[HeadingNode], node: HeadingNode) -> list[tuple[int, int]]:
    """親の直接の子見出しごとに、見出し行から自身の節の終端（孫を含む）までの
    範囲を返す。

    ``expand_range`` が配下全体を1つの連続範囲として返すのに対し、こちらは
    「子節単位の選択」（予算超過時に子の類似度で優先順位付けする）のための
    候補粒度を提供する。子見出し同士が行範囲上連続していても、選択の単位
    としては別々の候補のままにする。
    """
    return [
        (nodes[child_index].heading_line, nodes[child_index].section_end_line)
        for child_index in node.children_indices
    ]


# ---------------------------------------------------------------------------
# 確定した資料の主要範囲と、資料の照合（plans/offline-ai-guided-follow-up-plan.md 4.1・4.2）
# ---------------------------------------------------------------------------

# 付随的な節の見出し。質問が明示しない限り、主要範囲より後に回す。
_ANCILLARY_HEADING_RE = re.compile(
    r"宛先|宛て|発出|発信|照会|問い?合わ?せ|連絡先|担当|様式|適用日|適用期日|施行日|施行期日|附則|付則"
)
# 見出しのない段落（宛名・日付・署名・連絡先）だけで構成される付随範囲。
_ANCILLARY_LINE_RE = re.compile(
    r"殿\s*$|様\s*$|^\s*(?:令和|平成|昭和)?\s*[0-9０-９元]+\s*年\s*[0-9０-９]+\s*月\s*[0-9０-９]+\s*日\s*(?:付)?\s*$"
    r"|(?:課長|部長|局長|室長|所長|係長|担当)\s*$|内線|電話|TEL|E-?mail|メール"
)
# 付随見出しの下でも、適用条件・経過措置・例外がある本文は主要範囲として扱う。
_QUALIFYING_BODY_RE = re.compile(
    r"経過措置|ただし|但し|例外|除く|除き|条件|要件|限り|適用しない|適用除外|場合"
)


def _paragraph_ranges(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """空行で区切った段落の行範囲（1-indexed, inclusive）。"""
    ranges: list[tuple[int, int]] = []
    first = None
    for line_no in range(start, min(end, len(lines)) + 1):
        if lines[line_no - 1].strip():
            if first is None:
                first = line_no
        elif first is not None:
            ranges.append((first, line_no - 1))
            first = None
    if first is not None:
        ranges.append((first, min(end, len(lines))))
    return ranges


def _body_text(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : min(end, len(lines))])


def _is_ancillary(heading: str, body: str) -> bool:
    if _QUALIFYING_BODY_RE.search(body):
        return False
    if heading:
        return bool(_ANCILLARY_HEADING_RE.search(heading))
    content = [line for line in body.splitlines() if line.strip()]
    return bool(content) and all(_ANCILLARY_LINE_RE.search(line) for line in content)


def primary_units(text: str) -> list[dict]:
    """資料を主要範囲の選定単位へ分ける（原文順）。

    文書全体を包む表題見出しがあれば、その直下の節を個別の単位とし、表題の
    下の節外本文は段落を単位にする。見出しのない資料は段落を単位にする。
    見出しの記号数ではなく見出し木で単位を決める。各単位には付随範囲か
    どうか（``ancillary``）と、深掘りが節ごとに読む範囲（``read_ranges``: 見出し木の
    各節の導入本文。本文のない見出しだけの節は除く）を付ける。
    """
    lines = text.splitlines()
    nodes = parse_heading_tree(text)
    units: list[dict] = []

    def add_paragraphs(start: int, end: int) -> None:
        for first, last in _paragraph_ranges(lines, start, end):
            body = _body_text(lines, first, last)
            units.append(
                {
                    "heading": "",
                    "start_line": first,
                    "end_line": last,
                    "ancillary": _is_ancillary("", body),
                    "read_ranges": [(first, last, "")],
                }
            )

    if not nodes:
        add_paragraphs(1, len(lines))
        return units
    roots = [node for node in nodes if node.parent_index is None]
    if roots[0].heading_line > 1:
        add_paragraphs(1, roots[0].heading_line - 1)
    sections: list[HeadingNode]
    if len(roots) == 1 and roots[0].children_indices:
        wrapper = roots[0]
        add_paragraphs(wrapper.body_start_line, nodes[wrapper.children_indices[0]].heading_line - 1)
        sections = [nodes[index] for index in wrapper.children_indices]
    elif len(roots) == 1:
        add_paragraphs(roots[0].body_start_line, roots[0].section_end_line)
        return units
    else:
        sections = roots
    for node in sections:
        subtree = [node]
        pending = list(reversed(node.children_indices))
        while pending:
            child = nodes[pending.pop()]
            subtree.append(child)
            pending.extend(reversed(child.children_indices))
        body = _body_text(lines, node.body_start_line, node.section_end_line)
        read_ranges = []
        for item in subtree:
            intro_end = (
                nodes[item.children_indices[0]].heading_line - 1
                if item.children_indices
                else item.section_end_line
            )
            if any(line.strip() for line in lines[item.heading_line : intro_end]):
                read_ranges.append((item.heading_line, intro_end, item.heading_text))
        units.append(
            {
                "heading": node.heading_text,
                "start_line": node.heading_line,
                "end_line": node.section_end_line,
                "ancillary": _is_ancillary(node.heading_text, body),
                "read_ranges": read_ranges,
            }
        )
    return units


def primary_section_ranges(text: str, *, budget_chars: int) -> dict:
    """主要範囲を予算内で節ごとに配分し、各節の先頭から取得する範囲を返す。

    1つの節が予算を占有しないよう、まず主要単位へ均等に配分し、余りを原文順に
    追加配分する。予算内に収まらない主要範囲は ``unread`` に入れる。付随範囲
    （宛先・単独の適用日・照会先等）は ``ancillary`` に分け、取得範囲へ入れない。
    """
    lines = text.splitlines()
    units = primary_units(text)
    main = [unit for unit in units if not unit["ancillary"]]
    ancillary = [
        {"start_line": u["start_line"], "end_line": u["end_line"], "heading": u["heading"]}
        for u in units
        if u["ancillary"]
    ]
    taken = {id(unit): unit["start_line"] - 1 for unit in main}
    remaining = max(0, int(budget_chars))

    def extend(unit: dict, allowance: int) -> int:
        used = 0
        line_no = taken[id(unit)] + 1
        while line_no <= min(unit["end_line"], len(lines)):
            cost = len(lines[line_no - 1]) + 1
            if used + cost > allowance:
                break
            used += cost
            taken[id(unit)] = line_no
            line_no += 1
        return used

    if main:
        share = remaining // len(main)
        for unit in main:
            remaining -= extend(unit, share)
        for unit in main:
            if remaining <= 0:
                break
            remaining -= extend(unit, remaining)
    ranges, unread = [], []
    for unit in main:
        end = taken[id(unit)]
        if end >= unit["start_line"]:
            ranges.append(
                {
                    "start_line": unit["start_line"],
                    "end_line": end,
                    "heading": unit["heading"],
                    "section_end_line": unit["end_line"],
                    "complete": end >= unit["end_line"],
                }
            )
        if end < unit["end_line"]:
            unread.append(
                {
                    "start_line": max(unit["start_line"], end + 1),
                    "end_line": unit["end_line"],
                    "heading": unit["heading"],
                }
            )
    return {"ranges": ranges, "unread": unread, "ancillary": ancillary}


_MATCH_STRIP_RE = re.compile(r"[\s　、。，．,.・:：;；\"'“”‘’「」『』（）()\[\]【】〔〕<>＜＞/／\\|｜~〜～\-－―‐]+")
_TITLE_NOISE_RE = re.compile(
    r"[（(](?:通知|事務連絡|依頼|周知|お知らせ|案内|写し?)[）)]|について$|に関する(?:件|こと)?$|の件$"
)
_DOCUMENT_NUMBER_RE = re.compile(r"第\s*([0-9０-９]+)\s*号")
TITLE_CORE_MIN_CHARS = 4
TITLE_WEAK_COVERAGE = 0.5


def normalize_match_text(value: str) -> str:
    """表記ゆれ（全角・半角、空白、括弧、記号）を除いた照合用の文字列。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return _MATCH_STRIP_RE.sub("", text)


def title_core(title: str) -> str:
    """表題から（通知）や「について」等の定型部分を除いた中核部分。"""
    core = str(title or "").strip()
    previous = None
    while core and core != previous:
        previous = core
        core = _TITLE_NOISE_RE.sub("", core).strip()
    return core


def document_title(text: str, *, source_title: str | None = None, path: str = "") -> str:
    """資料の表題: メタデータの表題、最初の見出し、ファイル名の順に採る。"""
    if source_title and str(source_title).strip():
        return str(source_title).strip()
    nodes = parse_heading_tree(text)
    if nodes and nodes[0].heading_text:
        return nodes[0].heading_text
    stem = path.replace("\\", "/").rsplit("/", 1)[-1]
    return stem.rsplit(".", 1)[0] if "." in stem else stem


def _bigram_coverage(needle: str, haystack: str) -> float:
    grams = {needle[i : i + 2] for i in range(len(needle) - 1)}
    if not grams:
        return 0.0
    return sum(gram in haystack for gram in grams) / len(grams)


def document_numbers(text: str) -> set[str]:
    return {
        unicodedata.normalize("NFKC", match.group(1)).lstrip("0") or "0"
        for match in _DOCUMENT_NUMBER_RE.finditer(unicodedata.normalize("NFKC", str(text or "")))
    }


def match_documents(query: str, documents: list[dict]) -> dict:
    """質問と資料を照合し、資料単位の候補と、1つに確定した資料を返す。

    ``documents`` は ``{"path", "title", "head_text"}`` の一覧（資料単位）。
    表題の中核部分が質問に現れる場合と、文書番号が一致する場合を強い一致とし、
    強い一致がちょうど1資料のときだけ確定する。発出者だけの一致では確定しない。
    モデルを呼ばない決定的な照合である。
    """
    normalized_query = normalize_match_text(query)
    query_numbers = document_numbers(query)
    candidates = []
    for document in documents:
        title = str(document.get("title") or "")
        core = normalize_match_text(title_core(title))
        reasons = []
        strength = 0
        coverage = _bigram_coverage(core, normalized_query) if core else 0.0
        if len(core) >= TITLE_CORE_MIN_CHARS and core in normalized_query:
            strength, reasons = 2, ["title"]
        if query_numbers and query_numbers & document_numbers(str(document.get("head_text") or "")):
            strength = 2
            reasons.append("document_number")
        if strength == 0 and len(core) >= TITLE_CORE_MIN_CHARS and coverage >= TITLE_WEAK_COVERAGE:
            strength, reasons = 1, ["title_partial"]
        if strength:
            candidates.append(
                {
                    "path": str(document.get("path") or ""),
                    "title": title,
                    "strength": strength,
                    "coverage": round(coverage, 4),
                    "reasons": reasons,
                }
            )
    candidates.sort(key=lambda item: (-item["strength"], -item["coverage"], item["path"]))
    strong = [item for item in candidates if item["strength"] == 2]
    confirmed = strong[0]["path"] if len(strong) == 1 else None
    return {"confirmed": confirmed, "candidates": candidates}


def documents_from_texts(texts: dict[str, str], titles: dict[str, str] | None = None) -> list[dict]:
    """``{path: text}`` から照合用の資料一覧を作る（冒頭30行を文書番号照合に使う）。"""
    titles = titles or {}
    documents = []
    for path, text in texts.items():
        documents.append(
            {
                "path": path,
                "title": document_title(text, source_title=titles.get(path), path=path),
                "head_text": "\n".join(text.splitlines()[:30]),
            }
        )
    return documents


def documents_from_chunks(chunks: list[dict]) -> list[dict]:
    """検索チャンクから照合用の資料一覧を作る（資料ファイルを読み直さない）。

    表題はメタデータの ``source_title``、1行目の見出し、ファイル名の順に採る。
    文書番号の照合には冒頭30行に掛かるチャンク本文を使う。
    """
    by_path: dict[str, list[dict]] = {}
    for chunk in chunks:
        path = str(chunk.get("path") or "").replace("\\", "/")
        if path:
            by_path.setdefault(path, []).append(chunk)
    documents = []
    for path, items in by_path.items():
        items = sorted(items, key=lambda c: int(c.get("start_line") or 0))
        source_title = next((str(c["source_title"]) for c in items if c.get("source_title")), "")
        first_text = str(items[0].get("text") or "") if int(items[0].get("start_line") or 0) == 1 else ""
        head = "\n".join(str(c.get("text") or "") for c in items if int(c.get("start_line") or 0) <= 30)
        documents.append(
            {
                "path": path,
                "title": document_title(first_text, source_title=source_title, path=path),
                "head_text": head,
                "file_sha256": str(items[0].get("file_sha256") or ""),
            }
        )
    return documents
