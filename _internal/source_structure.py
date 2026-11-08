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
    r"|実施時期|施行時期|適用時期"
)
# 見出しのない段落（宛名・日付・署名・連絡先）だけで構成される付随範囲。
_ANCILLARY_LINE_RE = re.compile(
    r"殿\s*$|様\s*$|^\s*(?:令和|平成|昭和)?\s*[0-9０-９元]+\s*年\s*[0-9０-９]+\s*月\s*[0-9０-９]+\s*日\s*(?:付)?\s*$"
    r"|(?:課長|部長|局長|室長|所長|係長|担当|の長)\s*(?:[（(][^）)]*[）)])?\s*$|内線|電話|TEL|E-?mail|メール"
    r"|第\s*[0-9０-９]+\s*号\s*$|^\s*事務連絡\s*$|^\s*【[^】]*】\s*$|^\s*(?:記|以上)\s*$"
)
# 行頭の引用記号・強調記号は付随行の判定に含めない。
_LINE_MARKUP_RE = re.compile(r"^[\s>＞*＊_]+|[\s*＊_]+$")
# 最初の見出しより前の短い行だけの段落（宛先・発出者・文書番号の並び）は付随範囲とする。
HEADER_PARAGRAPH_MAX_LINE_CHARS = 40
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


def _is_ancillary(heading: str, body: str, *, header: bool = False) -> bool:
    content = [_LINE_MARKUP_RE.sub("", line) for line in body.splitlines() if line.strip()]
    content = [line for line in content if line]
    if not heading and content and all(re.fullmatch(r"(?:記|以上)", line) for line in content):
        return True  # 「記」「以上」だけの段落
    if _QUALIFYING_BODY_RE.search(body):
        return False
    if heading:
        return bool(_ANCILLARY_HEADING_RE.search(heading))
    if header and content and all(
        "。" not in line and len(line) <= HEADER_PARAGRAPH_MAX_LINE_CHARS for line in content
    ):
        return True
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

    def add_paragraphs(start: int, end: int, *, header: bool = False) -> None:
        for first, last in _paragraph_ranges(lines, start, end):
            body = _body_text(lines, first, last)
            units.append(
                {
                    "heading": "",
                    "start_line": first,
                    "end_line": last,
                    "ancillary": _is_ancillary("", body, header=header),
                    "read_ranges": [(first, last, "")],
                }
            )

    if not nodes:
        add_paragraphs(1, len(lines))
        return units
    roots = [node for node in nodes if node.parent_index is None]
    if roots[0].heading_line > 1:
        add_paragraphs(1, roots[0].heading_line - 1, header=True)
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
    r"[（(](?:通知|事務連絡|依頼|周知|お知らせ|案内|写し?|別添|別紙|参考)[0-9０-９]*[）)]|について$|に関する(?:件|こと)?$|の件$"
)
_BRACKET_RE = re.compile(r"[（(]([^（）()]+)[）)]")
_FILE_PREFIX_RE = re.compile(r"^(?:[0-9０-９]+[．.、]\s*|[A-Za-z]{0,3}[-_]?[0-9]{1,4}[-_]\s*)")
# 変換・分割で付いたファイル名の接尾部（-resegmented-01、_part7 等）は表題に含めない。
_FILE_SUFFIX_RE = re.compile(r"(?:[-_](?:resegmented|part|section|sec)[-_]?[0-9]+[A-Za-z0-9_\-]*)+$", re.IGNORECASE)
# 発出者（役職）と文書種別。発出者だけの一致では資料を確定しない。
_ISSUER_RE = re.compile(
    r"[一-龥ァ-ヴー]{0,8}?(?:課長|室長|局長|部長|次長|所長|官房長|審議官|次官|大臣|センター長|専門官)"
)
_KIND_WORDS = ("通知", "事務連絡", "通達", "依頼", "周知")
# 文書番号: 「会発1225第２号」のような接頭部を含めて丸ごと照合する。
_DOCUMENT_NUMBER_RE = re.compile(r"([一-龥ァ-ヶA-Za-z]{0,4}[0-9]*)第([0-9]+)号")
_SUBJECT_TERM_RE = re.compile(r"[一-龥々]{2,}|[ァ-ヴー]{2,}|[A-Za-z][A-Za-z0-9\-]{1,}")
_GENERIC_QUERY_TERMS = frozenset(
    {
        "内容", "詳細", "内容詳細", "概要", "要点", "全体", "説明", "取扱", "取扱い", "取り扱い",
        "資料", "文書", "通知", "事務連絡", "通達", "教", "関", "事", "求", "場合", "以下", "以上",
    }
)
TITLE_CORE_MIN_CHARS = 4
TITLE_WEAK_COVERAGE = 0.5
HEADER_MAX_LINES = 30


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


def _is_issuer_or_kind_only(text: str) -> bool:
    """「会計課長通知」のように発出者と文書種別だけでできた句か。"""
    rest = unicodedata.normalize("NFKC", text).strip()
    for word in _KIND_WORDS:
        rest = rest.replace(word, "")
    rest = _ISSUER_RE.sub("", rest)
    return not normalize_match_text(rest)


def title_variants(titles) -> list[str]:
    """表題の別表記（メタデータ・見出し・ファイル名）から照合用の中核候補を作る。

    「会計課長通知（会計手続における押印の見直しについて）」のように正式表題が
    括弧内にある場合は括弧内も候補にする。発出者と文書種別だけの句は候補にしない。
    """
    variants: list[str] = []

    def add(value: str) -> None:
        core = title_core(value)
        if core and not _is_issuer_or_kind_only(core) and core not in variants:
            variants.append(core)

    for raw in titles:
        title = _FILE_SUFFIX_RE.sub("", _FILE_PREFIX_RE.sub("", str(raw or "").strip()))
        if not title:
            continue
        add(title)
        for inner in _BRACKET_RE.findall(title):
            add(inner)
        add(_BRACKET_RE.sub("", title))
    return variants


def title_exclusions(titles) -> tuple[str, ...]:
    """質問範囲の判定で尋ねている項目から除く句（表題の別表記・発出者・文書種別）。"""
    values: list[str] = []
    for raw in titles:
        raw = str(raw or "").strip()
        if raw:
            values.append(raw)
            values.extend(_BRACKET_RE.findall(raw))
    values.extend(title_variants(titles))
    return tuple(dict.fromkeys(values))


def document_title(text: str, *, source_title: str | None = None, path: str = "") -> str:
    """資料の表題: メタデータの表題、最初の見出し、ファイル名の順に採る。"""
    if source_title and str(source_title).strip():
        return str(source_title).strip()
    nodes = parse_heading_tree(text)
    if nodes and nodes[0].heading_text:
        return nodes[0].heading_text
    return _path_stem(path)


def _path_stem(path: str) -> str:
    stem = str(path or "").replace("\\", "/").rsplit("/", 1)[-1]
    return stem.rsplit(".", 1)[0] if "." in stem else stem


def _first_heading(text: str) -> str:
    nodes = parse_heading_tree(text)
    return nodes[0].heading_text if nodes else ""


def _header_text(text: str) -> str:
    """文書番号・発出者を探す冒頭部: 最初の見出しより前（最大30行）。見出しが1行目なら冒頭10行。

    本文中で他の通知を引用した番号（「…付け会発…号をもって通知」等）を自身の番号と
    取り違えないため、見出しより後は使わない。
    """
    lines = str(text or "").splitlines()
    nodes = parse_heading_tree(text)
    first = nodes[0].heading_line if nodes else len(lines) + 1
    end = first - 1 if first > 1 else 10
    return "\n".join(lines[: min(end, HEADER_MAX_LINES)])


def _bigram_coverage(needle: str, haystack: str) -> float:
    grams = {needle[i : i + 2] for i in range(len(needle) - 1)}
    if not grams:
        return 0.0
    return sum(gram in haystack for gram in grams) / len(grams)


def document_numbers(text: str) -> set[tuple[str, str]]:
    """文書番号を ``(接頭部, 番号)`` の組で返す（全角・半角と空白を正規化）。"""
    normalized = re.sub(r"[ 　\t]+", "", unicodedata.normalize("NFKC", str(text or "")))
    numbers = set()
    for match in _DOCUMENT_NUMBER_RE.finditer(normalized):
        prefix = match.group(1).lower()
        number = match.group(2).lstrip("0") or "0"
        numbers.add((prefix, number))
    return numbers


def _numbers_match(query_numbers, document_numbers_) -> bool:
    for q_prefix, q_number in query_numbers:
        for d_prefix, d_number in document_numbers_:
            if q_number != d_number:
                continue
            # 接頭部（会発1225 等）を書いた質問は、接頭部まで一致した場合だけ強い一致とする。
            if not q_prefix or q_prefix == d_prefix:
                return True
    return False


def _issuers(text: str) -> set[str]:
    return {normalize_match_text(m.group(0)) for m in _ISSUER_RE.finditer(unicodedata.normalize("NFKC", text))}


def _subject_terms(query: str, issuers: set[str]) -> list[str]:
    terms = []
    for term in _SUBJECT_TERM_RE.findall(unicodedata.normalize("NFKC", query)):
        if term in _GENERIC_QUERY_TERMS:
            continue
        normalized = normalize_match_text(term)
        if any(issuer and issuer in normalized for issuer in issuers):
            continue  # 「会計課長通知」等は発出者の句であり主題ではない
        if any(word in term for word in _KIND_WORDS) and _is_issuer_or_kind_only(term):
            continue
        terms.append(normalized)
    return list(dict.fromkeys(terms))


def match_documents(query: str, documents: list[dict]) -> dict:
    """質問と資料を照合し、資料単位の候補と、1つに確定した資料を返す。

    ``documents`` は ``{"path", "title", "titles", "head_text"}`` の一覧（資料単位）。
    強い一致（確定に使う）: 表題の別表記の中核が質問に現れる、または文書番号が一致する。
    弱い一致（候補だけ）: 質問の発出者が資料の表題・冒頭に現れ主題の語が表題に現れる、
    主題の語がすべて表題に現れる、または表題の2文字単位の重なりが半分以上。強い一致がちょうど1資料のときだけ確定し、
    発出者だけの一致では確定しない。モデルを呼ばない決定的な照合である。
    """
    normalized_query = normalize_match_text(query)
    query_numbers = document_numbers(query)
    query_issuers = _issuers(query)
    subjects = _subject_terms(query, query_issuers)
    candidates = []
    for document in documents:
        title = str(document.get("title") or "")
        titles = document.get("titles") or [title]
        variants = [normalize_match_text(v) for v in title_variants(titles)]
        reasons: list[str] = []
        strength = 0
        coverages = [(v, _bigram_coverage(v, normalized_query)) for v in variants]
        coverage = max((c for _, c in coverages), default=0.0)
        if any(len(v) >= TITLE_CORE_MIN_CHARS and v in normalized_query for v in variants):
            strength, reasons = 2, ["title"]
        if query_numbers and _numbers_match(query_numbers, document_numbers(str(document.get("head_text") or ""))):
            strength = 2
            reasons.append("document_number")
        title_text = normalize_match_text(" ".join(str(t or "") for t in titles))
        subject_hits = [term for term in subjects if term in title_text]
        if strength == 0:
            issuer_text = title_text + normalize_match_text(str(document.get("head_text") or ""))
            issuer_hit = any(issuer in issuer_text for issuer in query_issuers)
            if issuer_hit and subject_hits:
                strength, reasons = 1, ["issuer_subject"]
            elif subjects and len(subject_hits) == len(subjects) and sum(map(len, subjects)) >= TITLE_CORE_MIN_CHARS:
                # 表題を知らない質問: 主題の語がすべて表題に現れる資料を候補にする（確定しない）。
                strength, reasons = 1, ["subject"]
            elif any(len(v) >= TITLE_CORE_MIN_CHARS and c >= TITLE_WEAK_COVERAGE for v, c in coverages):
                strength, reasons = 1, ["title_partial"]
        if strength:
            candidates.append(
                {
                    "path": str(document.get("path") or ""),
                    "title": title,
                    "strength": strength,
                    "coverage": round(coverage, 4),
                    "subject_hits": len(subject_hits),
                    "reasons": reasons,
                    "exclude": list(title_exclusions(titles)),
                }
            )
    candidates.sort(
        key=lambda item: (-item["strength"], -item["subject_hits"], -item["coverage"], item["path"])
    )
    strong = [item for item in candidates if item["strength"] == 2]
    confirmed = strong[0]["path"] if len(strong) == 1 else None
    return {"confirmed": confirmed, "candidates": candidates}


def candidate_for(matched: dict, path: str) -> dict:
    return next(item for item in matched["candidates"] if item["path"] == path)


def documents_from_texts(texts: dict[str, str], titles: dict[str, str] | None = None) -> list[dict]:
    """``{path: text}`` から照合用の資料一覧を作る。"""
    titles = titles or {}
    documents = []
    for path, text in texts.items():
        variants = [titles.get(path) or "", _first_heading(text), _path_stem(path)]
        documents.append(
            {
                "path": path,
                "title": document_title(text, source_title=titles.get(path), path=path),
                "titles": [v for v in dict.fromkeys(variants) if v],
                "head_text": _header_text(text),
            }
        )
    return documents


def documents_from_chunks(chunks: list[dict]) -> list[dict]:
    """検索チャンクから照合用の資料一覧を作る（資料ファイルを読み直さない）。

    表題はメタデータの ``source_title``、最初の見出し、ファイル名の順に採り、別表記として
    すべて照合に使う。文書番号・発出者の照合には冒頭部（最初の見出しより前）を使う。
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
        heading = next((str(c.get("heading") or "") for c in items if c.get("heading")), "") or next(
            (_first_heading(str(c.get("text") or "")) for c in items if _first_heading(str(c.get("text") or ""))),
            "",
        )
        head_lines: list[str] = []
        for chunk in items:
            if int(chunk.get("start_line") or 0) > HEADER_MAX_LINES:
                break
            for line in str(chunk.get("text") or "").splitlines():
                if line.lstrip().startswith("#") and head_lines:
                    break
                head_lines.append(line)
            else:
                continue
            break
        if not head_lines and items:
            head_lines = str(items[0].get("text") or "").splitlines()[:10]
        variants = [source_title, heading, _path_stem(path)]
        documents.append(
            {
                "path": path,
                "title": source_title or heading or _path_stem(path),
                "titles": [v for v in dict.fromkeys(variants) if v],
                "head_text": "\n".join(head_lines[:HEADER_MAX_LINES]),
                "file_sha256": str(items[0].get("file_sha256") or ""),
            }
        )
    return documents
