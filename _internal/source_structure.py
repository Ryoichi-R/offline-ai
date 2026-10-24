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
