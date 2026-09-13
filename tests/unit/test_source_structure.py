"""_internal/source_structure.py の見出しツリー解析・展開範囲計算の単体テスト。

対象: H1→H2→H3、レベル飛び、同名見出し、EOF、空の節、同階層への漏出、
コードフェンス、長い配下、通常の短文（見出しなし）。
"""

import os

os.environ.pop("OLLAMA_HOST", None)

import source_structure as ss  # noqa: E402


def _tree(text):
    return ss.parse_heading_tree(text)


def test_h1_h2_h3_builds_nested_parent_chain():
    text = "# 一章\n本文A\n## 一節\n本文B\n### 一項\n本文C\n"
    nodes = _tree(text)
    assert [n.level for n in nodes] == [1, 2, 3]
    h1, h2, h3 = nodes
    assert h1.parent_index is None
    assert h2.parent_index == h1.index
    assert h3.parent_index == h2.index
    assert h2.index in h1.children_indices
    assert h3.index in h2.children_indices


def test_level_skip_h1_directly_to_h3_parents_to_h1():
    text = "# 一章\n本文A\n### 深い項\n本文B\n"
    nodes = _tree(text)
    h1, h3 = nodes
    assert h1.level == 1
    assert h3.level == 3
    assert h3.parent_index == h1.index


def test_same_heading_text_distinguished_by_line_position():
    text = "# 記\n本文1\n# 記\n本文2\n"
    nodes = _tree(text)
    assert len(nodes) == 2
    assert nodes[0].heading_text == "記"
    assert nodes[1].heading_text == "記"
    assert nodes[0].heading_line != nodes[1].heading_line
    assert nodes[0].index != nodes[1].index


def test_eof_closes_open_sections_at_last_line():
    text = "# 一章\n本文A\n本文B\n本文C"
    nodes = _tree(text)
    (h1,) = nodes
    total_lines = len(text.splitlines())
    assert h1.section_end_line == total_lines


def test_empty_section_has_no_body_lines():
    text = "# 一章\n# 二章\n本文\n"
    nodes = _tree(text)
    h1, h2 = nodes
    # 一章の直後に二章が来るため、一章の節は本文を持たない。
    assert h1.section_end_line == h1.heading_line
    assert h1.body_start_line > h1.section_end_line


def test_sibling_section_does_not_leak_into_previous_section():
    text = "# 一章\n## 子1\n本文1\n## 子2\n本文2\n"
    nodes = _tree(text)
    h1, child1, child2 = nodes
    assert child1.section_end_line == child2.heading_line - 1
    assert child2.section_end_line == len(text.splitlines())
    assert h1.section_end_line == len(text.splitlines())


def test_code_fence_backtick_does_not_create_heading():
    text = "# 一章\n```\n# これは見出しではない\n```\n本文\n"
    nodes = _tree(text)
    assert len(nodes) == 1
    assert nodes[0].heading_text == "一章"


def test_code_fence_tilde_does_not_create_heading():
    text = "# 一章\n~~~\n# これも見出しではない\n~~~\n本文\n"
    nodes = _tree(text)
    assert len(nodes) == 1


def test_unclosed_fence_suppresses_headings_to_eof():
    text = "# 一章\n```\n# 見出し風\n本文\n"
    nodes = _tree(text)
    assert len(nodes) == 1
    assert nodes[0].heading_text == "一章"


def test_long_subtree_is_captured_as_single_section():
    body_lines = "\n".join(f"本文行{i}" for i in range(200))
    text = f"# 概要\n## 手順\n{body_lines}\n"
    nodes = _tree(text)
    h1, h2 = nodes
    assert h2.section_end_line - h2.body_start_line + 1 == 200


def test_plain_short_text_without_heading_yields_no_nodes():
    text = "見出しのない短い本文です。\n"
    nodes = _tree(text)
    assert nodes == []


def test_closing_hash_sequence_is_stripped():
    text = "## 見出し ##\n本文\n"
    nodes = _tree(text)
    assert nodes[0].heading_text == "見出し"


def test_heading_without_closing_hash_is_unaffected():
    text = "## 見出し\n本文\n"
    nodes = _tree(text)
    assert nodes[0].heading_text == "見出し"


# --- find_expandable_parent / expand_range -----------------------------------


def test_find_expandable_parent_matches_blank_gap_and_start_line():
    text = "# 親\n\n## 子\n本文\n"
    lines = text.splitlines()
    nodes = _tree(text)
    parent = nodes[0]
    found = ss.find_expandable_parent(nodes, lines, start_line=parent.heading_line)
    assert found is not None
    assert found.index == parent.index


def test_find_expandable_parent_rejects_non_blank_gap():
    text = "# 親\n親自身の本文\n## 子\n本文\n"
    lines = text.splitlines()
    nodes = _tree(text)
    parent = nodes[0]
    found = ss.find_expandable_parent(nodes, lines, start_line=parent.heading_line)
    assert found is None


def test_find_expandable_parent_rejects_leaf_without_children():
    text = "# 親\n本文のみ\n"
    lines = text.splitlines()
    nodes = _tree(text)
    parent = nodes[0]
    found = ss.find_expandable_parent(nodes, lines, start_line=parent.heading_line)
    assert found is None


def test_find_expandable_parent_returns_none_for_unmatched_start_line():
    text = "# 親\n\n## 子\n本文\n"
    lines = text.splitlines()
    nodes = _tree(text)
    found = ss.find_expandable_parent(nodes, lines, start_line=999)
    assert found is None


def test_expand_range_covers_all_descendants_in_document_order():
    text = "# 親\n\n## 子1\n本文1\n## 子2\n本文2\n"
    nodes = _tree(text)
    parent = nodes[0]
    start, end = ss.expand_range(nodes, parent)
    assert start == nodes[1].heading_line  # 子1の見出し行から
    assert end == len(text.splitlines())


def test_expand_range_without_children_returns_own_body():
    text = "# 親\n本文のみ\n"
    nodes = _tree(text)
    parent = nodes[0]
    start, end = ss.expand_range(nodes, parent)
    assert start == parent.body_start_line
    assert end == parent.section_end_line
