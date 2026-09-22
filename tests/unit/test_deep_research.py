"""deep 調査モードの予算・原文範囲・世代固定契約。"""

import json

import pytest

import deep_research
import search


def test_validate_deep_timeout_keeps_normal_range_separate():
    assert deep_research.validate_deep_timeout_seconds("1800") == 1800
    assert deep_research.validate_deep_timeout_seconds(300) == 300
    with pytest.raises(ValueError):
        deep_research.validate_deep_timeout_seconds(299)
    with pytest.raises(ValueError):
        deep_research.validate_deep_timeout_seconds(1801)


def test_split_line_range_keeps_long_line_character_offsets():
    line = "あ" * 8500
    units = deep_research._split_line_range([line], 1, 1, max_chars=4000)
    assert [unit["text"] for unit in units] == [
        line[:4000],
        line[4000:8000],
        line[8000:],
    ]
    assert [(unit["char_start"], unit["char_end"]) for unit in units] == [
        (0, 3999),
        (4000, 7999),
        (8000, 8499),
    ]


def test_section_range_selects_inner_section_without_losing_heading_context():
    text = "# 規程\n導入\n## 条件\n適用条件\n## 例外\n除外条件\n"
    assert deep_research._section_range(text, 4, 4) == (3, 4, "条件")
    assert deep_research._section_range(text, 6, 6) == (5, 6, "例外")


def test_run_deep_research_reads_multiple_sections_and_validates_citations(tmp_path, monkeypatch):
    root = tmp_path / "skill-source"
    root.mkdir()
    source = root / "notice.md"
    source.write_text(
        "# 公共調達委員会\n導入文\n## 付議条件\n条件Aを満たす場合。\n## 除外\n例外Bは対象外。\n",
        encoding="utf-8",
    )
    chunks = search.build_source_chunks(root)
    by_heading = {chunk["heading"]: chunk for chunk in chunks if chunk.get("heading")}
    candidates = [
        {
            "path": "notice.md",
            "chunk_id": by_heading["付議条件"]["chunk_id"],
            "heading": "付議条件",
            "start_line": by_heading["付議条件"]["start_line"],
            "end_line": by_heading["付議条件"]["end_line"],
            "score": 2,
            "rrf_score": 0.5,
            "source_sha256": by_heading["付議条件"]["file_sha256"],
        },
        {
            "path": "notice.md",
            "chunk_id": by_heading["除外"]["chunk_id"],
            "heading": "除外",
            "start_line": by_heading["除外"]["start_line"],
            "end_line": by_heading["除外"]["end_line"],
            "score": 1,
            "rrf_score": 0.4,
            "source_sha256": by_heading["除外"]["file_sha256"],
        },
    ]

    monkeypatch.setattr(deep_research, "_retrieve_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(
        deep_research,
        "_call_ollama_json",
        lambda *args, **kwargs: (
            {
                "supported": True,
                "contradictions": [],
                "missing_conditions": [],
                "unsupported_claims": [],
            }
            if args[1] == deep_research.DEEP_VERIFY_SYSTEM
            else {"queries": [], "unresolved": []}
            if args[1] == deep_research.DEEP_GAPS_SYSTEM
            else {
                "subject": "公共調達委員会",
                "scope": "通知の適用範囲",
                "conditions": ["本文に記載された条件"],
                "exceptions": ["本文に記載された例外"],
                "references": [],
            }
        ),
    )
    monkeypatch.setattr(
        deep_research,
        "_call_ollama_text",
        lambda *args, **kwargs: "条件Aを確認した。[E1]\n例外Bを確認した。[E2]",
    )

    result = deep_research.run_deep_research(
        "公共調達委員会の条件",
        model="test-model",
        source_root=root,
        timeout_seconds=300,
    )

    assert result.status in {"completed", "partial"}
    assert len(result.evidence) == 3
    assert {(item["start_line"], item["end_line"]) for item in result.evidence} == {
        (1, 2),
        (3, 4),
        (5, 6),
    }
    assert result.answer.startswith("条件Aを確認した")
    public = result.to_public_dict()
    assert "excerpt" in public["evidence"][0]
    assert "条件Aを確認した" not in json.dumps(public["diagnostics"], ensure_ascii=False)


def test_run_deep_research_does_not_mix_changed_source_generation(tmp_path, monkeypatch):
    root = tmp_path / "skill-source"
    root.mkdir()
    source = root / "notice.md"
    source.write_text("# 規程\n本文\n", encoding="utf-8")
    chunk = search.build_source_chunks(root)[0]
    candidate = {
        "path": "notice.md",
        "chunk_id": chunk["chunk_id"],
        "heading": chunk["heading"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
        "source_sha256": "different-generation",
    }
    monkeypatch.setattr(deep_research, "_retrieve_candidates", lambda *args, **kwargs: [candidate])
    result = deep_research.run_deep_research(
        "規程",
        model="",
        source_root=root,
        timeout_seconds=300,
    )
    assert result.status == "failed"
    assert result.stop_reason == "source_changed"
    assert result.evidence == []
    assert any("source_changed" in item for item in result.unconfirmed)


def test_run_deep_research_returns_cancelled_terminal_state(tmp_path):
    class Cancelled(Exception):
        code = "cancelled"

    (tmp_path / "notice.md").write_text("# 規程\n本文\n", encoding="utf-8")

    def cancel():
        raise Cancelled()

    result = deep_research.run_deep_research(
        "規程",
        model="",
        source_root=tmp_path,
        timeout_seconds=300,
        cancel_check=cancel,
    )
    assert result.status == "cancelled"
    assert result.stop_reason == "cancelled"
