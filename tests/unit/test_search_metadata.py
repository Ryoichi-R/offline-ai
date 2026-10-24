import json
import os

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402


def test_chunk_source_file_attaches_sidecar_metadata(tmp_path):
    source = tmp_path / "procedure.md"
    source.write_text("# 申請手順\n本文1\n本文2\n\n# 添付書類\n本文3", encoding="utf-8")
    sidecar = source.with_name(source.name + ".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "sourcePath": "procedure.pdf",
                "sourceTitle": "申請手順書",
                "parser": "structured-fixture",
                "warnings": ["OCR confidence is low"],
                "pages": [
                    {
                        "page": 3,
                        "blocks": [
                            {
                                "blockId": "b1",
                                "layoutType": "table",
                                "heading": "申請手順",
                                "text": "本文1",
                                "lineStart": 1,
                                "lineEnd": 3,
                                "page": 3,
                                "confidence": 0.91,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    chunks = search.chunk_source_file(source, "docs/procedure.md", max_chars=200)

    assert chunks[0]["parser"] == "structured-fixture"
    assert chunks[0]["page"] == 3
    assert chunks[0]["layout_type"] == "table"
    assert chunks[0]["source_title"] == "申請手順書"
    assert chunks[0]["parser_warnings"] == ["OCR confidence is low"]


def test_merge_results_preserves_metadata():
    keyword = [
        {
            "path": "docs/procedure.md",
            "chunk_id": "docs/procedure.md#0001",
            "snippet": "申請",
            "source": "keyword",
            "page": 2,
            "layout_type": "text",
            "parser_warnings": ["table boundary uncertain"],
        }
    ]
    embed = [
        {
            "path": "docs/procedure.md",
            "chunk_id": "docs/procedure.md#0001",
            "snippet": "",
            "source": "embedding",
            "parser": "pdftotext",
        }
    ]

    merged = search.merge_results(keyword, embed)

    assert merged[0]["page"] == 2
    assert merged[0]["parser"] == "pdftotext"
    assert merged[0]["layout_type"] == "text"
    assert merged[0]["parser_warnings"] == ["table boundary uncertain"]
