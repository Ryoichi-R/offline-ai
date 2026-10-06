import json

import pytest

import document_schema


def test_build_and_load_metadata_sidecar(tmp_path):
    pdf = tmp_path / "sample.pdf"
    md = tmp_path / "sample.md"
    md.write_text("# 手続\n本文", encoding="utf-8")

    metadata = document_schema.build_document_metadata(
        source_path=pdf,
        markdown_path=md,
        markdown_text=md.read_text(encoding="utf-8"),
        parser="pdftotext",
    )
    sidecar = document_schema.write_metadata_sidecar(md, metadata)

    assert sidecar.name == "sample.md.metadata.json"
    loaded = document_schema.load_metadata_sidecar(md)
    assert loaded["schemaVersion"] == 1
    assert loaded["parser"] == "pdftotext"
    assert loaded["pages"][0]["blocks"][0]["lineStart"] == 1


def test_validate_unknown_layout_type_is_normalized(tmp_path):
    data = {
        "schemaVersion": 1,
        "sourcePath": "x.pdf",
        "pages": [{"page": 1, "blocks": [{"layoutType": "vendor-specific"}]}],
    }

    validated = document_schema.validate_document_metadata(json.loads(json.dumps(data)))

    assert validated["pages"][0]["blocks"][0]["layoutType"] == "unknown"


def test_load_metadata_sidecar_ignores_content_hash_mismatch(tmp_path):
    md = tmp_path / "sample.md"
    md.write_text("old", encoding="utf-8")
    metadata = document_schema.build_document_metadata(
        source_path=tmp_path / "sample.pdf",
        markdown_path=md,
        markdown_text="old",
        parser="pdftotext",
    )
    document_schema.write_metadata_sidecar(md, metadata)
    md.write_text("new", encoding="utf-8")

    assert document_schema.load_metadata_sidecar(md) is None


def test_metadata_for_line_range_falls_back_when_no_overlap():
    metadata = {
        "schemaVersion": 1,
        "sourcePath": "sample.pdf",
        "sourceTitle": "Sample",
        "parser": "structured",
        "pages": [
            {
                "page": 1,
                "blocks": [
                    {
                        "blockId": "b1",
                        "layoutType": "table",
                        "lineStart": 10,
                        "lineEnd": 20,
                        "page": 1,
                    }
                ],
            }
        ],
    }

    result = document_schema.metadata_for_line_range(metadata, 1, 2)

    assert result == {"parser": "structured", "source_title": "Sample"}


@pytest.mark.parametrize("warnings", [None, "not-a-list", ["ok", 1]])
def test_load_metadata_sidecar_rejects_invalid_warnings(tmp_path, warnings):
    md = tmp_path / "sample.md"
    md.write_text("body", encoding="utf-8")
    sidecar = document_schema.default_sidecar_path(md)
    sidecar.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "sourcePath": "sample.pdf",
                "warnings": warnings,
                "pages": [],
            }
        ),
        encoding="utf-8",
    )

    assert document_schema.load_metadata_sidecar(md) is None


@pytest.mark.parametrize(
    "pages",
    [
        ["not-an-object"],
        [{"blocks": ["not-an-object"]}],
        [{"blocks": [{"lineStart": "1", "lineEnd": 2}]}],
    ],
)
def test_load_metadata_sidecar_rejects_invalid_nested_objects(tmp_path, pages):
    md = tmp_path / "sample.md"
    md.write_text("body", encoding="utf-8")
    document_schema.default_sidecar_path(md).write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "sourcePath": "sample.pdf",
                "warnings": [],
                "pages": pages,
            }
        ),
        encoding="utf-8",
    )

    assert document_schema.load_metadata_sidecar(md) is None


def test_load_metadata_sidecar_treats_unrepresentable_sidecar_name_as_absent(tmp_path, monkeypatch):
    """`.metadata.json`を付けると名前長上限を超える資料でも、本文読取を失敗にしない。"""
    import errno
    from pathlib import Path

    md = tmp_path / "sample.md"
    md.write_text("# 規程\n本文", encoding="utf-8")
    original_exists = Path.exists

    def exists_with_name_limit(self, *args, **kwargs):
        if self.name.endswith(".metadata.json"):
            raise OSError(errno.ENAMETOOLONG, "File name too long", str(self))
        return original_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists_with_name_limit)

    assert document_schema.load_metadata_sidecar(md) is None
