"""JSON-serializable document metadata schema for offline-ai."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
LAYOUT_TYPES = {
    "title",
    "text",
    "table",
    "figure",
    "caption",
    "header",
    "footer",
    "reference",
    "equation",
    "unknown",
}


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sidecar_candidates(markdown_path: Path) -> list[Path]:
    """Return supported metadata sidecar names for a Markdown file."""
    return [
        markdown_path.with_name(markdown_path.name + ".metadata.json"),
        markdown_path.with_suffix(".metadata.json"),
    ]


def default_sidecar_path(markdown_path: Path) -> Path:
    return sidecar_candidates(markdown_path)[0]


def _line_blocks(source_path: Path, text: str, parser: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    blocks: list[dict[str, Any]] = []
    current_heading = ""
    start_line = 1
    buffer: list[str] = []

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line
        block_text = "\n".join(buffer).strip()
        if not block_text:
            buffer = []
            start_line = end_line + 1
            return
        block_index = len(blocks) + 1
        blocks.append(
            {
                "blockId": f"{source_path.name}#b{block_index:04d}",
                "layoutType": "title" if block_text.startswith("#") and len(buffer) == 1 else "text",
                "heading": current_heading,
                "text": block_text,
                "markdown": block_text,
                "page": None,
                "bbox": None,
                "confidence": None,
                "lineStart": start_line,
                "lineEnd": end_line,
                "parser": parser,
            }
        )
        buffer = []
        start_line = end_line + 1

    for line_no, line in enumerate(lines, 1):
        if line.startswith("#"):
            if buffer:
                flush(line_no - 1)
            current_heading = line.lstrip("#").strip()
            start_line = line_no
        buffer.append(line)
    if buffer:
        flush(len(lines))
    if not blocks and text.strip():
        blocks.append(
            {
                "blockId": f"{source_path.name}#b0001",
                "layoutType": "text",
                "heading": "",
                "text": text.strip(),
                "markdown": text.strip(),
                "page": None,
                "bbox": None,
                "confidence": None,
                "lineStart": 1,
                "lineEnd": max(len(lines), 1),
                "parser": parser,
            }
        )
    return blocks


def build_document_metadata(
    *,
    source_path: Path,
    markdown_path: Path,
    markdown_text: str,
    parser: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    blocks = _line_blocks(source_path, markdown_text, parser)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourcePath": str(source_path),
        "markdownPath": str(markdown_path),
        "sourceTitle": source_path.stem,
        "parser": parser,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "contentSha256": content_sha256(markdown_text),
        "warnings": warnings or [],
        "pages": [{"page": None, "blocks": blocks}],
    }


def validate_document_metadata(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("metadata must be an object")
    if data.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schemaVersion: {data.get('schemaVersion')}")
    if not data.get("sourcePath"):
        raise ValueError("sourcePath is required")
    warnings = data.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(warning, str) for warning in warnings):
        raise ValueError("warnings must be a list of strings")
    pages = data.get("pages")
    if not isinstance(pages, list):
        raise ValueError("pages must be a list")
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("each page must be an object")
        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            raise ValueError("page.blocks must be a list")
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("each block must be an object")
            for line_key in ("lineStart", "lineEnd"):
                line_value = block.get(line_key)
                if line_value is not None and (
                    isinstance(line_value, bool) or not isinstance(line_value, int)
                ):
                    raise ValueError(f"block.{line_key} must be an integer or null")
            layout_type = block.get("layoutType", "unknown")
            if layout_type not in LAYOUT_TYPES:
                block["layoutType"] = "unknown"
    return data


def write_metadata_sidecar(markdown_path: Path, metadata: dict[str, Any]) -> Path:
    validated = validate_document_metadata(metadata)
    sidecar = default_sidecar_path(markdown_path)
    sidecar.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    return sidecar


def load_metadata_sidecar(markdown_path: Path) -> dict[str, Any] | None:
    for candidate in sidecar_candidates(markdown_path):
        try:
            if not candidate.exists():
                continue
        except OSError:
            # A sidecar name that the filesystem cannot represent (for example
            # ENAMETOOLONG once ".metadata.json" is appended) cannot exist, so
            # treat it as "no sidecar" instead of failing the source document.
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            validated = validate_document_metadata(data)
            expected_hash = validated.get("contentSha256")
            if expected_hash:
                current_hash = content_sha256(markdown_path.read_text(encoding="utf-8", errors="replace"))
                if current_hash != expected_hash:
                    return None
            return validated
        except (OSError, json.JSONDecodeError, ValueError):
            return None
    return None


def metadata_for_line_range(metadata: dict[str, Any] | None, start_line: int | None, end_line: int | None) -> dict[str, Any]:
    if not metadata:
        return {}
    parser_warnings = [
        str(warning).strip()[:500]
        for warning in metadata.get("warnings", [])
        if str(warning).strip()
    ][:10]
    best_block: dict[str, Any] | None = None
    best_overlap = 0
    for page in metadata.get("pages", []):
        for block in page.get("blocks", []):
            block_start = block.get("lineStart")
            block_end = block.get("lineEnd")
            overlap = 0
            if start_line is not None and end_line is not None and block_start is not None and block_end is not None:
                overlap = max(0, min(end_line, int(block_end)) - max(start_line, int(block_start)) + 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_block = {**block, "page": block.get("page", page.get("page"))}
    if not best_block:
        result = {
            "parser": metadata.get("parser"),
            "source_title": metadata.get("sourceTitle"),
        }
        if parser_warnings:
            result["parser_warnings"] = parser_warnings
        return result
    result = {
        "parser": best_block.get("parser", metadata.get("parser")),
        "page": best_block.get("page"),
        "bbox": best_block.get("bbox"),
        "layout_type": best_block.get("layoutType"),
        "source_title": metadata.get("sourceTitle"),
        "confidence": best_block.get("confidence"),
        "block_id": best_block.get("blockId"),
    }
    if parser_warnings:
        result["parser_warnings"] = parser_warnings
    return result
