"""Read-only local corpus tools and immutable, turn-scoped evidence records."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from source_structure import (
    classify_range,
    document_numbers,
    document_title,
    group_split_documents,
    match_documents,
    normalize_match_text,
    parse_heading_tree,
    title_variants,
)
from source_view import EvidenceViewError, _normalize_relative_path, read_source_snapshot

TOOL_BYTES = 32 * 1024
SOURCE_ID = re.compile(r"S[1-9][0-9]{0,8}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def byte_size(value) -> int:
    return len(encode(value).encode("utf-8"))


def _function(name, description, required, **properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_ARGUMENTS = {
    "search_sources": {"query", "limit", "source_id"},
    "read_source": {
        "source_id",
        "path",
        "section_id",
        "start_line",
        "end_line",
        "cursor",
    },
    "list_sources": {"query", "limit", "cursor"},
}

_TEXT = {"type": "string"}
_INTEGER = {"type": "integer"}
TOOLS = [
    _function(
        "search_sources",
        "資料を検索する。表題・文書番号・条項・短い主題を使う。対象資料を限定するときだけ取得済みの source_id を指定する。",
        ["query"],
        query=_TEXT,
        limit=_INTEGER,
        source_id=_TEXT,
    ),
    _function(
        "read_source",
        "候補の原文を読む。通常は source_id だけを渡す。続きは返された cursor だけを渡す。範囲を変える場合は節IDか行範囲の一方を使う。",
        [],
        source_id=_TEXT,
        path=_TEXT,
        section_id=_TEXT,
        start_line=_INTEGER,
        end_line=_INTEGER,
        cursor=_TEXT,
    ),
    _function(
        "list_sources",
        "資料の表題・文書番号・節IDを一覧する。最初は cursor を省略し、続きは直前の一覧が返した cursor を使う。",
        [],
        query=_TEXT,
        limit=_INTEGER,
        cursor=_TEXT,
    ),
]


_ARTICLE = re.compile(r"第\s*([0-9一二三四五六七八九十百千〇零]{1,12})\s*(条|項|号|款|節)")
_DIGITS = dict(zip("〇一二三四五六七八九", range(10)))
_DIGITS["零"] = 0
_UNITS = {"十": 10, "百": 100, "千": 1000}


def _number(value):
    if value.isdecimal():
        return int(value)
    if all(c in _DIGITS for c in value):
        return int("".join(str(_DIGITS[c]) for c in value))
    total, digit = 0, 0
    for char in value:
        if char in _UNITS:
            total += (digit or 1) * _UNITS[char]
            digit = 0
        else:
            digit = _DIGITS.get(char, 0)
    return total + digit


def _focus_text(value):
    normalized = _ARTICLE.sub(lambda m: f"第{_number(m[1])}{m[2]}", normalize_match_text(value))
    return re.sub(
        r"(第[0-9]+条の)([0-9一二三四五六七八九十百千〇零]{1,12})",
        lambda m: m[1] + str(_number(m[2])),
        normalized,
    )


_ARTICLE_LABEL = re.compile(r"第[0-9]+条(?:の[0-9]+)?")
_EXCEPTION_QUERY = re.compile(r"除外|除く|除か|対象外|例外|適用外")
_EXCEPTION_CLAUSE = re.compile(
    r"場合を除|を除[くき]|を除外|にかかわらず|に関わらず|案件としない|対象としない|ただし"
)
_LAW_REFERENCE = re.compile(r"(法|令)(第[0-9]+条(?:の[0-9]+)?)(第[0-9]+項)?")
_LAW_ALIAS = re.compile(r"[（(][^（）()]{0,120}以下[「『](法|令)[」』]という[。．]?[）)]")
_DISPOSAL_TOPIC = re.compile(r"売[り]?払|売却|廃棄|不用")


def _topic_text(value):
    return re.sub(r"売り?払い?", "売払", _focus_text(value))


def _article_label(normalized):
    key = normalized.lstrip("#*")
    article = _ARTICLE_LABEL.match(key)
    if article and not re.match(r"第[0-9]+項", key[article.end() :]):
        return article[0]
    return None


_PRECEDING_ITEM_REFERENCE = re.compile(
    r"前項(第[0-9]+号(?:(?:又は|若しくは|及び|並びに)第[0-9]+号)+)"
)
_NUMBERED_ITEM = re.compile(r"^\s*[-*]\s+([一二三四五六七八九十百0-9]{1,5})\s+(.+)$")


def _preceding_items(lines, first, hit):
    """Find the cited list items in the immediately preceding paragraph."""
    clause = _focus_text(lines[hit - 1])
    relation = _PRECEDING_ITEM_REFERENCE.search(clause)
    if not relation:
        return None
    numbers = {n for n in re.findall(r"第([0-9]+)号", relation[1])}
    items = {}
    for index in range(hit - 1, first - 1, -1):
        line = unicodedata.normalize("NFKC", lines[index - 1])
        if re.match(r"^\s*(?:#{1,6}\s|[0-9]+[.、\s]|第.+項)", line):
            break
        item = _NUMBERED_ITEM.match(line)
        if item and (number := str(_number(item[1]))) in numbers:
            items[number] = index
    if items.keys() == numbers and max(items.values()) - min(items.values()) <= 8:
        return min(items.values()), max(items.values())
    return None


def _paragraph_lines(lines, first, last, paragraph):
    if not paragraph:
        return None
    number = paragraph[1:-1]
    for index in range(first, last + 1):
        line = unicodedata.normalize("NFKC", lines[index - 1])
        if re.match(rf"^\s*{number}[.、\s]", line) or _focus_text(line).startswith(paragraph):
            return index
    return None


def _focus_ranges(
    lines, text, query, check, *, exceptions=False, article_only=False, by_heading=False
):
    normalized = _focus_text(query)
    normalize = _topic_text if by_heading else _focus_text
    phrases = [normalize(piece) for piece in re.split(r"[\s、，,／/]+", query)]
    phrases = [p for p in phrases if len(p) >= 2 and not _ARTICLE.fullmatch(p)]
    if by_heading and _DISPOSAL_TOPIC.search(query):
        phrases.extend(p for p in ("売払", "不用の決定", "廃棄の基準") if p not in phrases)
    if by_heading and "審査" in normalized and "チェックリスト" in normalized:
        phrases.extend(p for p in ("審査書類", "書類の提出期限", "審査時期") if p not in phrases)
    if by_heading and "旅行命令" in normalized and "請求" in normalized:
        phrases.extend(
            p for p in ("変更の申請", "旅費の請求手続", "請求書及び必要な資料") if p not in phrases
        )
    grams = {p[i : i + 3] for p in phrases for i in range(len(p) - 2)}
    articles = set(_ARTICLE_LABEL.findall(normalized))
    paragraphs = [m[1] for m in _ARTICLE.finditer(normalized) if m[2] == "項"]
    paragraph_pattern = (
        re.compile(r"^\s*[* ]*(?:" + "|".join(rf"(?:{p}[.、\s]|第{p}項)" for p in paragraphs) + ")")
        if paragraphs
        else None
    )
    headings = iter(parse_heading_tree(text))
    next_heading = next(headings, None)
    active_headings = []
    sections = {}
    parent_article = False
    for index, line in enumerate(lines, 1):
        if index % 256 == 0:
            check()
        key = normalize(line)
        if index == 1 and line.startswith("#") and not articles:
            continue
        article = _article_label(key) if articles else None
        if article:
            parent_article = article in articles
        while next_heading and next_heading.heading_line <= index:
            active_headings.append(next_heading)
            next_heading = next(headings, None)
        while active_headings and active_headings[-1].section_end_line < index:
            active_headings.pop()
        if article_only and not parent_article:
            continue
        overlap = sum(g in key for g in grams)
        exact = sum(len(p) for p in phrases if p in key)
        node = active_headings[-1] if active_headings else None
        heading = normalize(node.heading_text) if node and by_heading else ""
        if by_heading and node and len(active_headings) > 1:
            parent = active_headings[-2]
            if node.heading_line - parent.heading_line <= 3:
                if re.match(r"別紙[0-9]", normalize(parent.heading_text)):
                    heading += normalize(parent.heading_text)
        heading_hits = sum(min(len(p), 8) + 4 for p in phrases if p in heading)
        if by_heading and _DISPOSAL_TOPIC.search(query):
            heading_hits += 14 * bool(_DISPOSAL_TOPIC.search(heading))
            if "廃棄" in query:
                heading_hits += 32 * bool(re.search(r"不用の決定等|廃棄の基準", heading))
            elif re.search(r"売り?払", query):
                heading_hits += 24 * bool(
                    re.search(r"^売払|売払.*措置の請求", heading.strip("（）()"))
                )
        # A standalone operative clause outranks a parenthetical mention or a checklist.
        clause = _EXCEPTION_CLAUSE.search(key) if exceptions else None
        if exceptions and not clause:
            continue
        operative = bool(
            clause
            and not line.lstrip().startswith(("|", "-", ">"))
            and clause[0]
            in {"にかかわらず", "に関わらず", "案件としない", "対象としない", "場合を除"}
        )
        article_score = 8 if parent_article else 0
        if article and article in articles:
            article_score += 100
        if (
            parent_article
            and paragraph_pattern
            and paragraph_pattern.match(unicodedata.normalize("NFKC", line))
        ):
            article_score += 120
        score = article_score + 4 * exact + overlap + 3 * heading_hits + (75 if operative else 0)
        if score < 3:
            continue
        start = node.heading_line if node else max(1, index - 2)
        if by_heading and node and len(active_headings) > 1:
            parent = active_headings[-2]
            if node.heading_line - parent.heading_line <= 3 and re.match(
                r"別紙[0-9]", normalize(parent.heading_text)
            ):
                start = parent.heading_line
        end = node.section_end_line if node else min(len(lines), index + 12)
        if by_heading and end - start > 70 and not article:
            continue
        if end - start > 70 or (article_score >= 100 and not article_only):
            start = max(start, index - 2)
            end = min(end, index + 28)
        record = (score, index, start, end)
        section = (start, end)
        if section not in sections or score > sections[section][0]:
            sections[section] = record
    return sorted(sections.values(), key=lambda row: (-row[0], row[1]))[:3]


@dataclass(frozen=True)
class Span:
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    text: str

    def matches(self, lines):
        if not 1 <= self.start_line <= self.end_line <= len(lines):
            return False
        selected = lines[self.start_line - 1 : self.end_line]
        if not 0 <= self.start_char <= len(
            selected[0]
        ) or not 0 <= self.end_char <= len(selected[-1]):
            return False
        if len(selected) == 1:
            return (
                self.start_char <= self.end_char
                and selected[0][self.start_char : self.end_char] == self.text
            )
        return (
            "\n".join(
                [
                    selected[0][self.start_char :],
                    *selected[1:-1],
                    selected[-1][: self.end_char],
                ]
            )
            == self.text
        )


@dataclass(frozen=True)
class Source:
    id: str
    path: str
    sha256: str
    title: str
    state: str
    turn: int
    generation: str
    spans: tuple[Span, ...]

    def public(self):
        return {**asdict(self), "citation": f"[{self.id}]"}

    @classmethod
    def parse(cls, value):
        required = {
            "id",
            "path",
            "sha256",
            "title",
            "state",
            "turn",
            "generation",
            "spans",
        }
        if not isinstance(value, dict) or not required <= value.keys():
            raise ValueError("invalid_source")
        if not all(
            isinstance(value[k], str)
            for k in ("id", "path", "sha256", "title", "state", "generation")
        ):
            raise ValueError("invalid_source")
        from source_view import _normalize_relative_path

        if (
            not SOURCE_ID.fullmatch(value["id"])
            or not DIGEST.fullmatch(value["sha256"])
            or not DIGEST.fullmatch(value["generation"])
            or len(value["path"]) > 4096
            or _normalize_relative_path(value["path"]) != value["path"]
            or len(value["title"]) > 240
            or value["state"] not in {"search_excerpt", "read"}
            or type(value["turn"]) is not int
            or not 1 <= value["turn"] <= 1_000_000
            or not isinstance(value["spans"], (list, tuple))
            or not 1 <= len(value["spans"]) <= 12
        ):
            raise ValueError("invalid_source")
        spans = []
        for item in value["spans"]:
            if not isinstance(item, dict) or set(item) != set(
                Span.__dataclass_fields__
            ):
                raise ValueError("invalid_source_range")
            span = Span(**item)
            if (
                not all(
                    type(v) is int and 0 <= v <= 16 * 1024 * 1024
                    for v in (
                        span.start_line,
                        span.end_line,
                        span.start_char,
                        span.end_char,
                    )
                )
                or not 1 <= span.start_line <= span.end_line
                or (
                    span.start_line == span.end_line and span.start_char > span.end_char
                )
                or not isinstance(span.text, str)
                or len(span.text) > 6000
                or (spans and spans[-1].end_line >= span.start_line)
            ):
                raise ValueError("invalid_source_range")
            spans.append(span)
        if sum(len(s.text) for s in spans) > 6000:
            raise ValueError("invalid_source_range")
        return cls(**{k: value[k] for k in required - {"spans"}}, spans=tuple(spans))


def _page(lines, start, end, column, limit):
    """Select real lines and carry the next line/character across bounded reads."""
    if not 1 <= start <= end <= len(lines) or not 0 <= column <= len(lines[start - 1]):
        raise ToolArgumentError("invalid_line_range")
    text = []
    size = 0
    for number in range(start, end + 1):
        offset = column if number == start else 0
        line = lines[number - 1]
        separator = bool(text)
        available = limit - size - separator
        if len(line) - offset > available:
            if text:
                return Span(start, number - 1, column, len(lines[number - 2]), "\n".join(text)), (
                    number,
                    0,
                )
            return Span(
                start,
                number,
                column,
                offset + available,
                line[offset : offset + available],
            ), (number, offset + available)
        text.append(line[offset:])
        size += len(text[-1]) + separator
    return Span(start, end, column, len(lines[end - 1]), "\n".join(text)), None


def verify_source(root, source, *, registry=None, session_id=None, cache=None):
    """Check only the exact, possibly disjoint spans supplied to the model."""
    try:
        key = (source.path, source.sha256)
        if cache is not None and key in cache:
            lines = cache[key]
        else:
            _, lines, _ = read_source_snapshot(root, source.path, source.sha256)
            if cache is not None:
                cache.clear()
                cache[key] = lines
    except EvidenceViewError as exc:
        state = (
            "missing"
            if exc.code == "source_missing"
            else "changed"
            if exc.code == "source_changed"
            else "unavailable"
        )
        return {**source.public(), "verification": state}
    if not all(span.matches(lines) for span in source.spans):
        return {**source.public(), "verification": "excerpt_mismatch"}
    result = {**source.public(), "verification": "matched"}
    if registry is not None:
        result["evidence"] = [
            {
                "span": index,
                "id": registry.register(
                    session_id=session_id,
                    relative_path=source.path,
                    start_line=span.start_line,
                    end_line=span.end_line,
                    source_sha256=source.sha256,
                ),
            }
            for index, span in enumerate(source.spans)
        ]
    return result


class ToolArgumentError(ValueError):
    """A fixed diagnostic code for rejected tool arguments."""


class SourceTools:
    def __init__(self, root, generation, turn, prior_sources, *, retrieve, list_paths, key=None):
        self.root = Path(root)
        self.generation = generation
        self.turn = turn
        self.sources = dict(prior_sources)
        self.new_ids = []
        self.retrieve = retrieve
        self.list_paths = list_paths
        self.key = key or secrets.token_bytes(32)
        self.cursors = {}
        self.read_ranges = {}
        self.references = {}
        self.trace = []
        self._next = max((int(id_[1:]) for id_ in self.sources), default=0) + 1

    def _cursor(self, value):
        token = secrets.token_urlsafe(18)
        self.cursors[token] = value
        return token

    def _section_id(self, path, digest, node):
        binding = encode([self.generation, path, digest, node.heading_line])
        return hmac.digest(self.key, binding.encode("utf-8"), "sha256").hex()[:32]

    def _record(self, path, sha, title, state, spans):
        source = Source(
            f"S{self._next}",
            path,
            sha,
            title[:240],
            state,
            self.turn,
            self.generation,
            tuple(spans),
        )
        self._next += 1
        self.sources[source.id] = source
        self.new_ids.append(source.id)
        return source

    @staticmethod
    def _title(path, text):
        return document_title(text[:6000], path=path)[:240] or Path(path).name[:240]

    def _sections(self, path, digest, text, *, near=None):
        nodes = parse_heading_tree(text)
        if near is not None:
            nodes = [n for n in nodes if n.heading_line <= near <= n.section_end_line]
        return [
            {
                "section_id": self._section_id(path, digest, n),
                "title": n.heading_text[:120],
                "level": n.level,
                "start_line": n.heading_line,
                "end_line": n.section_end_line,
            }
            for n in nodes[:20]
        ], len(nodes) > 20

    @staticmethod
    def _query(args, *, required=False):
        query = args.get("query", "")
        if not isinstance(query, str) or len(query) > 1000 or (required and not query.strip()):
            raise ToolArgumentError("invalid_query")
        return query.strip()

    @staticmethod
    def _limit(args, default, maximum):
        limit = args.get("limit", default)
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise ToolArgumentError("invalid_limit")
        return limit

    def _document_groups(self):
        rows = [
            {"path": path, "title": Path(path).stem, "head_text": Path(path).stem}
            for path in self.list_paths()
        ]
        return {row["path"]: row for row in group_split_documents(rows)}

    def _search(self, args, check, deadline):
        query, limit = self._query(args, required=True), self._limit(args, 8, 12)
        anchor_id = args.get("source_id")
        anchor = self.sources.get(anchor_id) if isinstance(anchor_id, str) else None
        if "source_id" in args and (anchor is None or anchor.generation != self.generation):
            raise ToolArgumentError("unknown_source")
        search_result = (
            {"matches": [], "route": "keyword"} if anchor else self.retrieve(query, check, deadline)
        )
        check()
        if search_result.get("error"):
            return {"error": search_result["error"], "retrieval": search_result["route"]}
        matches = search_result["matches"]
        groups = self._document_groups()
        group_by_part = {
            part["path"]: group for group in groups.values() for part in group.get("paths", [group])
        }
        titles = match_documents(query, list(groups.values()))["candidates"] if not anchor else []
        paths = []
        explicit_article = bool(_ARTICLE_LABEL.search(_focus_text(query)))
        exceptions = not (anchor or explicit_article) and bool(_EXCEPTION_QUERY.search(query))
        topical = not (explicit_article or exceptions) and len(query.split()) >= 3
        family = {}
        if topical:
            subjects = {_topic_text(word) for word in query.split() if len(_topic_text(word)) >= 4}
            for group in groups.values():
                labels = [
                    _topic_text(name)
                    for name in title_variants(group.get("titles", [group["title"]]))
                ]
                strength = max(
                    (
                        len(term) if term in label else 4 if label[:4] == term[:4] else 0
                        for label in labels
                        if len(label) >= 4
                        for term in subjects
                    ),
                    default=0,
                )
                if _DISPOSAL_TOPIC.search(query) and any(
                    label.endswith("物品管理法") for label in labels
                ):
                    strength = max(strength, 8)
                if strength:
                    family[group["path"]] = strength
        subject_groups = set()
        if exceptions:
            terms = {term for word in query.split() if len(term := _focus_text(word)) >= 6}
            # A distinctive query term can match part of a longer document title.
            # Keep these candidates bounded and let the operative clause rank them.
            subject_groups = {
                group["path"] for group in groups.values()
                if any(
                    term in _focus_text(title)
                    for title in title_variants(group.get("titles", [group["title"]]))
                    for term in terms
                )
            }
        matched_paths = {match["path"] for match in matches}
        if anchor:
            group = group_by_part.get(anchor.path)
            paths = [p["path"] for p in group.get("paths", [group])] if group else [anchor.path]
        else:
            for hit in titles:
                if hit["strength"] != 2:
                    continue
                group = groups[hit["path"]]
                paths.extend(p["path"] for p in group.get("paths", [group]))
            for path in sorted(subject_groups)[:4]:
                group = groups[path]
                paths.extend(p["path"] for p in group.get("paths", [group])[:4])
            for path in sorted(family, key=family.get, reverse=True)[:5]:
                group = groups[path]
                paths.extend(p["path"] for p in group.get("paths", [group])[:4])
            for match in matches:
                group = group_by_part.get(match["path"])
                if group:
                    paths.extend(
                        part["path"]
                        for part in group.get("paths", [group])
                        if explicit_article
                        or exceptions
                        or topical
                        or part["path"] not in matched_paths
                    )
        focus = []
        for path in list(dict.fromkeys(paths))[:24]:
            if not (anchor or explicit_article or exceptions or topical) and path in matched_paths:
                continue
            check()
            try:
                expected = anchor.sha256 if anchor and path == anchor.path else None
                sha, lines, text = read_source_snapshot(self.root, path, expected)
            except EvidenceViewError:
                continue
            for score, hit, first, last in _focus_ranges(
                lines, text, query, check, exceptions=exceptions, by_heading=topical
            ):
                if topical:
                    group = group_by_part.get(path, {})
                    score = score // 2 + 6 * family.get(group.get("path"), 0)
                    if any(_article_label(_focus_text(line)) for line in lines[first - 1 : last]):
                        score += 36
                        if (
                            _DISPOSAL_TOPIC.search(query)
                            and Path(path).stem.split("-")[0].endswith("法")
                            and _DISPOSAL_TOPIC.search(lines[first - 1])
                        ):
                            score += 45
                    if _DISPOSAL_TOPIC.search(query) and _DISPOSAL_TOPIC.search(lines[first - 1]):
                        score += 30
                    if "審査" in query and any(
                        term in lines[first - 1] for term in ("審査案件", "審査時期", "審査書類")
                    ):
                        score += 26
                if exceptions and score < 75:
                    continue
                if anchor or score >= 8:
                    preceding = _preceding_items(lines, first, hit) if exceptions else None
                    if preceding and group_by_part[path]["path"] in subject_groups:
                        score += 100
                    focus.append(
                        (
                            score,
                            {
                                "path": path,
                                "source_sha256": sha,
                                "start_line": hit if exceptions else max(first, hit - 1),
                                "end_line": min(last, hit + 8),
                                "suggested_read": (first, last),
                                "preceding_items": preceding,
                            },
                        )
                    )
        ranked = [match for _, match in sorted(focus, key=lambda item: -item[0])]
        if exceptions:
            for index, match in enumerate(ranked[:3]):
                reference = self._exception_reference(match, groups, check)
                if reference:
                    # Keep the reference and the clause that gives it meaning.
                    ranked.pop(index)
                    ranked[:0] = [reference, match]
                    break
        if anchor or ranked:
            focus_limit = min(5, max(3, limit // 2)) if topical else 3
            added = ranked[: (limit if anchor else focus_limit)]
            matches = [
                *added,
                *(
                    m
                    for m in matches
                    if not any(
                        a["path"] == m["path"]
                        and not m.get("group_id")
                        and isinstance(m.get("start_line"), int)
                        and isinstance(m.get("end_line"), int)
                        and a["start_line"] <= m["start_line"] <= m["end_line"] <= a["end_line"]
                        for a in added
                    )
                ),
            ]

        grouped_ranges = {}
        for match in matches:
            if match.get("group_id"):
                key = (match["path"], match.get("source_sha256"), match["group_id"])
                grouped_ranges.setdefault(key, []).append(match)
        candidates = []
        used = 0
        matched_paths = set()
        seen_groups = set()
        read_ranges = {}
        references = {}
        scoped = {}
        for match in matches:
            check()
            path = match["path"]
            start, end = match.get("start_line"), match.get("end_line")
            if type(start) is not int or type(end) is not int:
                continue
            key = (path, match.get("source_sha256"), match.get("group_id"))
            if key in seen_groups:
                continue
            group = grouped_ranges.get(key, [match])
            if match.get("group_id"):
                seen_groups.add(key)
            try:
                sha, lines, text = read_source_snapshot(
                    self.root,
                    path,
                    match.get("source_sha256") or match.get("file_sha256"),
                )
                if topical and "suggested_read" not in match:
                    covering = [
                        node
                        for node in parse_heading_tree(text)
                        if node.heading_line <= start <= node.section_end_line
                        and node.section_end_line - node.heading_line <= 60
                    ]
                    if covering:
                        node = covering[-1]
                        match = {
                            **match,
                            "suggested_read": (node.heading_line, node.section_end_line),
                        }
                spans = []
                ranges = sorted(
                    (row["start_line"], row["end_line"])
                    for row in group
                    if type(row.get("start_line")) is int and type(row.get("end_line")) is int
                )[:4]
                preceding = None
                if match.get("preceding_items"):
                    first, last = match["preceding_items"]
                    preceding, continuation = _page(lines, first, last, 0, 500)
                    if continuation:
                        preceding = None
                    else:
                        spans.append(preceding)
                for first, last in ranges:
                    if not spans or spans[-1].end_line < first:
                        span, _ = _page(lines, first, last, 0, 300 // len(ranges))
                        spans.append(span)
            except (EvidenceViewError, ValueError):
                continue
            if not spans:
                continue
            title = str(match.get("source_title") or self._title(path, text))[:240]
            sections, _ = self._sections(path, sha, text, near=start)
            ancillary = classify_range(text, start, end, lines=lines)
            candidates.append((path, sha, title, tuple(spans), sections, ancillary, None))
            if "suggested_read" in match:
                read_ranges[(path, spans[0])] = match["suggested_read"]
            if preceding:
                scoped[(path, spans[0])] = start
            if "referenced_from" in match:
                references[(path, spans[0])] = match["referenced_from"]
            matched_paths.add(path)
        # Add bounded navigation to siblings of the returned parts. The ranked
        # retrieval results themselves keep their order.
        sibling_parts = []
        if len(candidates) < limit and matched_paths:
            for group in groups.values():
                parts = group.get("paths", [])
                anchors = [p["path"] for p in parts if p["path"] in matched_paths]
                if anchors:
                    sibling_parts.extend(
                        (p["path"], anchors[0]) for p in parts if p["path"] not in matched_paths
                    )
        for path, anchor in sibling_parts[: min(2, max(0, limit - len(candidates)))]:
            check()
            try:
                sha, lines, text = read_source_snapshot(self.root, path)
                span, _ = _page(lines, 1, len(lines), 0, 200)
            except (EvidenceViewError, ValueError):
                continue
            sections, _ = self._sections(path, sha, text, near=span.start_line)
            candidates.append((path, sha, self._title(path, text), (span,), sections, "", anchor))
        results = []
        added_ranges = set()
        for path, sha, title, spans, sections, ancillary, anchor in sorted(
            candidates, key=lambda row: (bool(row[5]), bool(row[4]))
        ):
            if len(results) >= limit:
                break
            suggested = read_ranges.get((path, spans[0]))
            if topical and suggested:
                range_key = (path, suggested)
                if range_key in added_ranges:
                    continue
                added_ranges.add(range_key)
            source = Source(
                f"S{self._next}",
                path,
                sha,
                title,
                "search_excerpt",
                self.turn,
                self.generation,
                spans,
            )
            entry = {**source.public(), "sections": sections, "range_role": ancillary}
            if anchor:
                entry["same_document_as"] = anchor
            if suggested:
                entry["suggested_read"] = {"start_line": suggested[0], "end_line": suggested[1]}
            if scope_line := scoped.get((path, spans[0])):
                entry["scope_from_preceding_items"] = True
                entry["exception_scope_line"] = scope_line
            if reference := references.get((path, spans[0])):
                entry["referenced_from"] = reference
                entry["exception_target"] = reference["provision"]
            entry_bytes = byte_size(entry)
            if used + entry_bytes + 1024 > TOOL_BYTES:
                break
            used += entry_bytes + 1
            self._record(path, sha, title, source.state, source.spans)
            if suggested:
                self.read_ranges[source.id] = suggested
            if reference:
                self.references[source.id] = reference
            results.append(entry)
        self.trace.append(
            {
                "tool": "search_sources",
                "query": query,
                "results": len(results),
                "retrieval": search_result["route"],
                "embedding_calls": search_result.get("embedding_calls", 0),
                "embedding_seconds": search_result.get("embedding_seconds", 0),
            }
        )
        return {
            "untrusted_source_data": results,
            "retrieval": search_result["route"],
            "truncated": len(results) < len(candidates),
            "next_step": "原文は候補の source_id で読めます。referenced_from は参照元の資料と行です。文書内の別条項は source_id と条項で検索できます。",
        }

    def _exception_reference(self, match, groups, check):
        check()
        path = match["path"]
        try:
            _, lines, _ = read_source_snapshot(self.root, path, match["source_sha256"])
        except EvidenceViewError:
            return None
        hit = match["start_line"]
        clause = _focus_text(lines[hit - 1])
        references = list(_LAW_REFERENCE.finditer(clause))
        if not references:
            return None
        reference = references[-1]
        if not re.match(
            r"(?:の規定)?(?:による|により)[^。]{0,40}場合を除", clause[reference.end() :]
        ):
            return None
        alias, article, paragraph = reference.groups()
        names = [
            (normalize_match_text(title), row)
            for row in groups.values()
            for title in title_variants(row.get("titles", [row["title"]]))
        ]
        # Resolve the alias only when the document explicitly defines it.
        for line in lines[:120]:
            for definition in _LAW_ALIAS.finditer(line):
                if definition[1] != alias:
                    continue
                prefix = normalize_match_text(line[: definition.start()])
                options = [
                    (name, row) for name, row in names if len(name) >= 4 and prefix.endswith(name)
                ]
                if not options:
                    continue
                longest = max(len(name) for name, _ in options)
                candidates = {row["path"]: row for name, row in options if len(name) == longest}
                if len(candidates) != 1:
                    return None
                group = next(iter(candidates.values()))
                for part in group.get("paths", [group])[:8]:
                    check()
                    target = part["path"]
                    if target == path:
                        continue
                    try:
                        digest, body, text = read_source_snapshot(self.root, target)
                    except EvidenceViewError:
                        continue
                    for _, line_no, first, last in _focus_ranges(
                        body, text, article, check, article_only=True
                    ):
                        if _article_label(_focus_text(body[line_no - 1])) != article:
                            continue
                        target_line = _paragraph_lines(body, line_no, last, paragraph)
                        return {
                            "path": target,
                            "source_sha256": digest,
                            "start_line": line_no,
                            "end_line": min(last, line_no + 8),
                            "suggested_read": (first, last),
                            "referenced_from": {
                                "path": path,
                                "line": hit,
                                "source_sha256": match["source_sha256"],
                                "provision": article + (paragraph or ""),
                                **({"target_line": target_line} if target_line else {}),
                            },
                        }
        return None

    def _read(self, args, check):
        check()
        reference = None
        repairs = []
        prior = None
        if "cursor" in args:
            saved = self.cursors.get(args["cursor"]) if isinstance(args["cursor"], str) else None
            if saved and saved[0] == "read":
                if set(args) != {"cursor"}:
                    raise ToolArgumentError("invalid_cursor")
                _, path, expected, start, end, column, reference = saved
            elif "source_id" in args or "path" in args:
                # Treat a stale or fabricated token as redundant only if the other
                # arguments independently resolve to a valid fresh read.
                args = {key: value for key, value in args.items() if key != "cursor"}
                repairs.append("invalid_cursor_fresh_read")
            else:
                raise ToolArgumentError("invalid_cursor")
        if "cursor" not in args:
            if "source_id" not in args and "path" not in args:
                raise ToolArgumentError("invalid_target")
            if "source_id" in args:
                id_ = args["source_id"]
                prior = self.sources.get(id_) if isinstance(id_, str) else None
                if prior is None or prior.generation != self.generation:
                    raise ToolArgumentError("unknown_source")
                if "path" in args:
                    if args["path"] in ("", None):
                        repairs.append(
                            "null_path_omitted" if args["path"] is None else "empty_path_omitted"
                        )
                    else:
                        if _normalize_relative_path(args["path"]) != prior.path:
                            raise ToolArgumentError("invalid_target")
                        repairs.append("matching_source_path")
            path = prior.path if prior else args["path"]
            expected = prior.sha256 if prior else None
            if prior:
                reference = self.references.get(prior.id)
            start, end, column = args.get("start_line"), args.get("end_line"), 0
        if not isinstance(path, str) or len(path) > 4096:
            raise ToolArgumentError("invalid_path")
        sha, lines, text = read_source_snapshot(self.root, path, expected)
        check()
        if "section_id" in args:
            node = next(
                (
                    n
                    for n in parse_heading_tree(text)
                    if self._section_id(path, sha, n) == args["section_id"]
                ),
                None,
            )
            if node is None:
                raise ToolArgumentError("unknown_section")
            if {"start_line", "end_line"} & args.keys():
                if not (
                    type(start) is int and type(end) is int
                    and node.heading_line <= start <= end <= node.section_end_line
                ):
                    raise ToolArgumentError("ambiguous_range")
                repairs.append("range_within_section")
            else:
                start, end = node.heading_line, node.section_end_line
        elif not ({"start_line", "end_line"} & args.keys()) and prior is not None:
            if not all(span.matches(lines) for span in prior.spans):
                raise ToolArgumentError("source_changed")
            suggested = self.read_ranges.get(prior.id)
            if not suggested:
                return self._read_response(path, sha, text, prior.spans, repairs=repairs)
            start, end = suggested
        if type(start) is not int or type(end) is not int:
            raise ToolArgumentError("invalid_line_range")
        if prior and reference:
            suggested = self.read_ranges.get(prior.id)
            target_line = reference.get("target_line")
            if target_line is not None:
                if not start <= target_line <= end:
                    reference = None
            elif not suggested or not (start <= suggested[0] <= suggested[1] <= end):
                reference = None
        # JSON control-character escapes can use six bytes per character.
        span, continuation = _page(lines, start, end, column, 6000)
        limit = 6000
        while byte_size(asdict(span)) + byte_size(path) + 2048 > TOOL_BYTES:
            limit //= 2
            span, continuation = _page(lines, start, end, column, limit)
        cursor = (
            self._cursor(("read", path, sha, continuation[0], end, continuation[1], reference))
            if continuation
            else None
        )
        return self._read_response(path, sha, text, (span,), cursor, reference=reference, repairs=repairs)

    def read_preceding_scope(self, source, number, check):
        """Read the preceding paragraph within the same verified article."""
        check()
        try:
            sha, lines, text = read_source_snapshot(self.root, source.path, source.sha256)
        except EvidenceViewError as exc:
            return {"error": exc.code}
        if not 1 <= number <= len(lines) or not any(
            span.start_line <= number <= span.end_line and span.matches(lines)
            for span in source.spans
        ):
            return {"error": "invalid_line_range"}
        for start in range(number - 1, max(0, number - 33), -1):
            line = lines[start - 1]
            if _article_label(_focus_text(line)):
                if not _preceding_items(lines, start, number):
                    break
                span, continuation = _page(lines, start, number - 1, 0, 3000)
                if continuation:
                    return {"error": "tool_result_too_large"}
                check()
                return self._read_response(source.path, sha, text, (span,))
            if line.strip() and not _NUMBERED_ITEM.match(line):
                break
        return {"error": "preceding_scope_unavailable"}

    def read_linked_origin(self, reference, check):
        """Read the single line bound to a resolved exception reference."""
        check()
        path = reference["path"]
        try:
            sha, lines, text = read_source_snapshot(self.root, path, reference["source_sha256"])
        except EvidenceViewError as exc:
            return {"error": exc.code}
        number = reference["line"]
        if not 1 <= number <= len(lines):
            return {"error": "invalid_line_range"}
        span, continuation = _page(lines, number, number, 0, 3000)
        if continuation:
            return {"error": "tool_result_too_large"}
        check()
        return self._read_response(path, sha, text, (span,))

    def linked_exception_target(self, source, complete_lines, check):
        """Resolve a single second-paragraph payment exception from a verified read."""
        hits = [
            number
            for number, line in complete_lines.items()
            if re.search(r"第2項[^。]{0,40}支給する場合を除", _focus_text(line))
        ]
        if len(hits) != 1:
            return None
        target = self._exception_reference(
            {"path": source.path, "source_sha256": source.sha256, "start_line": hits[0]},
            self._document_groups(),
            check,
        )
        if target:
            ref = target["referenced_from"]
            if ref.get("target_line") and ref["provision"].endswith("第2項"):
                return target
        return None

    def read_linked_target(self, target, check):
        """Read the article whose reference and expected snapshot were resolved locally."""
        check()
        path = target["path"]
        try:
            sha, lines, text = read_source_snapshot(self.root, path, target["source_sha256"])
        except EvidenceViewError as exc:
            return {"error": exc.code}
        start, end = target["suggested_read"]
        span, continuation = _page(lines, start, end, 0, 3000)
        if continuation:
            return {"error": "tool_result_too_large"}
        check()
        return self._read_response(path, sha, text, (span,), reference=target["referenced_from"])

    def _read_exception_roles(self, path, sha, spans):
        origins = {
            ref["line"]: ref["provision"]
            for ref in self.references.values()
            if ref["path"] == path and ref["source_sha256"] == sha
        }
        roles = {}
        for span in spans:
            lines = span.text.split("\n")
            for offset, line in enumerate(lines):
                number = span.start_line + offset
                if "前項" in line and _EXCEPTION_CLAUSE.search(line) and _preceding_items(lines, 1, offset + 1):
                    roles.setdefault("exception_scope_line", number)
                if provision := origins.get(number):
                    text = _focus_text(line)
                    if _EXCEPTION_CLAUSE.search(text) and any(
                        ref[2] + (ref[3] or "") == provision for ref in _LAW_REFERENCE.finditer(text)
                    ):
                        roles.setdefault("exception_origin", {"line": number, "provision": provision})
        return roles

    def _read_response(self, path, sha, text, spans, cursor=None, *, reference=None, repairs=()):
        result = self._record(path, sha, self._title(path, text), "read", spans)
        if reference:
            self.references[result.id] = reference
        self.trace.append({"tool": "read_source", "path": path, "source_id": result.id})
        return {
            "untrusted_source_data": result.public(),
            "citation": f"[{result.id}]",
            "cursor": cursor,
            "truncated": cursor is not None,
            **self._read_exception_roles(path, sha, spans),
            **({"argument_repairs": repairs} if repairs else {}),
            **(
                {"exception_target": reference["provision"], "referenced_from": reference}
                if reference
                else {}
            ),
        }

    def _list(self, args, check):
        saved = None
        limit = self._limit(args, 10, 20)
        query, offset = self._query(args), 0
        paths = sorted(set(self.list_paths()))
        check()
        inventory = hashlib.sha256()
        for path in paths:
            check()
            try:
                stat = (self.root / path).lstat()
            except OSError:
                continue
            inventory.update(encode([path, stat.st_mtime_ns, stat.st_size, stat.st_ino]).encode())
        fingerprint = inventory.hexdigest()
        restarted = False
        if "cursor" in args:
            token = args["cursor"]
            if not isinstance(token, str):
                raise ToolArgumentError("invalid_cursor")
            saved = self.cursors.get(token)
            if saved and saved[0] == "list":
                if "query" not in args:
                    query = saved[2]
                if saved[1] == fingerprint and query == saved[2]:
                    offset = saved[3]
                else:
                    restarted = True
            else:
                restarted = True
        if query:
            documents = [
                {"path": path, "title": Path(path).stem, "head_text": ""} for path in paths
            ]
            groups = {row["path"]: row for row in group_split_documents(documents)}
            matched = match_documents(query, list(groups.values()))
            by_title = [
                p["path"]
                for row in matched["candidates"]
                for p in groups[row["path"]].get("paths", [row])
            ]
            paths = list(
                dict.fromkeys([*by_title, *(p for p in paths if query.casefold() in p.casefold())])
            )
        results, used = [], 0
        for index in range(offset, len(paths)):
            check()
            if len(results) >= limit:
                break
            path = paths[index]
            try:
                sha, _, text = read_source_snapshot(self.root, path)
            except EvidenceViewError:
                offset = index + 1
                continue
            sections, truncated = self._sections(path, sha, text)
            row = {
                "path": path,
                "title": self._title(path, text),
                "document_numbers": sorted(document_numbers(text[:6000]))[:12],
                "sections": sections,
                "sections_truncated": truncated,
            }
            size = byte_size(row)
            if used + size + 512 > TOOL_BYTES:
                if results:
                    break
                row.update(sections=[], sections_truncated=True)
                size = byte_size(row)
                if size + 512 > TOOL_BYTES:
                    raise ToolArgumentError("metadata_too_large")
            results.append(row)
            used += size + 1
            offset = index + 1
        cursor = self._cursor(("list", fingerprint, query, offset)) if offset < len(paths) else None
        return {
            "documents": results,
            "cursor": cursor,
            "truncated": cursor is not None,
            **(
                {
                    "cursor_restarted": True,
                    "next_step": "一覧の先頭を返しました。続きには今回返した cursor を使ってください。",
                }
                if restarted
                else {}
            ),
        }

    def execute(self, name, args, check, deadline):
        if name not in TOOL_ARGUMENTS:
            return {"error": "unknown_tool"}
        if not isinstance(args, dict):
            return {"error": "invalid_arguments", "detail": "invalid_argument_object"}
        if args.keys() - TOOL_ARGUMENTS[name]:
            return {"error": "invalid_arguments", "detail": "unexpected_argument"}
        args = dict(args)
        if args.get("source_id") == "":
            del args["source_id"]
        if args.get("cursor") is None:
            args.pop("cursor", None)
        if name == "read_source" and args.get("section_id") is None:
            args.pop("section_id", None)
        check()
        source_position = len(self.new_ids)
        cursor_keys = set(self.cursors)
        trace_position = len(self.trace)

        def rollback():
            for id_ in self.new_ids[source_position:]:
                del self.sources[id_]
                self.read_ranges.pop(id_, None)
                self.references.pop(id_, None)
                self._next -= 1
            del self.new_ids[source_position:]
            del self.trace[trace_position:]
            for token in self.cursors.keys() - cursor_keys:
                del self.cursors[token]

        try:
            if name == "search_sources":
                value = self._search(args, check, deadline)
            elif name == "read_source":
                value = self._read(args, check)
            else:
                value = self._list(args, check)
            check()
            if byte_size(value) <= TOOL_BYTES:
                return value
        except EvidenceViewError as exc:
            rollback()
            return {"error": exc.code}
        except ToolArgumentError as exc:
            rollback()
            value = {"error": "invalid_arguments", "detail": str(exc)}
            if name == "read_source":
                value["next_step"] = "候補を読むときは source_id だけ、続きを読むときは返された cursor だけを指定してください。"
            return value
        except (ValueError, TypeError):
            rollback()
            return {"error": "invalid_arguments"}
        except Exception:
            rollback()
            raise
        rollback()
        return {"error": "tool_result_too_large"}
