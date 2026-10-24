"""検索用資料の安全な根拠行読出し。

このモジュールは HTTP や検索実装へ依存せず、登録済みの不透明な根拠 ID
だけを source 相対パスへ解決する。本文は registry に保持せず、閲覧時に
認証セッション・source 境界・開いたハンドル・検索時 hash を確認する。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import ctypes
import hashlib
import os
from pathlib import Path
import re
import secrets
import threading
import time


MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024
MAX_VIEW_LINES = 200
MAX_VIEW_TEXT_BYTES = 64 * 1024
EVIDENCE_TTL_SECONDS = 30 * 60
EVIDENCE_REGISTRY_LIMIT = 512

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_EXCLUDED_SOURCE_DIRS = {"__pycache__", "node_modules"}
_EXCLUDED_SOURCE_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "secrets.*",
    "*.secret.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "*.metadata.json",
)
_SOURCE_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".html", ".htm"}


class EvidenceViewError(RuntimeError):
    """閲覧 API へ返せる安全なエラー。"""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class EvidenceReference:
    relative_path: str
    start_line: int
    end_line: int
    source_sha256: str
    session_id: str
    created_at: float


def _invalid_reference(message: str = "根拠を表示できません") -> EvidenceViewError:
    return EvidenceViewError("evidence_not_found", message, 404)


def _normalize_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise _invalid_reference()
    raw = value.replace("\\", "/").strip()
    if not raw or "\x00" in raw:
        raise _invalid_reference()
    # Drive 指定、UNC、POSIX 絶対 path、ADS、traversal、空の要素を拒否する。
    if raw.startswith("/") or raw.startswith("//") or re.match(r"^[A-Za-z]:", raw):
        raise _invalid_reference()
    if ":" in raw:
        raise _invalid_reference()
    parts = raw.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise _invalid_reference()
    for directory in parts[:-1]:
        if directory.startswith(".") or directory.lower() in _EXCLUDED_SOURCE_DIRS:
            raise _invalid_reference()
    filename = parts[-1]
    lower_name = filename.lower()
    if any(_fnmatch(lower_name, pattern) for pattern in _EXCLUDED_SOURCE_FILE_PATTERNS):
        raise _invalid_reference()
    if Path(lower_name).suffix not in _SOURCE_EXTENSIONS:
        raise _invalid_reference()
    return "/".join(parts)


def _fnmatch(value: str, pattern: str) -> bool:
    """少数の source policy pattern 用の標準ライブラリ実装。"""
    import fnmatch

    return fnmatch.fnmatchcase(value, pattern)


def _is_reparse_point(path: Path) -> bool:
    try:
        stat = os.lstat(path)
    except OSError as exc:
        raise _invalid_reference() from exc
    if path.is_symlink():
        return True
    # st_file_attributes は Windows の os.stat_result にだけ現れるため、
    # モジュール import 時には参照せず、読出し直前に確認する。
    return bool(getattr(stat, "st_file_attributes", 0) & 0x400)


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [os.path.normcase(os.path.abspath(str(path))), os.path.normcase(os.path.abspath(str(root)))]
        ) == os.path.normcase(os.path.abspath(str(root)))
    except (OSError, ValueError):
        return False


def _assert_path_boundary(
    root: Path,
    relative_path: str,
    *,
    missing_code: str = "evidence_unavailable",
) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500) from exc
    if not resolved_root.is_dir() or _is_reparse_point(resolved_root):
        raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500)

    candidate = resolved_root.joinpath(*relative_path.split("/"))
    current = resolved_root
    try:
        for part in relative_path.split("/"):
            current = current / part
            try:
                os.lstat(current)
            except FileNotFoundError as exc:
                if missing_code == "source_changed":
                    raise EvidenceViewError(
                        "source_changed",
                        "検索後に資料が削除されたため再検索してください",
                        409,
                    ) from exc
                raise EvidenceViewError(
                    "evidence_unavailable", "資料を読み出せません", 500
                ) from exc
            except OSError as exc:
                raise EvidenceViewError(
                    "evidence_unavailable", "資料を読み出せません", 500
                ) from exc
            if _is_reparse_point(current):
                raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500)
        resolved_candidate = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        if missing_code == "source_changed":
            raise EvidenceViewError(
                "source_changed",
                "検索後に資料が削除されたため再検索してください",
                409,
            ) from exc
        raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500) from exc
    except EvidenceViewError:
        raise
    except OSError as exc:
        raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500) from exc
    if not _is_within(resolved_candidate, resolved_root) or not resolved_candidate.is_file():
        raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500)
    return candidate


def _final_path_from_handle(fd: int) -> Path | None:
    """Windows の開いたハンドルの最終 path を遅延確認する。"""
    if os.name != "nt":
        return None
    try:
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetFinalPathNameByHandleW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
        kernel32.GetFinalPathNameByHandleW.restype = ctypes.c_uint32
        handle = msvcrt.get_osfhandle(fd)
        buffer = ctypes.create_unicode_buffer(32768)
        length = kernel32.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if not length or length >= len(buffer):
            return None
        value = buffer.value[:length]
        if value.startswith("\\\\?\\"):
            value = value[4:]
        return Path(value)
    except Exception:
        return None


def _identity(stat) -> tuple[int, int, int, int]:
    return (
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
        int(getattr(stat, "st_size", -1)),
        int(getattr(stat, "st_mtime_ns", 0)),
    )


def _truncate_utf8(value: str, max_bytes: int) -> str:
    raw = value.encode("utf-8")
    if len(raw) <= max_bytes:
        return value
    return raw[:max_bytes].decode("utf-8", errors="ignore")


class EvidenceRegistry:
    """セッション束縛された短期根拠 registry。"""

    def __init__(
        self,
        source_root: Path,
        *,
        ttl_seconds: int = EVIDENCE_TTL_SECONDS,
        max_entries: int = EVIDENCE_REGISTRY_LIMIT,
        clock=time.monotonic,
    ):
        self.source_root = Path(source_root)
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._entries: OrderedDict[str, EvidenceReference] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired_locked(self, now: float) -> None:
        while self._entries:
            first_id, first = next(iter(self._entries.items()))
            if now - first.created_at < self.ttl_seconds:
                break
            self._entries.pop(first_id, None)

    def register(
        self,
        *,
        session_id: str,
        relative_path: object,
        start_line: object,
        end_line: object,
        source_sha256: object,
    ) -> str | None:
        """表示可能な根拠だけ登録し、不正・不足時は None を返す。"""
        if not isinstance(session_id, str) or not session_id:
            return None
        try:
            normalized = _normalize_relative_path(relative_path)
            if (
                isinstance(start_line, bool)
                or not isinstance(start_line, int)
                or isinstance(end_line, bool)
                or not isinstance(end_line, int)
                or start_line < 1
                or end_line < start_line
            ):
                return None
            if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256):
                return None
            _assert_path_boundary(self.source_root, normalized)
        except EvidenceViewError:
            return None

        with self._lock:
            # 登録時刻と挿入順を同じlock下で確定し、期限順を維持する。
            now = self._clock()
            reference = EvidenceReference(
                relative_path=normalized,
                start_line=start_line,
                end_line=end_line,
                source_sha256=source_sha256.lower(),
                session_id=session_id,
                created_at=now,
            )
            self._purge_expired_locked(now)
            while len(self._entries) >= self.max_entries:
                self._entries.popitem(last=False)
            evidence_id = secrets.token_urlsafe(18)
            self._entries[evidence_id] = reference
        return evidence_id

    def view(self, evidence_id: object, *, session_id: str) -> dict:
        if not isinstance(evidence_id, str) or not evidence_id or not isinstance(session_id, str):
            raise _invalid_reference()
        with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            reference = self._entries.get(evidence_id)
            if reference is None or not secrets.compare_digest(reference.session_id, session_id):
                raise _invalid_reference()
            # 閲覧しても登録順・有効期限を延長しない。
        return self._read_reference(reference)

    def _read_reference(self, reference: EvidenceReference) -> dict:
        path = _assert_path_boundary(
            self.source_root,
            reference.relative_path,
            missing_code="source_changed",
        )
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(path, flags)
        except OSError as exc:
            raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500) from exc

        try:
            initial_handle_stat = os.fstat(fd)
            if initial_handle_stat.st_size > MAX_SOURCE_FILE_BYTES:
                raise EvidenceViewError("evidence_too_large", "資料が閲覧上限を超えています", 413)
            final_path = _final_path_from_handle(fd)
            if os.name == "nt":
                if final_path is None:
                    raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500)
                try:
                    root = self.source_root.resolve(strict=True)
                except OSError as exc:
                    raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500) from exc
                if not _is_within(final_path, root):
                    raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500)
            with os.fdopen(fd, "rb", closefd=True) as handle:
                data = handle.read(MAX_SOURCE_FILE_BYTES + 1)
                after_handle_stat = os.fstat(handle.fileno())
            fd = -1
            if len(data) > MAX_SOURCE_FILE_BYTES:
                raise EvidenceViewError("evidence_too_large", "資料が閲覧上限を超えています", 413)
            if _identity(initial_handle_stat) != _identity(after_handle_stat):
                raise EvidenceViewError("source_changed", "検索後に資料が変更されたため再検索してください", 409)
            try:
                current_path_stat = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                raise EvidenceViewError("source_changed", "検索後に資料が変更されたため再検索してください", 409) from exc
            if _identity(initial_handle_stat) != _identity(current_path_stat):
                raise EvidenceViewError("source_changed", "検索後に資料が変更されたため再検索してください", 409)
            digest = hashlib.sha256(data).hexdigest()
            if not secrets.compare_digest(digest, reference.source_sha256):
                raise EvidenceViewError("source_changed", "検索後に資料が変更されたため再検索してください", 409)
        except EvidenceViewError:
            raise
        except (OSError, ValueError) as exc:
            raise EvidenceViewError("evidence_unavailable", "資料を読み出せません", 500) from exc
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

        lines = data.decode("utf-8", errors="replace").splitlines()
        if lines:
            highlight_start = min(reference.start_line, len(lines))
            highlight_end = min(max(reference.end_line, highlight_start), len(lines))
            first_line = max(1, highlight_start - 10)
            last_line = min(len(lines), highlight_end + 10, first_line + MAX_VIEW_LINES - 1)
        else:
            highlight_start = reference.start_line
            highlight_end = reference.end_line
            first_line = last_line = 0

        view_lines = []
        used_bytes = 0
        truncated = bool(lines and last_line < min(len(lines), highlight_end + 10))
        for number in range(first_line, last_line + 1):
            line = lines[number - 1]
            line_bytes = len(line.encode("utf-8")) + (1 if view_lines else 0)
            if used_bytes + line_bytes > MAX_VIEW_TEXT_BYTES:
                remaining = MAX_VIEW_TEXT_BYTES - used_bytes
                if remaining > 0:
                    view_lines.append(
                        {
                            "lineNumber": number,
                            "text": _truncate_utf8(line, remaining),
                            "highlighted": highlight_start <= number <= highlight_end,
                        }
                    )
                truncated = True
                break
            view_lines.append(
                {
                    "lineNumber": number,
                    "text": line,
                    "highlighted": highlight_start <= number <= highlight_end,
                }
            )
            used_bytes += line_bytes

        return {
            "relativePath": reference.relative_path,
            "sourceSha256": reference.source_sha256,
            "startLine": reference.start_line,
            "endLine": reference.end_line,
            "highlightStartLine": highlight_start,
            "highlightEndLine": highlight_end,
            "lines": view_lines,
            "truncated": truncated,
        }
