"""CLI標準出力のUTF-8強制（cp932コンソールでのUnicodeEncodeError回避）。

build_evidence_summary が根拠一覧の区切りにem dash（U+2014）を使っており、
cp932（Windows日本語既定コードページ）のコンソールへ print すると
UnicodeEncodeError で search.py の CLI 全体がクラッシュしていた
（P2実機受入で発見、tests/eval/run_eval.py の既存 _force_utf8_stdio と
同じ対処を search.main() にも適用して解消する）。
"""

import io
import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import search  # noqa: E402


class _FakeStream:
    """sys.stdout/stderr相当。encoding属性とreconfigure()の有無を切替できる。"""

    def __init__(self, *, has_reconfigure: bool, raise_on_reconfigure: Exception | None = None):
        self.encoding = "cp932"
        self._raise = raise_on_reconfigure
        if has_reconfigure:
            self.reconfigure = self._reconfigure

    def _reconfigure(self, encoding=None, errors=None):
        if self._raise is not None:
            raise self._raise
        self.encoding = encoding


def test_force_utf8_stdio_reconfigures_stdout_and_stderr(monkeypatch):
    fake_out = _FakeStream(has_reconfigure=True)
    fake_err = _FakeStream(has_reconfigure=True)
    monkeypatch.setattr(search.sys, "stdout", fake_out)
    monkeypatch.setattr(search.sys, "stderr", fake_err)

    search._force_utf8_stdio()

    assert fake_out.encoding == "utf-8"
    assert fake_err.encoding == "utf-8"


def test_force_utf8_stdio_tolerates_stream_without_reconfigure(monkeypatch):
    """reconfigure を持たないstream（古いPython/一部のリダイレクト先）でも例外を出さない。"""
    fake_out = _FakeStream(has_reconfigure=False)
    monkeypatch.setattr(search.sys, "stdout", fake_out)
    monkeypatch.setattr(search.sys, "stderr", fake_out)

    search._force_utf8_stdio()  # 例外を送出しないことを確認する


def test_force_utf8_stdio_tolerates_reconfigure_failure(monkeypatch):
    """reconfigure自体が失敗する環境（すでに一部を読み書き済み等）でも落ちない。"""
    fake_out = _FakeStream(has_reconfigure=True, raise_on_reconfigure=ValueError("already used"))
    monkeypatch.setattr(search.sys, "stdout", fake_out)
    monkeypatch.setattr(search.sys, "stderr", fake_out)

    search._force_utf8_stdio()  # 例外を送出しないことを確認する


def test_evidence_summary_em_dash_crashes_raw_cp932_but_not_after_reconfigure():
    """build_evidence_summary の区切り文字（em dash, U+2014）は、素のcp932
    ストリームへの書き込みでは回帰の原因そのもの（UnicodeEncodeError）を再現し、
    reconfigure(errors="replace") 後は同じ内容を書き込んでもクラッシュしない
    （文字化けはあり得るが、CLI全体が落ちないことを保証する）。
    """
    summary = search.build_evidence_summary(
        [{"path": "doc.md", "heading": "見出し", "start_line": 1, "end_line": 2}],
        evidence_status="sufficient",
        confidence=0.9,
    )
    assert "—" in summary  # 区切り文字自体は変更しない（表示上の互換性を保つ）

    raw_cp932 = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict")
    try:
        raw_cp932.write(summary)
        raw_cp932.flush()
        raise AssertionError("expected UnicodeEncodeError under raw cp932")
    except UnicodeEncodeError:
        pass

    reconfigured = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict")
    reconfigured.reconfigure(encoding="utf-8", errors="replace")
    reconfigured.write(summary)  # reconfigure後は例外を送出しない
    reconfigured.flush()


def test_main_calls_force_utf8_stdio_before_parsing_args(monkeypatch):
    """main() はCLI引数解析やOllama呼び出しより前に必ずUTF-8化する。"""
    calls = []
    monkeypatch.setattr(search, "_force_utf8_stdio", lambda: calls.append("utf8"))
    monkeypatch.setattr(search.sys, "argv", ["search.py", "--help"])

    with pytest.raises(SystemExit):
        search.main()  # --help はargparseが直後にSystemExitする

    assert calls == ["utf8"]
