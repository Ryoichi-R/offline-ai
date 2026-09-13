"""_resolve_reasoning() のテスト。"""

import os
from unittest.mock import patch

import pytest

from search import _resolve_reasoning


class TestResolveReasoning:
    """_resolve_reasoning() の優先順位解決・フォールバックテスト。"""

    def test_cli_arg_low(self):
        assert _resolve_reasoning("low") == "low"

    def test_cli_arg_medium(self):
        assert _resolve_reasoning("medium") == "medium"

    def test_cli_arg_high(self):
        assert _resolve_reasoning("high") == "high"

    def test_cli_arg_case_insensitive(self):
        assert _resolve_reasoning("HIGH") == "high"
        assert _resolve_reasoning("Low") == "low"
        assert _resolve_reasoning("MEDIUM") == "medium"

    def test_cli_arg_with_whitespace(self):
        assert _resolve_reasoning("  high  ") == "high"

    def test_invalid_cli_arg_fallback_to_low(self, capsys):
        result = _resolve_reasoning("invalid")
        assert result == "low"
        captured = capsys.readouterr()
        assert "無効な reasoning 値" in captured.err

    def test_none_cli_arg_uses_env_var(self):
        with patch.dict(os.environ, {"OFFLINE_AI_REASONING": "high"}):
            assert _resolve_reasoning(None) == "high"

    def test_none_cli_arg_env_var_case_insensitive(self):
        with patch.dict(os.environ, {"OFFLINE_AI_REASONING": "HIGH"}):
            assert _resolve_reasoning(None) == "high"

    def test_none_cli_arg_invalid_env_var_fallback(self, capsys):
        with patch.dict(os.environ, {"OFFLINE_AI_REASONING": "invalid"}):
            result = _resolve_reasoning(None)
            assert result == "low"

    def test_none_cli_arg_no_env_var_default_low(self):
        with patch.dict(os.environ, {}, clear=True):
            # OFFLINE_AI_REASONING が設定されていない場合
            env = os.environ.copy()
            env.pop("OFFLINE_AI_REASONING", None)
            with patch.dict(os.environ, env, clear=True):
                assert _resolve_reasoning(None) == "low"

    def test_cli_arg_overrides_env_var(self):
        with patch.dict(os.environ, {"OFFLINE_AI_REASONING": "low"}):
            assert _resolve_reasoning("high") == "high"

    def test_empty_string_cli_arg_uses_env_var(self):
        """空文字列は falsy なので環境変数にフォールバックする。"""
        with patch.dict(os.environ, {"OFFLINE_AI_REASONING": "high"}):
            assert _resolve_reasoning("") == "high"
