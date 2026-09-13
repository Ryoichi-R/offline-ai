"""_is_model_available() と _migrate_model_file() のテスト。"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from search import (
    DEFAULT_MODEL,
    ModelStatus,
    detect_model,
    _is_model_available,
    _migrate_model_file,
    _LEGACY_MODELS,
)


class TestMigrateModelFile:
    """_migrate_model_file() のテスト。"""

    def test_default_model_is_gpt_oss_20b(self):
        assert DEFAULT_MODEL == "gpt-oss:20b"

    def test_default_model_is_not_legacy(self):
        assert DEFAULT_MODEL not in _LEGACY_MODELS

    def test_legacy_qwen_3b_migrated(self, tmp_path):
        model_file = tmp_path / ".model"
        model_file.write_text("qwen2.5:3b", encoding="utf-8")
        result = _migrate_model_file(model_file, "qwen2.5:3b")
        assert result == DEFAULT_MODEL
        assert model_file.read_text(encoding="utf-8") == DEFAULT_MODEL

    def test_legacy_qwen_7b_migrated(self, tmp_path):
        model_file = tmp_path / ".model"
        model_file.write_text("qwen2.5:7b", encoding="utf-8")
        result = _migrate_model_file(model_file, "qwen2.5:7b")
        assert result == DEFAULT_MODEL

    def test_legacy_qwen_14b_migrated(self, tmp_path):
        model_file = tmp_path / ".model"
        model_file.write_text("qwen2.5:14b", encoding="utf-8")
        result = _migrate_model_file(model_file, "qwen2.5:14b")
        assert result == DEFAULT_MODEL

    def test_legacy_qwen35_migrated(self, tmp_path):
        model_file = tmp_path / ".model"
        model_file.write_text("qwen3.5:35b-a3b", encoding="utf-8")
        result = _migrate_model_file(model_file, "qwen3.5:35b-a3b")
        assert result == DEFAULT_MODEL
        assert model_file.read_text(encoding="utf-8") == DEFAULT_MODEL

    def test_gpt_oss_20b_is_not_migrated(self, tmp_path):
        model_file = tmp_path / ".model"
        model_file.write_text("gpt-oss:20b", encoding="utf-8")
        result = _migrate_model_file(model_file, "gpt-oss:20b")
        assert result == "gpt-oss:20b"
        assert model_file.read_text(encoding="utf-8") == "gpt-oss:20b"

    def test_current_model_not_migrated(self, tmp_path):
        model_file = tmp_path / ".model"
        model_file.write_text(DEFAULT_MODEL, encoding="utf-8")
        result = _migrate_model_file(model_file, DEFAULT_MODEL)
        assert result == DEFAULT_MODEL

    def test_unknown_model_not_migrated(self, tmp_path):
        model_file = tmp_path / ".model"
        model_file.write_text("llama3:8b", encoding="utf-8")
        result = _migrate_model_file(model_file, "llama3:8b")
        assert result == "llama3:8b"
        # ファイルは変更されない
        assert model_file.read_text(encoding="utf-8") == "llama3:8b"

    def test_all_legacy_models_covered(self):
        assert _LEGACY_MODELS == {
            "qwen2.5:3b",
            "qwen2.5:7b",
            "qwen2.5:14b",
            "qwen3.5:35b-a3b",
        }

    def test_download_catalog_legacy_denylist_matches_runtime(self):
        catalog_path = Path(__file__).parents[2] / "_internal" / "download-manifest.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        assert set(catalog["legacyMigrationModels"]) == _LEGACY_MODELS
        selectable = {
            model["name"]
            for model in catalog["models"]
            if model.get("role") == "chat" and model.get("selectable") is True
        }
        assert selectable.isdisjoint(_LEGACY_MODELS)

    def test_migration_prints_warning(self, tmp_path, capsys):
        model_file = tmp_path / ".model"
        model_file.write_text("qwen2.5:7b", encoding="utf-8")
        _migrate_model_file(model_file, "qwen2.5:7b")
        captured = capsys.readouterr()
        assert "旧モデル" in captured.err
        assert DEFAULT_MODEL in captured.err

    def test_detect_model_missing_file_returns_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("search.MODEL_CONFIG", tmp_path / ".model")
        assert detect_model() == DEFAULT_MODEL

    def test_detect_model_empty_file_returns_default(self, tmp_path, monkeypatch):
        model_file = tmp_path / ".model"
        model_file.write_text("", encoding="utf-8")
        monkeypatch.setattr("search.MODEL_CONFIG", model_file)
        assert detect_model() == DEFAULT_MODEL

    def test_detect_model_respects_gpt_oss_20b(self, tmp_path, monkeypatch):
        model_file = tmp_path / ".model"
        model_file.write_text("gpt-oss:20b", encoding="utf-8")
        monkeypatch.setattr("search.MODEL_CONFIG", model_file)
        before = model_file.read_bytes()
        assert detect_model() == "gpt-oss:20b"
        assert model_file.read_bytes() == before

    def test_existing_qwen_model_is_respected_and_unchanged(self, tmp_path, monkeypatch):
        model_file = tmp_path / ".model"
        model_file.write_text("qwen3.5:9b\n", encoding="utf-8")
        monkeypatch.setattr("search.MODEL_CONFIG", model_file)
        before = model_file.read_bytes()
        assert detect_model() == "qwen3.5:9b"
        assert model_file.read_bytes() == before

    def test_detect_model_unknown_model_is_respected(self, tmp_path, monkeypatch):
        model_file = tmp_path / ".model"
        model_file.write_text("llama3:8b", encoding="utf-8")
        monkeypatch.setattr("search.MODEL_CONFIG", model_file)
        assert detect_model() == "llama3:8b"


class TestIsModelAvailable:
    """_is_model_available() の3状態テスト。"""

    def _mock_urlopen(self, models_list):
        """Ollama API レスポンスをモックする。"""
        response_data = json.dumps({"models": [{"name": m} for m in models_list]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("search.urllib.request.urlopen")
    def test_available_when_model_exists(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(["gpt-oss:20b", "llama3:8b"])
        result = _is_model_available("gpt-oss:20b")
        assert result == ModelStatus.AVAILABLE

    @patch("search.urllib.request.urlopen")
    def test_available_when_latest_tag_is_omitted(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(["bge-m3:latest"])
        result = _is_model_available("bge-m3")
        assert result == ModelStatus.AVAILABLE

    @patch("search.urllib.request.urlopen")
    def test_explicit_nonlatest_tag_does_not_match_latest(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(["bge-m3:latest"])
        result = _is_model_available("bge-m3:v1")
        assert result == ModelStatus.UNAVAILABLE

    @patch("search.urllib.request.urlopen")
    def test_unavailable_when_model_missing(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(["llama3:8b"])
        result = _is_model_available("gpt-oss:20b")
        assert result == ModelStatus.UNAVAILABLE

    @patch("search.urllib.request.urlopen")
    def test_unavailable_when_empty_list(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen([])
        result = _is_model_available("gpt-oss:20b")
        assert result == ModelStatus.UNAVAILABLE

    @patch("search.urllib.request.urlopen")
    def test_unknown_on_connection_error(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")
        result = _is_model_available("gpt-oss:20b")
        assert result == ModelStatus.UNKNOWN

    @patch("search.urllib.request.urlopen")
    def test_unknown_on_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = _is_model_available("gpt-oss:20b")
        assert result == ModelStatus.UNKNOWN

    @patch("search.urllib.request.urlopen")
    def test_unknown_on_generic_exception(self, mock_urlopen):
        mock_urlopen.side_effect = RuntimeError("unexpected")
        result = _is_model_available("gpt-oss:20b")
        assert result == ModelStatus.UNKNOWN
