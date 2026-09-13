from model_config import DEFAULT_MODEL, detect_model_config
import search


def test_detect_model_config_accepts_utf8_bom(tmp_path):
    model_file = tmp_path / ".model"
    model_file.write_bytes(("﻿" + DEFAULT_MODEL).encode("utf-8"))

    assert detect_model_config(model_file) == DEFAULT_MODEL


def test_detect_embed_model_accepts_utf8_bom(tmp_path, monkeypatch):
    embed_file = tmp_path / ".model_embed"
    embed_file.write_bytes(("﻿" + search.RECOMMENDED_EMBED_MODEL).encode("utf-8"))
    monkeypatch.setattr(search, "EMBED_MODEL_CONFIG", embed_file)

    assert search.detect_embed_model() == search.RECOMMENDED_EMBED_MODEL
