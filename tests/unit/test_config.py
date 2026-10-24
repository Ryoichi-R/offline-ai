import importlib.util
from pathlib import Path

INTERNAL_DIR = Path(__file__).resolve().parents[2] / "_internal"

import pytest

from config import (
    SEARCH_TIMEOUT_DEFAULT,
    SEARCH_TIMEOUT_MAX,
    SEARCH_TIMEOUT_MIN,
    env_timeout,
    env_int,
    get_ollama_host,
    get_rerank_config,
    validate_ollama_host,
    validate_rerank_endpoint,
    validate_search_timeout_seconds,
    validate_timeout,
)


def test_env_int_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("OFFLINE_AI_TEST_INT", "not-int")

    assert env_int("OFFLINE_AI_TEST_INT", 7) == 7


def test_env_int_out_of_range_falls_back(monkeypatch):
    monkeypatch.setenv("OFFLINE_AI_TEST_INT", "0")

    assert env_int("OFFLINE_AI_TEST_INT", 7, min_value=1) == 7


def test_env_int_valid(monkeypatch):
    monkeypatch.setenv("OFFLINE_AI_TEST_INT", "12")

    assert env_int("OFFLINE_AI_TEST_INT", 7, min_value=1, max_value=20) == 12


def test_env_timeout_rejects_invalid_values(monkeypatch):
    for value in ("0", "-1", "not-a-number", "999999"):
        monkeypatch.setenv("OFFLINE_AI_TEST_TIMEOUT", value)
        with pytest.raises(ValueError, match="OFFLINE_AI_TEST_TIMEOUT"):
            env_timeout("OFFLINE_AI_TEST_TIMEOUT", 10, max_value=3600)


def test_validate_timeout_accepts_bounded_float():
    assert validate_timeout("TEST_TIMEOUT", "1.5", max_value=10) == 1.5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
        ("https://[::1]:11434", "https://[::1]:11434"),
    ],
)
def test_validate_ollama_host_accepts_loopback_base_urls(value, expected):
    assert validate_ollama_host(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "ftp://localhost:11434",
        "http://example.com:11434",
        "http://user@localhost:11434",
        "http://localhost:11434/api",
        "http://localhost:not-a-port",
        "localhost:11434",
    ],
)
def test_validate_ollama_host_rejects_unsafe_or_non_base_urls(value):
    with pytest.raises(ValueError, match="OLLAMA_HOST"):
        validate_ollama_host(value)


def test_get_ollama_host_reads_shared_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:22434/")

    assert get_ollama_host() == "http://127.0.0.1:22434"


def test_validate_rerank_endpoint_accepts_loopback_alias():
    assert (
        validate_rerank_endpoint("http://127.0.0.1:8012/v1/rerank")
        == "http://127.0.0.1:8012/v1/rerank"
    )


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com:8012/v1/rerank",
        "ftp://127.0.0.1:8012/v1/rerank",
        "http://127.0.0.1:8012/other",
        "http://user@127.0.0.1:8012/v1/rerank",
    ],
)
def test_validate_rerank_endpoint_rejects_remote_or_unknown_endpoint(value):
    with pytest.raises(ValueError, match="OFFLINE_AI_RERANK_ENDPOINT"):
        validate_rerank_endpoint(value)


def test_config_import_fails_fast_for_invalid_ollama_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example.com:11434")
    spec = importlib.util.spec_from_file_location(
        "config_invalid_host_test", INTERNAL_DIR / "config.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    with pytest.raises(ValueError, match="許可されないホスト"):
        spec.loader.exec_module(module)


def test_rerank_config_defaults_off(monkeypatch):
    for name in (
        "OFFLINE_AI_RERANK_MODE",
        "OFFLINE_AI_RERANK_MODEL",
        "OFFLINE_AI_RERANK_ENDPOINT",
        "OFFLINE_AI_RERANK_STRICT",
    ):
        monkeypatch.delenv(name, raising=False)

    config = get_rerank_config()

    assert config.mode == "off"
    assert config.top_n == 24


def test_local_rerank_requires_model_when_strict(monkeypatch):
    monkeypatch.setenv("OFFLINE_AI_RERANK_MODE", "local")
    monkeypatch.delenv("OFFLINE_AI_RERANK_MODEL", raising=False)
    monkeypatch.setenv("OFFLINE_AI_RERANK_STRICT", "true")

    with pytest.raises(ValueError):
        get_rerank_config()


def test_local_rerank_config_accepts_loopback_endpoint(monkeypatch):
    monkeypatch.setenv("OFFLINE_AI_RERANK_MODE", "local")
    monkeypatch.setenv("OFFLINE_AI_RERANK_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setenv("OFFLINE_AI_RERANK_ENDPOINT", "http://localhost:8012/v1/rerank")

    config = get_rerank_config()

    assert config.mode == "local"
    assert config.endpoint == "http://localhost:8012/v1/rerank"


# --- 検索単位の全体timeout契約（300〜600秒） ---------------------------------


def test_search_timeout_bounds_are_300_to_600_default_300():
    assert SEARCH_TIMEOUT_MIN == 300
    assert SEARCH_TIMEOUT_MAX == 600
    assert SEARCH_TIMEOUT_DEFAULT == 300


@pytest.mark.parametrize("value", [300, 450, 600, "300", "600", " 300 ", 500])
def test_validate_search_timeout_seconds_accepts_boundary_and_mid_values(value):
    result = validate_search_timeout_seconds(value)
    assert isinstance(result, int)
    assert SEARCH_TIMEOUT_MIN <= result <= SEARCH_TIMEOUT_MAX


@pytest.mark.parametrize(
    "value",
    [119, 601, 0, -1, "119", "601", "300.5", "abc", "", True, False, 1.5, 300.0, "1_000", None],
)
def test_validate_search_timeout_seconds_rejects_out_of_contract_values(value):
    with pytest.raises(ValueError):
        validate_search_timeout_seconds(value)
