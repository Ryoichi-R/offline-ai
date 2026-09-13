import os

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402


def test_normalize_source_path_merges_windows_and_posix():
    assert search.normalize_source_path(r"rules\leave.md") == "rules/leave.md"
    assert search.normalize_source_path("skill-source/rules/leave.md") == "rules/leave.md"


def test_source_policy_excludes_secret_and_hidden_paths():
    assert not search.is_allowed_source_path("secrets.json")
    assert not search.is_allowed_source_path("secrets.yaml")
    assert not search.is_allowed_source_path(".env")
    assert not search.is_allowed_source_path(".env.local")
    assert not search.is_allowed_source_path(".hidden/doc.md")
    assert not search.is_allowed_source_path("keys/private.pem")
    assert search.is_allowed_source_path("rules/leave.md")
    assert search.is_allowed_source_path("faq/index.html")
    assert not search.is_allowed_source_path("logs/app.log")


def test_iter_source_files_uses_common_policy(tmp_path):
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "leave.md").write_text("休暇", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "doc.md").write_text("secret", encoding="utf-8")
    (tmp_path / "secrets.json").write_text("secret", encoding="utf-8")

    rels = [rel for _, rel in search.iter_source_files(tmp_path)]

    assert rels == ["rules/leave.md"]


def test_source_policy_excludes_metadata_sidecars():
    """sidecar は原本の複製であり、独立した資料として検索・根拠提示してはならない。"""
    assert not search.is_allowed_source_path("guides/handbook.md.metadata.json")
    assert not search.is_allowed_source_path("guides/handbook.metadata.json")
    assert not search.is_allowed_source_path(r"guides\handbook.md.metadata.json")
    # sidecar 以外の JSON 資料は従来どおり検索対象に残す。
    assert search.is_allowed_source_path("data/inventory.json")
    assert search.is_allowed_source_path("data/metadata-guide.json")


def test_iter_source_files_skips_metadata_sidecars(tmp_path):
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "handbook.md").write_text("# 手引き", encoding="utf-8")
    (tmp_path / "guides" / "handbook.md.metadata.json").write_text("{}", encoding="utf-8")

    rels = [rel for _, rel in search.iter_source_files(tmp_path)]

    assert rels == ["guides/handbook.md"]
