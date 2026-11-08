"""Keep selected documents separate from hidden archives before normalization."""

import os

import pytest

import search


def put(root, path):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("synthetic reference", encoding="utf-8")
    return target


def test_hidden_archive_with_nested_skill_source_never_becomes_live(tmp_path):
    put(tmp_path, "rules/current.md")
    put(tmp_path, ".research-policy/archive/skill-source/legacy/old.md")
    put(tmp_path, ".hidden/skill-source/secrets.json")
    files = search.iter_source_files(tmp_path)
    assert [rel for _, rel in files] == ["rules/current.md"]
    chunks, manifest = search.build_source_snapshot(tmp_path)
    assert set(manifest) == {"rules/current.md"}
    assert {chunk["path"] for chunk in chunks} == {"rules/current.md"}


def test_visible_nested_skill_source_preserves_relative_path(tmp_path):
    put(tmp_path, "collection/skill-source/current.md")
    assert [rel for _, rel in search.iter_source_files(tmp_path)] == [
        "collection/skill-source/current.md"
    ]
    chunks, manifest = search.build_source_snapshot(tmp_path)
    relative = "collection/skill-source/current.md"
    assert set(manifest) == {relative}
    assert {chunk["path"] for chunk in chunks} == {relative}
    reader = search.SkillSourceLineReader(tmp_path)
    assert reader(relative, manifest[relative]["file_sha256"]) == ["synthetic reference"]


@pytest.mark.parametrize("directory", ["node_modules", "__pycache__", ".hidden"])
def test_excluded_trees_are_not_entered(tmp_path, monkeypatch, directory):
    put(tmp_path, directory + "/skill-source/old.md")
    put(tmp_path, "current.md")
    original = os.scandir

    def scan(path):
        if directory in str(path).split(os.sep):
            raise AssertionError("Excluded directory entered")
        return original(path)

    monkeypatch.setattr(os, "scandir", scan)
    assert [rel for _, rel in search.iter_source_files(tmp_path)] == ["current.md"]


def test_symlink_files_and_directories_are_not_followed(tmp_path):
    source = tmp_path / "selected"
    source.mkdir()
    external = put(tmp_path, "outside/other.md")
    try:
        (source / "other.md").symlink_to(external)
        (source / "linked").symlink_to(external.parent, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    assert search.iter_source_files(source) == []


def test_scan_failure_is_not_reported_as_an_empty_success(tmp_path, monkeypatch):
    def scan(_):
        raise PermissionError("Synthetic unavailable folder")

    monkeypatch.setattr(os, "scandir", scan)
    with pytest.raises(PermissionError):
        search.build_source_snapshot(tmp_path)


def test_missing_root_remains_an_empty_scan(tmp_path):
    assert search.iter_source_files(tmp_path / "missing") == []


@pytest.mark.parametrize(
    "relative",
    [
        "collection/skill-source/current.md",
        ".hidden/skill-source/old.md",
        "node_modules/skill-source/old.md",
    ],
)
def test_relative_ancestors_survive_normalization(relative):
    assert search.normalize_source_path(relative) == relative


@pytest.mark.parametrize(
    "relative", [".hidden/skill-source/old.md", "node_modules/skill-source/old.md"]
)
def test_nested_marker_cannot_remove_excluded_ancestors(relative):
    assert not search.is_allowed_source_path(relative)


def test_legacy_absolute_paths_and_relative_cleanup_keep_original_contract():
    assert (
        search.normalize_source_path(r"C:\selected\skill-source\rules\current.md")
        == "rules/current.md"
    )
    assert (
        search.normalize_source_path("/selected/skill-source/rules/current.md")
        == "rules/current.md"
    )
    assert search.normalize_source_path("rules/./current.md") == "rules/current.md"
