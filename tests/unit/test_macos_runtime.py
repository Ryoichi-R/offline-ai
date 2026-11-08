from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location(
    "offline_ai_macos_runtime", ROOT / "scripts" / "macos_runtime.py"
)
macos_runtime = module_from_spec(SPEC)
SPEC.loader.exec_module(macos_runtime)


def test_mount_identity_accepts_configured_exfat_volume(tmp_path):
    mount = tmp_path / "SUNEAST"
    models = mount / "ollama-models"
    models.mkdir(parents=True)
    resolved, resolved_mount = macos_runtime.validate_mount_identity(
        models,
        "AABB-CCDD",
        {
            "MountPoint": str(mount),
            "VolumeUUID": "aabb-ccdd",
            "FilesystemName": "ExFAT",
        },
    )
    assert resolved == models.resolve()
    assert resolved_mount == mount.resolve()


def test_mount_identity_rejects_same_name_volume_with_different_uuid(tmp_path):
    mount = tmp_path / "SUNEAST"
    models = mount / "ollama-models"
    models.mkdir(parents=True)
    with pytest.raises(ValueError, match="UUID"):
        macos_runtime.validate_mount_identity(
            models,
            "expected",
            {
                "MountPoint": str(mount),
                "VolumeUUID": "different",
                "FilesystemName": "ExFAT",
            },
        )


def test_mount_identity_rejects_model_directory_outside_mount(tmp_path):
    mount = tmp_path / "SUNEAST"
    models = tmp_path / "other" / "ollama-models"
    mount.mkdir()
    models.mkdir(parents=True)
    with pytest.raises(ValueError, match="inside"):
        macos_runtime.validate_mount_identity(
            models,
            "uuid",
            {"MountPoint": str(mount), "VolumeUUID": "uuid", "FilesystemName": "ExFAT"},
        )


def test_owner_record_requires_same_path_volume_binary_and_process_start():
    record = {
        "modelsDirectory": "/Volumes/SUNEAST/ollama-models",
        "volumeUuid": "AABB-CCDD",
        "ollamaExecutable": "/opt/ollama",
        "processStart": "Mon Sep 27 10:00:00 2026",
    }
    assert macos_runtime.owner_record_matches(
        record,
        "/Volumes/SUNEAST/ollama-models",
        "aabb-ccdd",
        "/opt/ollama",
        "Mon Sep 27 10:00:00 2026",
        "/opt/ollama serve",
    )
    assert not macos_runtime.owner_record_matches(
        record,
        "/Volumes/OTHER/ollama-models",
        "aabb-ccdd",
        "/opt/ollama",
        "Mon Sep 27 10:00:00 2026",
        "/opt/ollama serve",
    )


def test_free_space_guard_rejects_a_full_model_volume(tmp_path):
    with pytest.raises(ValueError, match="minimum free space"):
        macos_runtime.validate_free_space(tmp_path, 1024, free_bytes=1023)


def test_free_space_guard_accepts_configured_reserve(tmp_path):
    assert macos_runtime.validate_free_space(tmp_path, 1024, free_bytes=1024) == 1024


def _launcher_fixture(tmp_path):
    import json
    import shutil
    import sys

    root = tmp_path / "Mac's working tree"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (root / "_internal").mkdir()
    for name in ("start-macos.sh", "start-ollama-macos.sh"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    python = root / "Python's directory" / "python"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    ollama = root / "fake ollama"
    ollama.write_text("#!/bin/bash\nprintf 'ollama version 0.34.4\\n'\n")
    ollama.chmod(0o700)
    config = root / "runtime.json"
    config.write_text(
        json.dumps(
            {
                "python_executable": str(python),
                "ollama_executable": str(ollama),
                "models_directory": str(root / "models"),
                "volume_uuid": "test-uuid",
                "minimum_free_bytes": 1024,
                "ollama_version": "0.34.4",
            }
        )
    )
    (scripts / "macos_runtime.py").write_text("""import json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
command = sys.argv[1]
with (root / "commands").open("a") as log:
    log.write(command + "\\n")
if command == "config-value":
    config = json.loads(Path(sys.argv[sys.argv.index("--path")+1]).read_text())
    print(config[sys.argv[sys.argv.index("--key")+1]])
elif command == "resolve-model":
    print("qwen3.5:122b")
elif command == "validate-storage" and (root / "reject-storage").exists():
    raise SystemExit(1)
""")
    for name in ("search.py", "web_server.py"):
        (root / "_internal" / name).write_text("""from pathlib import Path
root = Path(__file__).resolve().parents[1]
(root / "product-started").write_text("started")
""")
    return root, config


@pytest.mark.parametrize("mode", ["cli", "web"])
@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="POSIX launcher requires bash and symlink support",
)
def test_launcher_waits_for_storage_validation_even_with_owned_server(tmp_path, mode):
    import subprocess

    root, config = _launcher_fixture(tmp_path)
    (root / "reject-storage").touch()
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts/start-macos.sh"),
            "--runtime-config",
            str(config),
            mode,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert not (root / "product-started").exists()
    assert "validate-storage" in (root / "commands").read_text()


@pytest.mark.parametrize("mode", ["cli", "web"])
@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="POSIX launcher requires bash and symlink support",
)
def test_launcher_accepts_spaces_and_apostrophes_after_child_validation(tmp_path, mode):
    import subprocess

    root, config = _launcher_fixture(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts/start-macos.sh"),
            "--runtime-config",
            str(config),
            "--model",
            "qwen3.5:122b",
            mode,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "product-started").exists()
    assert "validate-storage" in (root / "commands").read_text()


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="POSIX launcher requires bash and symlink support",
)
def test_launcher_rejects_success_exit_without_readiness_notification(tmp_path):
    import subprocess

    root, config = _launcher_fixture(tmp_path)
    (root / "scripts/start-ollama-macos.sh").write_text("#!/bin/bash\nexit 0\n")
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts/start-macos.sh"),
            "--runtime-config",
            str(config),
            "cli",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert not (root / "product-started").exists()


def test_storage_queries_the_mount_not_a_nested_model_directory(tmp_path, monkeypatch):
    import plistlib
    import types

    mount = tmp_path / "external"
    models = mount / "nested" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(macos_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(
        macos_runtime.os.path, "ismount", lambda path: Path(path) == mount
    )

    def diskutil(command, **kwargs):
        assert command == ["diskutil", "info", "-plist", str(mount)]
        return types.SimpleNamespace(
            stdout=plistlib.dumps(
                {
                    "MountPoint": str(mount),
                    "VolumeUUID": "EXPECTED",
                    "FilesystemName": "ExFAT",
                }
            )
        )

    monkeypatch.setattr(macos_runtime.subprocess, "run", diskutil)
    assert macos_runtime.validate_storage(str(models), "EXPECTED", 1) == str(models)
    assert list(models.iterdir()) == []
