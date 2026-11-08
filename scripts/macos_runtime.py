"""Mac local-runtime checks for offline-ai. No model downloads or user data access."""

from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
OLLAMA_API = "http://127.0.0.1:11434/api/tags"
CONFIG_KEYS = {
    "models_directory",
    "volume_uuid",
    "ollama_executable",
    "python_executable",
    "ollama_version",
    "minimum_free_bytes",
}


def load_runtime_config(path):
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Mac runtime configを読み込めません") from exc
    if not isinstance(data, dict) or data.get("schemaVersion") != 1:
        raise ValueError("Mac runtime config schemaが不正です")
    required = {
        "models_directory": data.get("modelsDirectory"),
        "volume_uuid": data.get("volumeUuid"),
        "ollama_executable": data.get("ollamaExecutable"),
        "python_executable": data.get("pythonExecutable"),
        "ollama_version": data.get("ollamaVersion"),
        "minimum_free_bytes": data.get("minimumFreeBytes"),
    }
    string_values = {
        key: value for key, value in required.items() if key != "minimum_free_bytes"
    }
    if any(
        not isinstance(value, str) or not value.strip()
        for value in string_values.values()
    ):
        raise ValueError("Mac runtime configに必須項目がありません")
    minimum_free = required["minimum_free_bytes"]
    if type(minimum_free) is not int or minimum_free < 1:
        raise ValueError("Mac runtime configの最低空き容量が不正です")
    required["minimum_free_bytes"] = minimum_free
    return required


def validate_mount_identity(models_directory, expected_uuid, info):
    """Check path containment, mounted-volume UUID, and the expected exFAT type."""
    models = Path(models_directory).resolve(strict=True)
    if not models.is_dir():
        raise ValueError("model directory is not a directory")
    mount_value = info.get("MountPoint")
    actual_uuid = str(info.get("VolumeUUID", "")).strip().upper()
    filesystem = (
        str(info.get("FilesystemName") or info.get("FilesystemType") or "")
        .strip()
        .lower()
    )
    if not mount_value or not actual_uuid:
        raise ValueError("diskutil did not return mounted volume identity")
    mount = Path(mount_value).resolve(strict=True)
    try:
        relative = models.relative_to(mount)
    except ValueError as exc:
        raise ValueError(
            "model directory is not inside the requested mounted volume"
        ) from exc
    if not relative.parts:
        raise ValueError("model directory must be below the volume mount point")
    if actual_uuid != str(expected_uuid).strip().upper():
        raise ValueError("mounted volume UUID does not match the configured volume")
    if filesystem not in {"exfat", "exfat (ms-dos)"}:
        raise ValueError("model volume is not exFAT")
    if models.stat().st_dev != mount.stat().st_dev:
        raise ValueError("model directory is not on the mounted volume device")
    return models, mount


def validate_free_space(models_directory, minimum_free_bytes, *, free_bytes=None):
    available = (
        shutil.disk_usage(models_directory).free if free_bytes is None else free_bytes
    )
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 1:
        raise ValueError("minimum free space must be a positive integer")
    if type(available) is not int or available < minimum_free_bytes:
        raise ValueError("model volume has less than the configured minimum free space")
    return available


def validate_storage(models_directory, expected_uuid, minimum_free_bytes):
    if sys.platform != "darwin":
        raise ValueError("Mac runtime entrypoint requires macOS")
    models = Path(models_directory).expanduser()
    if not models.is_absolute():
        raise ValueError("model directory must be an absolute path")
    if not models.is_dir():
        raise ValueError("model directory is unavailable; check the external volume")
    models = models.resolve(strict=True)
    # diskutil accepts a mounted volume, not an arbitrary directory below it.
    mount = models
    while not os.path.ismount(mount) and mount != mount.parent:
        mount = mount.parent
    try:
        completed = subprocess.run(
            ["diskutil", "info", "-plist", str(mount)],
            check=True,
            capture_output=True,
            timeout=15,
        )
        info = plistlib.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, plistlib.InvalidFileException) as exc:
        raise ValueError(
            "model volume identity could not be verified with diskutil"
        ) from exc
    resolved, _ = validate_mount_identity(models, expected_uuid, info)
    validate_free_space(resolved, minimum_free_bytes)
    probe_path = None
    try:
        fd, probe_path = tempfile.mkstemp(
            prefix=".offline-ai-write-check-", dir=str(resolved)
        )
        os.write(fd, b"offline-ai storage probe")
        os.fsync(fd)
        os.close(fd)
        fd = None
    except OSError as exc:
        raise ValueError("model directory is not writable") from exc
    finally:
        if "fd" in locals() and fd is not None:
            os.close(fd)
        if probe_path:
            try:
                os.unlink(probe_path)
            except OSError:
                pass
    return str(resolved)


def _state_directory():
    root = Path(tempfile.gettempdir()) / "offline-ai-runtime"
    if root.exists() and root.is_symlink():
        raise ValueError("runtime state directory must not be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat()
    if info.st_uid != os.getuid() or not root.is_dir():
        raise ValueError("runtime state directory owner is invalid")
    os.chmod(str(root), 0o700)
    return root


def _owner_path():
    return _state_directory() / "ollama-owner.json"


def _process_value(pid, field):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", field + "="],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _process_snapshot(pid):
    return _process_value(pid, "lstart"), _process_value(pid, "command")


def owner_record_matches(
    record,
    models_directory,
    volume_uuid,
    ollama_executable,
    process_start,
    process_command,
):
    expected_models = os.path.realpath(str(models_directory))
    expected_binary = os.path.realpath(str(ollama_executable))
    if not isinstance(record, dict):
        return False
    if record.get("modelsDirectory") != expected_models:
        return False
    if str(record.get("volumeUuid", "")).upper() != str(volume_uuid).upper():
        return False
    if record.get("ollamaExecutable") != expected_binary:
        return False
    if record.get("processStart") != process_start or not process_start:
        return False
    command = str(process_command or "")
    return (
        expected_binary in command
        and re.search(r"(?:^|\s)serve(?:\s|$)", command) is not None
    )


def _read_owner():
    path = _owner_path()
    if path.is_symlink():
        raise ValueError("runtime ownership record must not be a symlink")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime ownership record is invalid") from exc


def _write_owner(args):
    binary = os.path.realpath(args.ollama_executable)
    pid = int(args.pid)
    start = ""
    command = ""
    for _ in range(20):
        start, command = _process_snapshot(pid)
        if start and binary in command and re.search(r"(?:^|\s)serve(?:\s|$)", command):
            break
        time.sleep(0.1)
    if not owner_record_matches(
        {
            "modelsDirectory": os.path.realpath(args.models_directory),
            "volumeUuid": args.volume_uuid,
            "ollamaExecutable": binary,
            "processStart": start,
        },
        args.models_directory,
        args.volume_uuid,
        binary,
        start,
        command,
    ):
        raise ValueError("started Ollama process could not be identified")
    record = {
        "schemaVersion": 1,
        "pid": pid,
        "processStart": start,
        "modelsDirectory": os.path.realpath(args.models_directory),
        "volumeUuid": args.volume_uuid.upper(),
        "ollamaExecutable": binary,
    }
    path = _owner_path()
    existing = _read_owner()
    if existing and int(existing.get("pid", -1)) != pid:
        old_start, old_command = _process_snapshot(existing.get("pid", -1))
        if owner_record_matches(
            existing,
            args.models_directory,
            args.volume_uuid,
            args.ollama_executable,
            old_start,
            old_command,
        ):
            raise ValueError("a different owned Ollama process is already recorded")
    fd, temporary = tempfile.mkstemp(
        prefix="ollama-owner-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _owner_matches(args):
    record = _read_owner()
    if not record:
        return False
    try:
        pid = int(record.get("pid"))
        os.kill(pid, 0)
    except (TypeError, ValueError, OSError):
        return False
    start, command = _process_snapshot(pid)
    return owner_record_matches(
        record,
        args.models_directory,
        args.volume_uuid,
        args.ollama_executable,
        start,
        command,
    )


def _remove_owner(pid):
    path = _owner_path()
    record = _read_owner()
    if record and int(record.get("pid", -1)) == int(pid):
        path.unlink()


def server_ready(timeout=2.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(OLLAMA_API, method="GET")
    try:
        with opener.open(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.request.URLError, TimeoutError):
        return False


def verify_model(model):
    sys.path.insert(0, str(REPO_ROOT / "_internal"))
    from search import get_model_identity

    return get_model_identity(model)


def main(argv=None):
    parser = argparse.ArgumentParser(description="offline-ai Mac runtime checks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_parser = subparsers.add_parser("config-value")
    config_parser.add_argument("--path", required=True)
    config_parser.add_argument("--key", required=True, choices=sorted(CONFIG_KEYS))
    storage_parser = subparsers.add_parser("validate-storage")
    storage_parser.add_argument("--models-directory", required=True)
    storage_parser.add_argument("--volume-uuid", required=True)
    storage_parser.add_argument("--minimum-free-bytes", required=True, type=int)
    resolve_parser = subparsers.add_parser("resolve-model")
    resolve_parser.add_argument("--model", default=None)
    resolve_parser.add_argument("--model-file", required=True)
    model_parser = subparsers.add_parser("verify-model")
    model_parser.add_argument("--model", required=True)
    subparsers.add_parser("server-ready")
    owner_parser = subparsers.add_parser("owner-matches")
    owner_parser.add_argument("--models-directory", required=True)
    owner_parser.add_argument("--volume-uuid", required=True)
    owner_parser.add_argument("--ollama-executable", required=True)
    write_parser = subparsers.add_parser("write-owner")
    write_parser.add_argument("--pid", required=True, type=int)
    write_parser.add_argument("--models-directory", required=True)
    write_parser.add_argument("--volume-uuid", required=True)
    write_parser.add_argument("--ollama-executable", required=True)
    remove_parser = subparsers.add_parser("remove-owner")
    remove_parser.add_argument("--pid", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "config-value":
            print(load_runtime_config(args.path)[args.key])
        elif args.command == "validate-storage":
            validate_storage(
                args.models_directory, args.volume_uuid, args.minimum_free_bytes
            )
        elif args.command == "resolve-model":
            sys.path.insert(0, str(REPO_ROOT / "_internal"))
            from model_config import resolve_model_selection

            selection = resolve_model_selection(args.model, Path(args.model_file))
            print(selection.model)
        elif args.command == "verify-model":
            identity = verify_model(args.model)
            print(identity["digest"])
        elif args.command == "server-ready":
            return 0 if server_ready() else 1
        elif args.command == "owner-matches":
            return 0 if _owner_matches(args) else 1
        elif args.command == "write-owner":
            _write_owner(args)
        elif args.command == "remove-owner":
            _remove_owner(args.pid)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Mac runtime check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
