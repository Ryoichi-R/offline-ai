"""Authenticated folder settings, active-root isolation and next-process readers."""

import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import source_folder
import source_view
import web_server


@pytest.fixture
def server(tmp_path):
    active = tmp_path / "old"
    active.mkdir()
    app = tmp_path / "app"
    instance = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token="synthetic-session",
        source_settings_root=app,
        source_root=active,
        evidence_registry=source_view.EvidenceRegistry(active),
    )
    instance.bind_port = instance.server_address[1]
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    yield instance, app, active
    instance.shutdown_jobs_once()
    instance.shutdown()
    worker.join(timeout=5)
    instance.server_close()


def request(server, method, path="/api/source-folder", payload=None, *, auth=True, origin=None):
    headers = {"Host": f"127.0.0.1:{server.bind_port}"}
    if auth:
        headers["Cookie"] = "offlineai_session=synthetic-session"
    if origin:
        headers["Origin"] = origin
    body = None if payload is None else json.dumps(payload)
    if body:
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    value = json.loads(response.read())
    status = response.status
    connection.close()
    return status, value


def test_save_persists_new_folder_and_leaves_current_registry_and_root_intact(server, tmp_path):
    instance, app, active = server
    new = tmp_path / "Google Drive 資料"
    new.mkdir()
    status, info = request(instance, "POST", payload={"path": str(new)})
    assert status == 200 and info["restartRequired"]
    assert info["activePath"] == str(active) and info["savedPath"] == str(new)
    assert instance.source_root == active and instance.evidence_registry.source_root == active
    assert source_folder.load_source_root(app) == new
    assert request(instance, "GET")[1] == info
    assert list(new.iterdir()) == []


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/source-folder"),
        ("POST", "/api/source-folder"),
        ("POST", "/api/source-folder/pick"),
    ],
)
def test_settings_and_dialog_require_session_and_origin(server, method, path):
    instance, app, _ = server
    assert request(instance, method, path, {}, auth=False)[0] == 403
    assert request(instance, method, path, {}, origin="https://foreign.invalid")[0] == 403
    assert not source_folder.settings_path(app).exists()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"path": "relative"},
        {"path": None},
        {"path": "/missing/folder"},
        {"path": "x", "command": "ignored"},
        [],
    ],
)
def test_invalid_selection_does_not_change_persisted_settings(server, payload):
    instance, app, _ = server
    assert request(instance, "POST", payload=payload)[0] == 400
    assert not source_folder.settings_path(app).exists()


def test_native_picker_cancel_error_and_single_dialog_guard(server, monkeypatch):
    instance, _, _ = server
    monkeypatch.setattr(web_server, "pick_source_folder", lambda: None)
    assert request(instance, "POST", "/api/source-folder/pick", {}) == (
        200,
        {"path": None, "cancelled": True},
    )
    instance._source_picker_lock.acquire()
    try:
        assert request(instance, "POST", "/api/source-folder/pick", {})[0] == 503
    finally:
        instance._source_picker_lock.release()

    def fail():
        raise source_folder.SourceFolderError("選択できません")

    monkeypatch.setattr(web_server, "pick_source_folder", fail)
    assert request(instance, "POST", "/api/source-folder/pick", {})[0] == 400
    assert request(instance, "POST", "/api/source-folder/pick", {"command": "no"})[0] == 400
    assert not instance._source_picker_lock.locked()


def test_save_io_error_is_reported_without_changing_current_root(server, monkeypatch):
    instance, _, active = server

    def fail(*args):
        raise PermissionError()

    monkeypatch.setattr(web_server, "save_source_root", fail)
    status, value = request(instance, "POST", payload={"path": str(active)})
    assert status == 500 and value["error"]["code"] == "source_settings_unavailable"
    assert instance.source_root == active


def test_next_process_uses_selected_root_for_search_index_and_evidence(tmp_path):
    app = tmp_path / "app"
    docs = tmp_path / "selected"
    docs.mkdir()
    (docs / "synthetic.md").write_text("# Example\nSynthetic reference text\n")
    source_folder.save_source_root(app, str(docs))
    internal = Path(web_server.__file__).parent
    code = """
from pathlib import Path
import json,sys
sys.path.insert(0,sys.argv[1])
import source_folder
source_folder.settings_path=lambda _:Path(sys.argv[2])
import search,index_service,web_services
root=search.SKILL_SOURCE_DIR
print(json.dumps(dict(search=str(root),index=str(index_service.SKILL_SOURCE_DIR),
 evidence=str(web_services.create_evidence_registry().source_root),
 files=[rel for _,rel in search.iter_source_files()],
 chunks=len(search.build_source_snapshot(root)[0]))))
"""
    done = subprocess.run(
        [sys.executable, "-B", "-c", code, str(internal), str(source_folder.settings_path(app))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr
    value = json.loads(done.stdout)
    assert value["search"] == value["index"] == value["evidence"] == str(docs)
    assert value["files"] == ["synthetic.md"] and value["chunks"] > 0
