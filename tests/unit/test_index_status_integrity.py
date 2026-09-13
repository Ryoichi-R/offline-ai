"""index status の overlay整合・cancelガード・reconcile・孤児tmp回収の回帰テスト（Phase 3, F-9）。"""

from contextlib import contextmanager
import json
import threading

import index_service
import search


def _writer_lock(available: bool):
    @contextmanager
    def _lock():
        yield available

    return _lock


# --- status() overlay 条件 ---------------------------------------------------


def test_status_does_not_overlay_persisted_cancelling_when_actual_is_ready(monkeypatch):
    """F-9: worker不在で実測readyのとき、永続cancellingで上書きしない。"""
    persisted = {
        "state": "cancelling",
        "job_id": "orphan-job",
        "generation": "g",
        "cancel_requested": True,
        "updated_at": "2026-09-06T17:36:47Z",
    }
    actual_ready = {
        "state": "ready",
        "embed_model": "bge-m3:latest",
        "generation": "g",
        "total": 3,
        "processed": 3,
        "generated": 3,
        "reused": 0,
        "failed": 0,
        "checkpointed": 0,
    }
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    monkeypatch.setattr(
        index_service, "get_embed_index_status", lambda _model, **_k: dict(actual_ready)
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")

    coordinator = index_service.IndexCoordinator()
    result = coordinator.status()

    assert result["state"] == "ready"
    assert result.get("cancel_requested") is not True


def test_status_overlays_persisted_when_actual_is_not_ready(monkeypatch):
    """実測がready以外（stale等）ならworker不在でも永続値を維持する（既存契約）。"""
    persisted = {
        "state": "failed",
        "job_id": "job-2",
        "generation": "g",
        "error_code": "EMBED_BATCH_FAILED",
    }
    actual_stale = {"state": "stale", "generation": "g", "total": 3}
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    monkeypatch.setattr(
        index_service, "get_embed_index_status", lambda _model, **_k: dict(actual_stale)
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")

    coordinator = index_service.IndexCoordinator()
    result = coordinator.status()

    assert result["state"] == "failed"
    assert result["error_code"] == "EMBED_BATCH_FAILED"


def test_status_overlays_when_worker_alive_even_if_actual_ready(monkeypatch):
    """既存契約の維持: workerが本当に生存中なら実測readyでも永続値をoverlayする。"""
    persisted = {"state": "building", "job_id": "job-3", "generation": "g"}
    actual_ready = {"state": "ready", "generation": "g", "total": 1}
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    monkeypatch.setattr(
        index_service, "get_embed_index_status", lambda _model, **_k: dict(actual_ready)
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")

    coordinator = index_service.IndexCoordinator()
    coordinator._job_id = "job-3"

    class _FakeWorker:
        def is_alive(self):
            return True

    coordinator._worker = _FakeWorker()
    result = coordinator.status()

    assert result["state"] == "building"


# --- cancel() の所有者不在ガード ---------------------------------------------


def test_cancel_does_not_write_cancelling_when_owner_absent(monkeypatch):
    """cancel(): worker不在かつwriter lock取得可のとき cancelling を書き込まない。"""
    persisted = {"state": "building", "job_id": "job-4", "generation": "g"}
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    saved = []

    def save(value):
        saved.append(dict(value))
        return True

    monkeypatch.setattr(index_service, "save_index_status", save)
    monkeypatch.setattr(
        index_service,
        "get_embed_index_status",
        lambda _model, **_k: {"state": "stale", "generation": "g"},
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")
    # writer lockが取得できる = 他プロセスも所有していない（所有者不在）。
    monkeypatch.setattr(index_service, "_embed_writer_lock", _writer_lock(True))

    coordinator = index_service.IndexCoordinator()
    result = coordinator.cancel("job-4")

    assert result["state"] != "cancelling"
    assert not any(s.get("state") == "cancelling" for s in saved)


def test_cancel_writes_cancelling_when_worker_alive(monkeypatch):
    """workerがこのプロセスで生存中ならガード対象外で従来どおりcancellingを書く。"""
    persisted = {"state": "building", "job_id": "job-5", "generation": "g"}
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    saved = []

    def save(value):
        saved.append(dict(value))
        return True

    monkeypatch.setattr(index_service, "save_index_status", save)
    monkeypatch.setattr(
        index_service,
        "get_embed_index_status",
        lambda _model, **_k: {"state": "building", "generation": "g"},
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")

    def fail_lock():
        raise AssertionError("workerが生存中はwriter lockを問い合わせないはず")

    monkeypatch.setattr(index_service, "_embed_writer_lock", fail_lock)

    coordinator = index_service.IndexCoordinator()
    coordinator._job_id = "job-5"

    class _FakeWorker:
        def is_alive(self):
            return True

    coordinator._worker = _FakeWorker()
    result = coordinator.cancel("job-5")

    assert result["state"] == "cancelling"
    assert result["cancel_requested"] is True


def test_real_status_file_parallel_persist_has_no_shared_tmp_race(tmp_path, monkeypatch):
    """20 concurrent status writes must not fail or leave a shared tmp file."""
    status_path = tmp_path / "embed_index_status.json"
    monkeypatch.setattr(search, "EMBED_STATUS_PATH", status_path)

    coordinator = index_service.IndexCoordinator()
    barrier = threading.Barrier(20)
    failures = []

    def write_status(processed):
        try:
            barrier.wait(timeout=5)
            coordinator._persist(
                {
                    "state": "building",
                    "job_id": "parallel-job",
                    "generation": "parallel-generation",
                    "processed": processed,
                    "total": 20,
                }
            )
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            failures.append(exc)

    threads = [threading.Thread(target=write_status, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    saved = json.loads(status_path.read_text(encoding="utf-8"))
    assert saved["state"] == "building"
    assert saved["job_id"] == "parallel-job"
    assert 0 <= saved["processed"] <= 19
    assert not list(tmp_path.glob("embed_index_status.json.tmp.*"))


def test_stale_worker_cannot_overwrite_cancelling_state(tmp_path, monkeypatch):
    status_path = tmp_path / "embed_index_status.json"
    monkeypatch.setattr(search, "EMBED_STATUS_PATH", status_path)

    coordinator = index_service.IndexCoordinator()
    coordinator._persist(
        {
            "state": "building",
            "job_id": "race-job",
            "generation": "race-generation",
            "processed": 3,
        }
    )
    coordinator._persist(
        {
            "state": "cancelling",
            "job_id": "race-job",
            "generation": "race-generation",
            "processed": 3,
            "cancel_requested": True,
        }
    )
    coordinator._persist(
        {
            "state": "building",
            "job_id": "race-job",
            "generation": "race-generation",
            "processed": 4,
        }
    )

    saved = json.loads(status_path.read_text(encoding="utf-8"))
    assert saved["state"] == "cancelling"
    assert saved["processed"] == 3


# --- reconcile_orphaned_state -------------------------------------------------


def test_reconcile_restores_orphaned_cancelling_to_actual_state(monkeypatch):
    persisted = {
        "state": "cancelling",
        "job_id": "orphan-1",
        "generation": "g",
        "cancel_requested": True,
    }
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    saved = []

    def save(value):
        saved.append(dict(value))
        return True

    monkeypatch.setattr(index_service, "save_index_status", save)
    monkeypatch.setattr(
        index_service,
        "get_embed_index_status",
        lambda _model, **_k: {
            "state": "ready",
            "generation": "g",
            "embed_model": "bge-m3:latest",
            "total": 5,
        },
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(index_service, "_embed_writer_lock", _writer_lock(True))
    monkeypatch.setattr(
        index_service.IndexCoordinator,
        "_reconcile_orphaned_tmp",
        staticmethod(lambda: (None, None)),
    )

    coordinator = index_service.IndexCoordinator()
    coordinator.reconcile_orphaned_state()

    assert saved
    final = saved[-1]
    assert final["state"] == "ready"
    assert final["cancel_requested"] is False


def test_reconcile_does_nothing_when_worker_alive(monkeypatch):
    persisted = {"state": "building", "job_id": "job-6", "generation": "g"}
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))

    def fail_lock():
        raise AssertionError("workerが生存中はwriter lockを問い合わせないはず")

    monkeypatch.setattr(index_service, "_embed_writer_lock", fail_lock)

    coordinator = index_service.IndexCoordinator()
    coordinator._job_id = "job-6"

    class _FakeWorker:
        def is_alive(self):
            return True

    coordinator._worker = _FakeWorker()
    coordinator.reconcile_orphaned_state()  # 例外が出なければOK（何もしない）


def test_reconcile_does_nothing_when_state_not_orphan_candidate(monkeypatch):
    persisted = {"state": "ready", "job_id": "job-7", "generation": "g"}
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))

    def fail_lock():
        raise AssertionError("readyはreconcile対象外")

    monkeypatch.setattr(index_service, "_embed_writer_lock", fail_lock)

    coordinator = index_service.IndexCoordinator()
    coordinator.reconcile_orphaned_state()


def test_reconcile_does_nothing_when_lock_unavailable(monkeypatch):
    """他プロセスが本当にworkerとして生存中（lock取得不可）なら何もしない。"""
    persisted = {"state": "building", "job_id": "job-8", "generation": "g"}
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    saved = []
    monkeypatch.setattr(
        index_service, "save_index_status", lambda value: saved.append(value) or True
    )
    monkeypatch.setattr(index_service, "_embed_writer_lock", _writer_lock(False))

    coordinator = index_service.IndexCoordinator()
    coordinator.reconcile_orphaned_state()

    assert not saved


# --- 孤児 .tmp 回収 ------------------------------------------------------------


def test_reconcile_removes_orphaned_tmp_when_cache_is_readable(tmp_path, monkeypatch):
    cache_path = tmp_path / "embed_cache.json"
    cache_path.write_text('{"version": 4, "entries": {}}', encoding="utf-8")
    tmp_file = tmp_path / "embed_cache.json.tmp"
    tmp_file.write_text("partial", encoding="utf-8")

    monkeypatch.setattr(index_service, "EMBED_CACHE_PATH", cache_path)

    code, cache = index_service.IndexCoordinator._reconcile_orphaned_tmp()

    assert code is None
    assert cache == {"version": 4, "entries": {}}
    assert not tmp_file.exists()
    assert cache_path.exists()


def test_reconcile_keeps_orphaned_tmp_when_cache_is_missing(tmp_path, monkeypatch):
    """embed_cache.json 自体が無い場合は削除せず復旧材料を残す。"""
    cache_path = tmp_path / "embed_cache.json"
    tmp_file = tmp_path / "embed_cache.json.tmp"
    tmp_file.write_text("partial", encoding="utf-8")

    monkeypatch.setattr(index_service, "EMBED_CACHE_PATH", cache_path)

    code, cache = index_service.IndexCoordinator._reconcile_orphaned_tmp()

    assert code == "EMBED_CACHE_UNREADABLE"
    assert cache is None
    assert tmp_file.exists()


def test_reconcile_keeps_orphaned_tmp_when_cache_is_corrupt(tmp_path, monkeypatch):
    """破損cacheでは .tmp を消さない。

    ``load_embed_cache()`` は破損時も空キャッシュ（``entries`` が空 dict）を
    返す fail-open な契約なので、可読性判定にそれを使うと破損を「読めた」と
    誤判定して復旧材料を消してしまう。実ファイルを直接 parse することの回帰
    テストとして、モックを挟まず本物の破損ファイルで検証する。
    """
    cache_path = tmp_path / "embed_cache.json"
    cache_path.write_text("{ not valid json", encoding="utf-8")
    tmp_file = tmp_path / "embed_cache.json.tmp"
    tmp_file.write_text("partial", encoding="utf-8")

    monkeypatch.setattr(index_service, "EMBED_CACHE_PATH", cache_path)

    code, cache = index_service.IndexCoordinator._reconcile_orphaned_tmp()

    assert code == "EMBED_CACHE_UNREADABLE"
    assert cache is None
    assert tmp_file.exists()
    assert cache_path.read_text(encoding="utf-8") == "{ not valid json"


def test_reconcile_keeps_orphaned_tmp_when_cache_has_no_entries(tmp_path, monkeypatch):
    """JSONとして読めても entries を持たない構造なら復旧材料を残す。"""
    cache_path = tmp_path / "embed_cache.json"
    cache_path.write_text('{"version": 4}', encoding="utf-8")
    tmp_file = tmp_path / "embed_cache.json.tmp"
    tmp_file.write_text("partial", encoding="utf-8")

    monkeypatch.setattr(index_service, "EMBED_CACHE_PATH", cache_path)

    code, cache = index_service.IndexCoordinator._reconcile_orphaned_tmp()

    assert code == "EMBED_CACHE_UNREADABLE"
    assert cache is None
    assert tmp_file.exists()


def test_reconcile_tmp_noop_when_no_tmp_file(tmp_path, monkeypatch):
    cache_path = tmp_path / "embed_cache.json"
    cache_path.write_text('{"version": 4, "entries": {}}', encoding="utf-8")
    monkeypatch.setattr(index_service, "EMBED_CACHE_PATH", cache_path)

    assert index_service.IndexCoordinator._reconcile_orphaned_tmp() == (None, None)


def test_reconcile_reuses_parsed_cache_for_status_snapshot(tmp_path, monkeypatch):
    """孤児 .tmp の可読性判定で parse 済みの cache を status 判定へ渡す。

    441 MB 級の embed_cache.json を起動時に二度 parse しないための契約。
    """
    cache_path = tmp_path / "embed_cache.json"
    cache_path.write_text('{"version": 4, "entries": {}}', encoding="utf-8")
    tmp_file = tmp_path / "embed_cache.json.tmp"
    tmp_file.write_text("partial", encoding="utf-8")

    persisted = {"state": "building", "job_id": "orphan-reuse", "generation": "g"}
    seen = {}

    monkeypatch.setattr(index_service, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    monkeypatch.setattr(index_service, "save_index_status", lambda _value: True)
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(index_service, "_embed_writer_lock", _writer_lock(True))

    def fake_status(_model, **kwargs):
        seen.update(kwargs)
        return {"state": "ready", "generation": "g", "total": 0}

    monkeypatch.setattr(index_service, "get_embed_index_status", fake_status)

    index_service.IndexCoordinator().reconcile_orphaned_state()

    assert seen["cache"] == {"version": 4, "entries": {}}
    assert seen["validate_entries"] is False


def test_reconcile_persists_error_code_when_orphaned_tmp_is_kept(tmp_path, monkeypatch):
    """孤児 .tmp を残した理由が status の error_code として残る。"""
    cache_path = tmp_path / "embed_cache.json"
    cache_path.write_text("{ not valid json", encoding="utf-8")
    tmp_file = tmp_path / "embed_cache.json.tmp"
    tmp_file.write_text("partial", encoding="utf-8")

    persisted = {
        "state": "building",
        "job_id": "orphan-tmp",
        "generation": "g",
        "cancel_requested": True,
    }
    saved = []

    monkeypatch.setattr(index_service, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    monkeypatch.setattr(
        index_service, "save_index_status", lambda value: saved.append(dict(value)) or True
    )
    monkeypatch.setattr(
        index_service,
        "get_embed_index_status",
        lambda _model, **_k: {"state": "stale", "generation": "g", "total": 0},
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(index_service, "_embed_writer_lock", _writer_lock(True))

    index_service.IndexCoordinator().reconcile_orphaned_state()

    assert saved
    final = saved[-1]
    assert final["error_code"] == "EMBED_CACHE_UNREADABLE"
    assert final["state"] == "stale"
    assert final["cancel_requested"] is False
    assert tmp_file.exists()
