"""Index coordinator lifecycle and redacted control state tests."""

from contextlib import contextmanager

import pytest

import index_service
import search


def test_start_returns_existing_durable_job_without_second_worker(monkeypatch):
    persisted = {
        "state": "building",
        "job_id": "existing-job",
        "generation": "g",
        "embed_model": "bge-m3:latest",
        "total": 10,
        "processed": 2,
    }
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    monkeypatch.setattr(
        index_service,
        "get_embed_index_status",
        lambda _model, **_kwargs: dict(persisted),
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")

    coordinator = index_service.IndexCoordinator()
    result = coordinator.start()

    assert result["job_id"] == "existing-job"
    assert coordinator._worker is None


def test_cancel_is_durable_and_publishes_cancelling(monkeypatch):
    status = {
        "state": "building",
        "job_id": "job-1",
        "generation": "g",
        "embed_model": "bge-m3:latest",
    }
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(status))
    saved = []

    def save(value):
        saved.append(dict(value))
        status.update(value)
        return True

    monkeypatch.setattr(index_service, "save_index_status", save)
    monkeypatch.setattr(
        index_service,
        "get_embed_index_status",
        lambda _model, **_kwargs: dict(status),
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")

    # worker はこのプロセス内に存在しないが、他プロセス（実際の owner）が
    # writer lock を保持している状況を模擬する。所有者が実在するため
    # cancel() は cancelling への遷移を続行してよい（所有者不在ガードの対象外）。
    @contextmanager
    def unavailable_writer_lock():
        yield False

    monkeypatch.setattr(index_service, "_embed_writer_lock", unavailable_writer_lock)

    coordinator = index_service.IndexCoordinator()
    subscriber = coordinator.subscribe("job-1")
    result = coordinator.cancel("job-1")

    assert result["state"] == "cancelling"
    assert result["cancel_requested"] is True
    event = subscriber.get_nowait()
    while event["state"] != "cancelling":
        event = subscriber.get_nowait()
    assert event["type"] == "index_progress"
    assert event["state"] == "cancelling"
    assert "path" not in event
    assert saved[-1]["job_id"] == "job-1"


def test_resume_restarts_stale_durable_job_after_process_restart(monkeypatch):
    persisted = {
        "state": "building",
        "job_id": "stale-job",
        "generation": "g",
        "embed_model": "bge-m3:latest",
        "total": 10,
        "processed": 2,
    }
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    monkeypatch.setattr(
        index_service,
        "get_embed_index_status",
        lambda _model, **_kwargs: dict(persisted),
    )
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(index_service, "build_source_snapshot", lambda: ([], {}))
    monkeypatch.setattr(index_service, "compute_embed_generation", lambda *_args: "g")
    monkeypatch.setattr(index_service, "save_index_status", lambda value: True)

    @contextmanager
    def available_writer_lock():
        yield True

    monkeypatch.setattr(index_service, "_embed_writer_lock", available_writer_lock)
    started = []
    monkeypatch.setattr(
        index_service.IndexCoordinator,
        "_run_worker",
        lambda _self, job_id, *args, **kwargs: started.append(job_id),
    )

    coordinator = index_service.IndexCoordinator()
    result = coordinator.start(resume=True)
    if coordinator._worker is not None:
        coordinator._worker.join(timeout=1)

    assert result["job_id"] == "stale-job"
    assert started == ["stale-job"]


def test_start_rejects_generation_changed_after_preview(monkeypatch):
    persisted = {"state": "ready", "generation": "old-generation"}
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(index_service, "build_source_snapshot", lambda: ([], {}))
    monkeypatch.setattr(
        index_service,
        "get_embed_model_identity",
        lambda model: {"name": "bge-m3:latest", "digest": "fixture-digest"},
    )
    monkeypatch.setattr(index_service, "compute_embed_generation", lambda *_args: "new-generation")

    coordinator = index_service.IndexCoordinator()
    with pytest.raises(index_service.IndexGenerationChangedError):
        coordinator.start(expected_generation="old-generation")


def test_resume_rejects_mode_changed_from_saved_job(monkeypatch):
    persisted = {
        "state": "paused",
        "job_id": "saved-job",
        "mode": "incremental",
        "generation": "g",
    }
    monkeypatch.setattr(index_service, "load_index_status", lambda: dict(persisted))
    coordinator = index_service.IndexCoordinator()
    with pytest.raises(index_service.IndexGenerationChangedError):
        coordinator.start(resume=True, mode="full")


@pytest.mark.parametrize(
    "failure, expected_state, expected_code",
    [
        (
            search.EmbeddingBatchError("EMBED_HTTP_500", "fixture", retryable=False),
            "failed",
            "EMBED_HTTP_500",
        ),
        (RuntimeError("fixture"), "failed", "INDEX_INTERNAL_ERROR"),
        (index_service.IndexCancelled("fixture"), "cancelled", None),
    ],
)
def test_interrupted_progress_does_not_hide_worker_failure(
    monkeypatch, tmp_path, failure, expected_state, expected_code
):
    monkeypatch.setattr(search, "EMBED_STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(index_service, "build_source_chunks", lambda: [{}] * 10)
    monkeypatch.setattr(index_service, "compute_embed_generation", lambda *_: "fixture-generation")

    def interrupt(*args, emit_progress, **kwargs):
        emit_progress({"phase": "interrupted", "processed": 7, "total": 10, "checkpointed": 7})
        raise failure

    monkeypatch.setattr(index_service, "build_or_update_embed_index", interrupt)
    coordinator = index_service.IndexCoordinator()
    coordinator._run_worker("fixture-job")
    status = search.load_index_status()
    assert status["state"] == expected_state
    assert status.get("error_code") == expected_code
    assert status["checkpointed"] == 7
    assert status["cancel_requested"] is (expected_state == "cancelled")


@pytest.mark.parametrize(
    "failure",
    [
        index_service.IndexCancelled("fixture"),
        search.EmbeddingBatchError("EMBED_HTTP_500", "fixture", retryable=False),
        RuntimeError("fixture"),
    ],
)
def test_cancelled_worker_preserves_last_checkpoint_progress(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(search, "EMBED_STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(index_service, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(index_service, "build_source_chunks", lambda: [{}] * 10)
    monkeypatch.setattr(index_service, "compute_embed_generation", lambda *_: "fixture-generation")
    coordinator = index_service.IndexCoordinator()

    def interrupt(*args, emit_progress, **kwargs):
        coordinator._cancel.set()
        coordinator._persist(
            {**search.load_index_status(), "state": "cancelling", "cancel_requested": True}
        )
        emit_progress({"phase": "interrupted", "processed": 7, "total": 10, "checkpointed": 7})
        raise failure

    monkeypatch.setattr(index_service, "build_or_update_embed_index", interrupt)
    coordinator._run_worker("fixture-job")
    status = search.load_index_status()
    assert status["state"] == "cancelled"
    assert status["cancel_requested"] is True
    assert status["processed"] == 7
    assert status["checkpointed"] == 7
