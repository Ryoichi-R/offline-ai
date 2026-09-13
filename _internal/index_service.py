"""Durable, single-writer Embedding index coordinator.

The coordinator is intentionally small and standard-library-only.  The
process-wide writer lock remains in ``search.py``; this module owns the
background lifecycle, redacted status file, cancellation request, and bounded
event fan-out used by the CLI and Web server.
"""

from __future__ import annotations

from collections import deque
import json
import queue
import re
import threading
import time
import uuid

from search import (
    EMBED_CACHE_PATH,
    INDEX_MAX_SECONDS,
    EmbedBuildError,
    EmbeddingBatchError,
    EmbedIndexBusyError,
    SourceSnapshotError,
    build_or_update_embed_index,
    build_source_chunks,  # noqa: F401 - retained as a monkeypatch seam for legacy callers/tests
    build_source_snapshot,
    compute_embed_generation,
    detect_embed_model,
    get_embed_model_identity,
    get_embed_index_status,
    load_index_status,
    plan_embed_index_update,
    save_index_status,
    SKILL_SOURCE_DIR,
    IndexStatusConflictError,
    _generated_rate_record,
    _validate_embed_index_mode,
    _embed_writer_lock,
)


class IndexCancelled(RuntimeError):
    """The owner requested cancellation before the next batch."""


class IndexGenerationChangedError(RuntimeError):
    """The source changed after the confirmation/preview snapshot."""


class IndexModeConflictError(IndexGenerationChangedError):
    """The requested mode differs from the active or saved job."""


# 少量の生成ではモデル起動待ちが速度を支配し、次回見積もりを大きく歪める。
# この件数未満のjobでは、既存の同一モデル実績を上書きしない（実績が無ければ採用する）。
RATE_MIN_GENERATED_SAMPLE = 32


class IndexCoordinator:
    """One in-process background worker backed by a cross-process status file."""

    EVENT_BUFFER_SIZE = 64

    def __init__(self):
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self._job_id = ""
        self._events: deque[dict] = deque(maxlen=self.EVENT_BUFFER_SIZE)
        self._subscribers: list[queue.Queue] = []
        self._shutdown = False

    @staticmethod
    def _job_status_event(status: dict) -> dict:
        event = {"type": "index_progress"}
        event.update(
            {
                key: status.get(key)
                for key in (
                    "state",
                    "job_id",
                "generation",
                "embed_model",
                "mode",
                "total",
                    "processed",
                    "generated",
                    "reused",
                    "failed",
                    "checkpointed",
                    "elapsed_seconds",
                "rate_per_second",
                "rate_basis",
                "eta_seconds",
                    "error_code",
                    "cancel_requested",
                    "updated_at",
                )
                if key in status
            }
        )
        return event

    def _publish(self, status: dict) -> dict:
        event = self._job_status_event(status)
        with self._lock:
            self._events.append(event)
            for subscriber in list(self._subscribers):
                subscriber.put(event)
        return event

    def _persist(self, status: dict) -> dict:
        status = dict(status)
        allow_restart = bool(status.pop("_allow_restart", False))
        allow_generation_change = bool(status.pop("_allow_generation_change", False))
        status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["_status_guard"] = {
            "allow_restart": allow_restart,
            "allow_generation_change": allow_generation_change,
            "job_id": status.get("job_id"),
            "generation": status.get("generation"),
        }
        try:
            saved = save_index_status(status)
        except IndexStatusConflictError:
            # cancel/terminal stateが先に確定した場合、遅いworkerの古い
            # progress/final stateは捨て、確定済みstatusをそのまま返す。
            current = load_index_status()
            if current:
                return current
            raise
        if not saved:
            raise EmbedBuildError("INDEX_STATUS_PERSIST_FAILED", "index status save failed")
        status.pop("_status_guard", None)
        self._publish(status)
        return status

    def status(self) -> dict:
        """Return a redacted snapshot and never start a worker.

        ``validate_entries=False`` で全件検証を省略し、header 一致＋key集合
        一致までで ready 判定する（build 完了直後の全件検証は ``_run_worker``
        / ``run_blocking`` が別途行うため、ここでは省略してよい）。
        """
        with self._lock:
            local_job = self._job_id
            worker_alive = self._worker is not None and self._worker.is_alive()
        model = detect_embed_model()
        snapshot = get_embed_index_status(model, validate_entries=False)
        persisted = load_index_status()
        if worker_alive and persisted.get("job_id") == local_job:
            snapshot = {**snapshot, **persisted}
        elif (
            snapshot.get("state") != "ready"
            and persisted.get("generation") == snapshot.get("generation")
            and persisted.get("state")
            in {"building", "cancelling", "paused", "cancelled", "failed"}
        ):
            # worker不在で実測がready以外のときだけ永続値をoverlayする。
            # worker不在かつ実測readyなら、所有者不在のまま固着した
            # building/cancelling を実測へ委ねる（F-9対応）。
            snapshot = {**snapshot, **persisted}
        return snapshot

    def plan(self, *, mode: str = "incremental") -> dict:
        """Return a side-effect-free update preview for the current source."""
        effective_mode = _validate_embed_index_mode(mode)
        model = detect_embed_model()
        if not model:
            raise EmbedBuildError(
                "EMBED_MODEL_NOT_CONFIGURED", "embedding model is not configured"
            )
        chunks, manifest = build_source_snapshot()
        identity = get_embed_model_identity(model)
        return plan_embed_index_update(
            model,
            chunks,
            source_manifest=manifest,
            mode=effective_mode,
            model_identity=identity,
        )

    def _cancel_check(self, job_id: str, started: float) -> None:
        if self._cancel.is_set():
            raise IndexCancelled("cancelled")
        persisted = load_index_status()
        if (
            persisted.get("job_id") == job_id
            and persisted.get("cancel_requested") is True
        ):
            self._cancel.set()
            raise IndexCancelled("cancelled")
        if time.monotonic() - started >= INDEX_MAX_SECONDS:
            raise EmbedBuildError(
                "INDEX_DEADLINE_EXCEEDED", "index build exceeded its fail-safe deadline"
            )

    @staticmethod
    def _progress_rate_fields(
        progress: dict, total: int, started: float, previous_rate: dict
    ) -> dict:
        """実Embedding生成時間だけから速度・残り時間を出す。

        再利用・checkpoint件数、資料走査、cache読込/保存の時間は速度へ混ぜない。
        生成件数が少ない間は、同一モデルの既存実績（あれば）を維持する。
        """
        elapsed = max(0.001, time.monotonic() - started)
        processed = int(progress.get("processed", 0))
        generated = int(progress.get("generated", 0))
        try:
            embedding_seconds = float(progress.get("embedding_seconds") or 0.0)
        except (TypeError, ValueError):
            embedding_seconds = 0.0
        rate = previous_rate.get("rate_per_second")
        basis = previous_rate.get("rate_basis")
        if generated > 0 and embedding_seconds > 0 and (
            rate is None or generated >= RATE_MIN_GENERATED_SAMPLE
        ):
            rate = generated / embedding_seconds
            basis = "generated"
        eta = (
            max(0.0, (total - processed) / rate)
            if rate and processed < total
            else None
        )
        return {
            "elapsed_seconds": round(elapsed, 3),
            "rate_per_second": round(rate, 3) if rate else None,
            "rate_basis": basis if rate else None,
            "eta_seconds": round(eta, 3) if eta is not None else None,
        }

    @staticmethod
    def _job_result_fields(progress: dict) -> dict:
        return {
            key: progress[key]
            for key in ("processed", "generated", "reused", "failed", "checkpointed")
            if key in progress
        }

    def _run_worker(
        self,
        job_id: str,
        mode: str = "incremental",
        prepared: tuple | None = None,
        previous_rate: dict | None = None,
    ) -> None:
        started = time.monotonic()
        try:
            mode = _validate_embed_index_mode(mode)
            model = detect_embed_model()
            if not model:
                self._persist(
                    {
                        "state": "failed",
                        "job_id": job_id,
                        "error_code": "EMBED_MODEL_NOT_CONFIGURED",
                    }
                )
                return
            if prepared is None:
                chunks, manifest = build_source_snapshot()
                identity = get_embed_model_identity(model)
                generation = compute_embed_generation(model, chunks)
            else:
                chunks, manifest, identity, generation = prepared
            if previous_rate is None:
                previous_rate = _generated_rate_record(load_index_status(), model)
            last_progress: dict = {}
            self._persist(
                {
                    "state": "building",
                    "job_id": job_id,
                    "_allow_restart": True,
                    "_allow_generation_change": True,
                    "generation": generation,
                    "embed_model": model,
                    "mode": mode,
                    "total": len(chunks),
                    "processed": 0,
                    "generated": 0,
                    "reused": 0,
                    "failed": 0,
                    "checkpointed": 0,
                    **previous_rate,
                    "cancel_requested": False,
                }
            )

            def emit(progress: dict) -> None:
                last_progress.clear()
                last_progress.update(progress)
                self._persist(
                    {
                        **progress,
                        # An interruption alone does not imply cancellation.
                        # Preserve explicit cancellation (and its last progress),
                        # otherwise let the exception handler classify failure.
                        "state": "cancelled"
                        if progress.get("phase") == "interrupted" and self._cancel.is_set()
                        else "building",
                        "job_id": job_id,
                        "generation": generation,
                        "embed_model": model,
                        "mode": mode,
                        **self._progress_rate_fields(
                            progress, len(chunks), started, previous_rate
                        ),
                        "cancel_requested": self._cancel.is_set(),
                    }
                )

            build_or_update_embed_index(
                model,
                chunks,
                source_manifest=manifest,
                source_root=SKILL_SOURCE_DIR,
                mode=mode,
                model_identity=identity,
                emit_progress=emit,
                cancel_check=lambda: self._cancel_check(job_id, started),
            )
            final = get_embed_index_status(
                model,
                chunks,
                source_manifest=manifest,
                model_identity=identity,
            )
            self._persist(
                {
                    **final,
                    # ready判定値ではなく、このjobの生成/再利用件数と速度実績を残す。
                    **self._job_result_fields(last_progress),
                    **self._progress_rate_fields(
                        last_progress, len(chunks), started, previous_rate
                    ),
                    "eta_seconds": None,
                    "state": "ready",
                    "job_id": job_id,
                    "generation": generation,
                    "embed_model": model,
                    "mode": mode,
                    "cancel_requested": False,
                }
            )
        except IndexCancelled:
            self._persist(
                {
                    **load_index_status(),
                    "state": "cancelled",
                    "job_id": job_id,
                    "cancel_requested": True,
                }
            )
        except SourceSnapshotError as exc:
            self._persist(
                {
                    **load_index_status(),
                    "state": "failed",
                    "job_id": job_id,
                    "error_code": exc.code,
                }
            )
        except (EmbedBuildError, EmbeddingBatchError, EmbedIndexBusyError) as exc:
            self._persist(
                {
                    **load_index_status(),
                    "state": "failed",
                    "job_id": job_id,
                    "error_code": getattr(exc, "code", "INDEX_BUSY"),
                }
            )
        except BaseException:
            self._persist(
                {
                    **load_index_status(),
                    "state": "failed",
                    "job_id": job_id,
                    "error_code": "INDEX_INTERNAL_ERROR",
                }
            )
        finally:
            with self._lock:
                self._worker = None

    def start(
        self,
        *,
        resume: bool = False,
        mode: str | None = None,
        expected_generation: str | None = None,
    ) -> dict:
        """Start or join the current job; duplicate starts never create workers."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("INDEX_SHUTDOWN")
            if mode is not None:
                # 不正modeは既存jobとの比較より先に入力エラーとして拒否する。
                mode = _validate_embed_index_mode(mode)
            persisted = load_index_status()
            persisted_mode = str(persisted.get("mode") or "incremental")
            if self._worker is not None and self._worker.is_alive():
                if mode is not None and mode != persisted_mode:
                    raise IndexModeConflictError("active index job mode differs")
                return self.status()
            if resume and mode is not None and mode != persisted_mode:
                raise IndexModeConflictError("resume mode differs from saved job")
            effective_mode = persisted_mode if resume else (mode or "incremental")
            effective_mode = _validate_embed_index_mode(effective_mode)
            if persisted.get("state") in {"building", "cancelling"}:
                if not resume:
                    # A normal duplicate start joins the durable job identity.
                    return self.status()
                # After a process restart, the status file can remain building
                # even though no worker owns the writer lock anymore.  Probe
                # the same cross-process lock before resuming the checkpoint.
                with _embed_writer_lock() as available:
                    if not available:
                        return self.status()
            model = detect_embed_model()
            if not model:
                return self._persist({"state": "failed", "error_code": "EMBED_MODEL_NOT_CONFIGURED"})
            chunks, manifest = build_source_snapshot()
            identity = get_embed_model_identity(model)
            generation = compute_embed_generation(model, chunks)
            if expected_generation and expected_generation != generation:
                raise IndexGenerationChangedError(
                    "source changed after index preview; confirm again"
                )
            if resume and persisted.get("generation") and persisted.get("generation") != generation:
                raise IndexGenerationChangedError(
                    "source changed since the saved index job; confirm again"
                )
            job_id = (
                str(persisted.get("job_id"))
                if persisted.get("state") in {"building", "cancelling"}
                and persisted.get("job_id")
                else uuid.uuid4().hex
            )
            self._job_id = job_id
            self._cancel.clear()
            self._events.clear()
            previous_rate = _generated_rate_record(persisted, model)
            self._persist(
                {
                    "state": "building",
                    "job_id": job_id,
                    "_allow_restart": True,
                    "_allow_generation_change": True,
                    "embed_model": model,
                    "mode": effective_mode,
                    "generation": generation,
                    "processed": 0,
                    "generated": 0,
                    "reused": 0,
                    "failed": 0,
                    "checkpointed": 0,
                    **previous_rate,
                    "cancel_requested": False,
                }
            )
            self._worker = threading.Thread(
                target=self._run_worker,
                args=(
                    job_id,
                    effective_mode,
                    (chunks, manifest, identity, generation),
                    previous_rate,
                ),
                daemon=True,
                name="offline-ai-index-worker",
            )
            self._worker.start()
            return self.status()

    def run_blocking(self, *, resume: bool = False, mode: str | None = None, emit=None) -> int:
        """Build in the CLI process while using the same durable contract."""
        with self._lock:
            if self._shutdown:
                return 2
            persisted = load_index_status()
            persisted_mode = str(persisted.get("mode") or "incremental")
            if resume:
                effective_mode = persisted_mode if mode is None else mode
                if mode is not None and mode != persisted_mode:
                    return 2
            else:
                effective_mode = mode or "incremental"
            try:
                effective_mode = _validate_embed_index_mode(effective_mode)
            except ValueError:
                return 2
            if persisted.get("state") in {"building", "cancelling"}:
                if not resume:
                    return 1
                with _embed_writer_lock() as available:
                    if not available:
                        return 1
            job_id = (
                str(persisted.get("job_id"))
                if persisted.get("state") in {"building", "cancelling"}
                and persisted.get("job_id")
                else uuid.uuid4().hex
            )
            self._job_id = job_id
            self._cancel.clear()
        try:
            model = detect_embed_model()
            if not model:
                self._persist({"state": "failed", "job_id": job_id, "error_code": "EMBED_MODEL_NOT_CONFIGURED"})
                return 2
            chunks, manifest = build_source_snapshot()
            identity = get_embed_model_identity(model)
            generation = compute_embed_generation(model, chunks)
            previous_rate = _generated_rate_record(persisted, model)
            last_progress: dict = {}
            self._persist({"state": "building", "job_id": job_id, "_allow_restart": True, "_allow_generation_change": True, "generation": generation, "embed_model": model, "mode": effective_mode, "total": len(chunks), "processed": 0, "generated": 0, "reused": 0, "failed": 0, "checkpointed": 0, **previous_rate, "cancel_requested": False})
            started = time.monotonic()

            def progress(event: dict) -> None:
                last_progress.clear()
                last_progress.update(event)
                status = {
                    **event,
                    "state": "building",
                    "job_id": job_id,
                    "generation": generation,
                    "embed_model": model,
                    "mode": effective_mode,
                    **self._progress_rate_fields(event, len(chunks), started, previous_rate),
                }
                self._persist(status)
                if emit:
                    emit(status)

            build_or_update_embed_index(model, chunks, source_manifest=manifest, source_root=SKILL_SOURCE_DIR, mode=effective_mode, model_identity=identity, emit_progress=progress, cancel_check=lambda: self._cancel_check(job_id, started))
            self._persist({**get_embed_index_status(model, chunks, source_manifest=manifest, model_identity=identity), **self._job_result_fields(last_progress), **self._progress_rate_fields(last_progress, len(chunks), started, previous_rate), "eta_seconds": None, "state": "ready", "job_id": job_id, "generation": generation, "embed_model": model, "mode": effective_mode, "cancel_requested": False})
            return 0
        except KeyboardInterrupt:
            self._cancel.set()
            self._persist(
                {
                    **load_index_status(),
                    "state": "cancelled",
                    "job_id": job_id,
                    "cancel_requested": True,
                }
            )
            return 130
        except IndexCancelled:
            self._persist({**load_index_status(), "state": "cancelled", "job_id": job_id, "cancel_requested": True})
            return 130
        except SourceSnapshotError as exc:
            self._persist({**load_index_status(), "state": "failed", "job_id": job_id, "error_code": exc.code})
            return 1
        except (EmbedBuildError, EmbeddingBatchError, EmbedIndexBusyError) as exc:
            self._persist({**load_index_status(), "state": "failed", "job_id": job_id, "error_code": getattr(exc, "code", "INDEX_BUSY")})
            return 1
        except BaseException:
            self._persist({**load_index_status(), "state": "failed", "job_id": job_id, "error_code": "INDEX_INTERNAL_ERROR"})
            return 1

    def cancel(self, job_id: str | None = None) -> dict:
        with self._lock:
            persisted = load_index_status()
            current = job_id or persisted.get("job_id") or self._job_id
            if not current or persisted.get("job_id") != current:
                return self.status()
            if persisted.get("state") in {"building", "cancelling"}:
                worker_alive = self._worker is not None and self._worker.is_alive()
                if not worker_alive:
                    with _embed_writer_lock() as available:
                        if available:
                            # 所有者不在（このプロセスにworkerがおらず、他プロセスも
                            # writer lockを持っていない）。cancellingを書かず、
                            # 実測状態を再評価して確定させた結果を返す。
                            return self.status()
                self._cancel.set()
                return self._persist(
                    {**persisted, "state": "cancelling", "cancel_requested": True}
                )
            return self.status()

    def subscribe(self, job_id: str | None = None) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue()
        with self._lock:
            for event in self._events:
                if not job_id or event.get("job_id") == job_id:
                    subscriber.put(event)
            current = self.status()
            if (
                (self._worker is not None and self._worker.is_alive())
                or current.get("state") in {"building", "cancelling"}
            ):
                self._subscribers.append(subscriber)
            else:
                subscriber.put(self._job_status_event(current))
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

    def reconcile_orphaned_state(self) -> None:
        """所有者不在で固着した building/cancelling を実測状態へ戻す（F-9対応）。

        プロセス再起動直後（web_server の起動時、index.bat の入口）に
        呼び出し元が明示的に1回呼ぶ。``__init__`` 内では行わない
        （コンストラクタでのfilesystem/lock副作用を避け、既存のユニット
        テストが素通しでインスタンス化できる状態を保つため）。
        既存の ``start(resume=True)`` と同じ lock probe で「所有者不在」を
        判定する: workerがこのプロセスになく、かつ writer lock が取得
        できるなら、他プロセスの worker も存在しないとみなす。
        """
        with self._lock:
            persisted = load_index_status()
            if persisted.get("state") not in {"building", "cancelling"}:
                return
            worker_alive = self._worker is not None and self._worker.is_alive()
            if worker_alive:
                return
            with _embed_writer_lock() as available:
                if not available:
                    return
                tmp_error_code, parsed_cache = self._reconcile_orphaned_tmp()
                model = detect_embed_model()
                # 孤児 .tmp の可読性判定で既に parse 済みなら、そのまま渡して
                # 起動時に 441 MB 級の cache を二度 parse しないようにする。
                actual = (
                    get_embed_index_status(
                        model, cache=parsed_cache, validate_entries=False
                    )
                    if model
                    else {"state": "missing", "generation": None, "total": 0}
                )
                reconciled = {
                    **persisted,
                    "state": actual.get("state", "missing"),
                    "generation": actual.get("generation"),
                    "embed_model": actual.get("embed_model"),
                    "total": actual.get("total"),
                    "cancel_requested": False,
                }
                if tmp_error_code:
                    # 孤児 .tmp を残した理由を復旧材料として status へ残す。
                    reconciled["error_code"] = tmp_error_code
                reconciled["_allow_generation_change"] = True
                self._persist(reconciled)

    @staticmethod
    def _reconcile_orphaned_tmp() -> tuple[str | None, dict | None]:
        """正常な embed_cache.json が読める場合に限り、孤児 .tmp を削除する。

        破損・未存在で読めない場合は削除せず、復旧材料を残して error_code を
        返す。可読性の判定に ``load_embed_cache()`` は使わない。あちらは破損時
        も空キャッシュ（``entries`` が空 dict）を返す fail-open な契約であり、
        「読めた」と区別できないため、ここでは実ファイルを直接 parse する。

        戻り値は ``(error_code, cache)``。``cache`` は判定のために parse した
        ``embed_cache.json`` の内容で、孤児 .tmp が無い場合と読めなかった場合は
        ``None``。呼び出し元は同じ起動処理の中でこれを使い回し、大きな cache を
        二度 parse しないようにする。
        """
        tmp_pattern = re.compile(
            rf"^{re.escape(EMBED_CACHE_PATH.name)}\.tmp(?:\.[0-9a-f]{{32}})?$"
        )
        try:
            tmp_paths = sorted(
                path
                for path in EMBED_CACHE_PATH.parent.iterdir()
                if path.is_file() and not path.is_symlink() and tmp_pattern.match(path.name)
            )
        except OSError:
            tmp_paths = []
        if not tmp_paths:
            return None, None
        if not EMBED_CACHE_PATH.exists():
            return "EMBED_CACHE_UNREADABLE", None
        try:
            cache = json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return "EMBED_CACHE_UNREADABLE", None
        if not isinstance(cache, dict) or not isinstance(cache.get("entries"), dict):
            return "EMBED_CACHE_UNREADABLE", None
        for tmp_path in tmp_paths:
            try:
                tmp_path.unlink()
            except OSError:
                return "EMBED_CACHE_TMP_REMOVE_FAILED", cache
        return None, cache

    def is_running(self) -> bool:
        """このプロセスのworkerが索引を構築中か（ページ閉鎖時の自動停止を保留する判定用）。"""
        with self._lock:
            return self._worker is not None and self._worker.is_alive()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            worker_alive = self._worker is not None and self._worker.is_alive()
        if not worker_alive:
            # 実行中 build が無ければ状態を再評価する必要はない。cancel() は
            # status() 経由で embed_cache の parse と skill-source 走査を行い、
            # 実データでは Ctrl+C 終了が数秒遅れる。戻り値も使わないため、
            # cancel フラグのセットだけで早期復帰する。
            self._cancel.set()
            return
        try:
            self.cancel()
        except Exception:
            self._cancel.set()
