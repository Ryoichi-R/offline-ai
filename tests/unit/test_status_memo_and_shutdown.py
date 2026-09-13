"""status read 経路の memo と shutdown 早期復帰の回帰テスト。

対象:
- build_source_chunks / compute_embed_generation の read 経路 memo（O2-8）
- skill-source の追加・更新・削除による memo 無効化
- OFFLINE_AI_SOURCE_CHUNK_MEMO=0 での無効化
- writer 経路（build_source_chunks 直呼び）が memo をバイパスする
- IndexCoordinator.shutdown() が worker 不在時に状態を再評価しない（O2-7）

検証対象: _internal/search.py の状態メモと停止処理
"""

import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import index_service  # noqa: E402
import search  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_source_chunk_memo():
    search._invalidate_source_chunk_memo()
    yield
    search._invalidate_source_chunk_memo()


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "skill-source"
    root.mkdir()
    (root / "a.md").write_text("# 見出しA\n\n本文A です。\n", encoding="utf-8")
    (root / "b.md").write_text("# 見出しB\n\n本文B です。\n", encoding="utf-8")
    return root


def _count_calls(monkeypatch):
    calls = {"n": 0}
    original = search.build_source_chunks

    def counted(source_root=None):
        calls["n"] += 1
        return original(source_root)

    monkeypatch.setattr(search, "build_source_chunks", counted)
    return calls


# --- chunk memo -------------------------------------------------------------


def test_second_call_reuses_memo_for_identical_source_set(source_root, monkeypatch):
    calls = _count_calls(monkeypatch)
    first, key1 = search.load_source_chunks_cached(source_root)
    second, key2 = search.load_source_chunks_cached(source_root)
    assert calls["n"] == 1
    assert second is first
    assert key1 == key2


def test_memo_invalidated_when_file_added(source_root, monkeypatch):
    calls = _count_calls(monkeypatch)
    search.load_source_chunks_cached(source_root)
    (source_root / "c.md").write_text("# 見出しC\n\n本文C です。\n", encoding="utf-8")
    search.load_source_chunks_cached(source_root)
    assert calls["n"] == 2


def test_memo_invalidated_when_file_updated(source_root, monkeypatch):
    calls = _count_calls(monkeypatch)
    search.load_source_chunks_cached(source_root)
    target = source_root / "a.md"
    target.write_text("# 見出しA\n\n本文A を更新しました。\n", encoding="utf-8")
    os.utime(target, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    search.load_source_chunks_cached(source_root)
    assert calls["n"] == 2


def test_memo_invalidated_when_same_size_and_mtime_file_content_changes(
    source_root, monkeypatch
):
    """mtime/sizeが同じでもhash差分でmemoを無効化する。"""
    calls = _count_calls(monkeypatch)
    target = source_root / "a.md"
    search.load_source_chunks_cached(source_root)
    original_stat = target.stat()
    original_bytes = target.read_bytes()
    target.write_bytes(original_bytes.replace("本文A".encode(), "本文Z".encode()))
    # 内容だけを置換してから、元の時刻へ戻す。
    assert target.stat().st_size == original_stat.st_size
    os.utime(target, ns=(original_stat.st_mtime_ns, original_stat.st_mtime_ns))
    search.load_source_chunks_cached(source_root)
    assert calls["n"] == 2


def test_memo_invalidated_when_file_deleted(source_root, monkeypatch):
    calls = _count_calls(monkeypatch)
    search.load_source_chunks_cached(source_root)
    (source_root / "b.md").unlink()
    search.load_source_chunks_cached(source_root)
    assert calls["n"] == 2


def test_memo_can_be_disabled_by_env_flag(source_root, monkeypatch):
    monkeypatch.setattr(search, "SOURCE_CHUNK_MEMO_ENABLED", False)
    calls = _count_calls(monkeypatch)
    _, key1 = search.load_source_chunks_cached(source_root)
    _, key2 = search.load_source_chunks_cached(source_root)
    assert calls["n"] == 2
    assert key1 is None and key2 is None


def test_writer_path_bypasses_memo(source_root, monkeypatch):
    """build_source_chunks 直呼び（index build 経路）は memo を通らない。"""
    calls = _count_calls(monkeypatch)
    search.load_source_chunks_cached(source_root)
    search.build_source_chunks(source_root)
    search.build_source_chunks(source_root)
    assert calls["n"] == 3


# --- generation memo --------------------------------------------------------


def test_generation_memo_reused_for_same_key_and_model(source_root, monkeypatch):
    chunks, key = search.load_source_chunks_cached(source_root)
    calls = {"n": 0}
    original = search.compute_embed_generation

    def counted(embed_model, chunk_list):
        calls["n"] += 1
        return original(embed_model, chunk_list)

    monkeypatch.setattr(search, "compute_embed_generation", counted)
    first = search.compute_embed_generation_cached("bge-m3", chunks, key)
    second = search.compute_embed_generation_cached("bge-m3", chunks, key)
    assert calls["n"] == 1
    assert first == second


def test_generation_memo_recomputed_for_different_model(source_root, monkeypatch):
    chunks, key = search.load_source_chunks_cached(source_root)
    calls = {"n": 0}
    original = search.compute_embed_generation

    def counted(embed_model, chunk_list):
        calls["n"] += 1
        return original(embed_model, chunk_list)

    monkeypatch.setattr(search, "compute_embed_generation", counted)
    a = search.compute_embed_generation_cached("bge-m3", chunks, key)
    b = search.compute_embed_generation_cached("other-model", chunks, key)
    assert calls["n"] == 2
    assert a != b


def test_generation_memo_bypassed_without_memo_key(source_root, monkeypatch):
    chunks = search.build_source_chunks(source_root)
    calls = {"n": 0}
    original = search.compute_embed_generation

    def counted(embed_model, chunk_list):
        calls["n"] += 1
        return original(embed_model, chunk_list)

    monkeypatch.setattr(search, "compute_embed_generation", counted)
    search.compute_embed_generation_cached("bge-m3", chunks, None)
    search.compute_embed_generation_cached("bge-m3", chunks, None)
    assert calls["n"] == 2


# --- shutdown 早期復帰 ------------------------------------------------------


def test_shutdown_does_not_reevaluate_status_without_worker(monkeypatch):
    """worker 不在時に status()/load_index_status() を呼ばない（Ctrl+C 遅延の解消）。"""
    coordinator = index_service.IndexCoordinator()

    def fail(*_args, **_kwargs):  # pragma: no cover - 呼ばれたら契約違反
        raise AssertionError("shutdown が状態を再評価した")

    monkeypatch.setattr(index_service, "load_index_status", fail)
    monkeypatch.setattr(coordinator, "status", fail)

    coordinator.shutdown()

    assert coordinator._cancel.is_set()
    assert coordinator._shutdown


def test_shutdown_still_cancels_running_worker(monkeypatch):
    coordinator = index_service.IndexCoordinator()

    class _AliveWorker:
        def is_alive(self):
            return True

    coordinator._worker = _AliveWorker()
    called = {"n": 0}

    def cancel(job_id=None):
        called["n"] += 1
        return {}

    monkeypatch.setattr(coordinator, "cancel", cancel)

    coordinator.shutdown()

    assert called["n"] == 1


def test_shutdown_is_idempotent(monkeypatch):
    coordinator = index_service.IndexCoordinator()
    coordinator.shutdown()
    calls = {"n": 0}

    def cancel(job_id=None):  # pragma: no cover - 2回目は呼ばれない
        calls["n"] += 1
        return {}

    monkeypatch.setattr(coordinator, "cancel", cancel)
    coordinator.shutdown()
    assert calls["n"] == 0
