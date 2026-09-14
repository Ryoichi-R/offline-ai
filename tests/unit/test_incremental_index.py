"""ファイル単位の差分インデックス更新の回帰テスト。

対象: plans/offline-ai-file-incremental-index-plan.md のテスト計画
- 差分の正確性（未変更0件・追加/変更は当該ファイルの全chunk・削除/rename）
- 実hash判定（同size/同mtime変更、空ファイル、全削除、読取/解析失敗）
- incremental と full の登録内容・引用位置の同値性（決定的な偽Embedding）
- 互換契約（digest・parser・schema）とdigest不明時の安全側動作
- 候補検証・昇格直前の再確認・保存失敗時の旧cache保全
- 副作用のない事前解析と、生成件数だけを分母にした見積もり
- job状態・Web API・CLIのmode互換
"""

import hashlib
import http.client
import json
import os
import threading
import time
from pathlib import Path

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import index_cli  # noqa: E402
import index_service  # noqa: E402
import search  # noqa: E402
import source_view  # noqa: E402
import web_server  # noqa: E402

MODEL = "bge-m3"


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[0] / 255.0, digest[1] / 255.0, digest[2] / 255.0, 1.0]


class _Env:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.src = tmp_path / "skill-source"
        self.src.mkdir()
        self.cache_path = tmp_path / "embed_cache.json"
        self.checkpoint_path = tmp_path / "embed_cache.checkpoints"
        self.status_path = tmp_path / "embed_index_status.json"
        self.digest = "fixture-digest"
        self.calls: list[str] = []
        self.embed_delay = 0.0
        monkeypatch.setattr(search, "SKILL_SOURCE_DIR", self.src)
        monkeypatch.setattr(index_service, "SKILL_SOURCE_DIR", self.src)
        monkeypatch.setattr(search, "EMBED_CACHE_PATH", self.cache_path)
        monkeypatch.setattr(index_service, "EMBED_CACHE_PATH", self.cache_path)
        monkeypatch.setattr(search, "EMBED_CHECKPOINT_PATH", self.checkpoint_path)
        monkeypatch.setattr(search, "EMBED_LOCK_PATH", tmp_path / "embed_cache.lock")
        monkeypatch.setattr(search, "EMBED_STATUS_PATH", self.status_path)
        monkeypatch.setattr(search, "EMBED_PROGRESS_SECONDS", 3600)
        monkeypatch.setattr(search, "get_embed_model_identity", self.identity)
        monkeypatch.setattr(index_service, "get_embed_model_identity", self.identity)
        monkeypatch.setattr(index_service, "detect_embed_model", lambda: MODEL)
        monkeypatch.setattr(search, "_get_embedding", self.embed)
        search._invalidate_embed_cache_memo()
        search._invalidate_source_chunk_memo()

    def identity(self, model, **_kwargs):
        if not self.digest:
            return None
        return {"name": search._canonical_model_reference(model), "digest": self.digest}

    def embed(self, text, _model):
        self.calls.append(text)
        if self.embed_delay:
            time.sleep(self.embed_delay)
        return _vector(text)

    def write(self, rel: str, text: str) -> Path:
        path = self.src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def build(self, **kwargs):
        self.calls.clear()
        return search.build_or_update_embed_index(MODEL, **kwargs)

    def cache(self) -> dict:
        return json.loads(self.cache_path.read_text(encoding="utf-8"))

    def chunk_texts(self, rel: str) -> list[str]:
        return [
            chunk["text"]
            for chunk in search.build_source_chunks(self.src)
            if chunk["path"] == rel
        ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    return _Env(tmp_path, monkeypatch)


def _long_text(label: str, sections: int = 3) -> str:
    body = []
    for number in range(sections):
        body.append(f"# {label} 見出し{number}")
        body.extend(f"{label} 本文 {number}-{line} " + "あ" * 60 for line in range(30))
    return "\n".join(body) + "\n"


def _comparable(cache: dict) -> dict:
    entries = {
        chunk_id: {key: value for key, value in entry.items() if key != "modifiedAt"}
        for chunk_id, entry in cache["entries"].items()
    }
    return {"files": cache["files"], "entries": entries, "generation": cache["generation"]}


# --- 差分の正確性 --------------------------------------------------------------


def test_unchanged_source_generates_nothing_and_deleted_file_leaves_no_entries(env):
    env.write("a.md", "本文A")
    env.write("b.md", "本文B")
    env.build()
    assert sorted(env.calls) == ["本文A", "本文B"]

    env.build()
    assert env.calls == []

    (env.src / "b.md").unlink()
    env.build()
    cache = env.cache()
    assert env.calls == []
    assert set(cache["files"]) == {"a.md"}
    assert all(entry["path"] == "a.md" for entry in cache["entries"].values())
    assert search.get_embed_index_status(MODEL)["state"] == "ready"


def test_added_and_changed_files_recompute_all_of_their_chunks_only(env):
    env.write("big.md", _long_text("big"))
    env.write("same.md", _long_text("same"))
    env.build()
    assert len(env.chunk_texts("big.md")) >= 2

    env.write("big.md", _long_text("big") + "末尾だけ追記\n")
    env.write("new.md", "追加本文")
    env.build()

    expected = env.chunk_texts("big.md") + env.chunk_texts("new.md")
    assert sorted(env.calls) == sorted(expected)


def test_rename_is_handled_as_delete_plus_add(env):
    env.write("old-name.md", "移動する本文")
    env.write("keep.md", "残る本文")
    env.build()

    (env.src / "old-name.md").rename(env.src / "new-name.md")
    env.build()

    cache = env.cache()
    assert env.calls == ["移動する本文"]
    assert set(cache["files"]) == {"keep.md", "new-name.md"}
    assert not any(entry["path"] == "old-name.md" for entry in cache["entries"].values())


# --- hash判定 ------------------------------------------------------------------


def test_same_size_and_mtime_change_is_detected_by_file_hash(env):
    target = env.write("a.md", "本文A")
    env.write("b.md", "本文B")
    env.build()
    stat = target.stat()

    target.write_text("本文Z", encoding="utf-8")
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert target.stat().st_size == stat.st_size
    assert target.stat().st_mtime_ns == stat.st_mtime_ns

    plan = search.plan_embed_index_update(MODEL)
    assert (plan["changed_files"], plan["generated_chunks"]) == (1, 1)
    env.build()
    assert env.calls == ["本文Z"]
    assert search.get_embed_index_status(MODEL)["state"] == "ready"


def test_empty_file_is_kept_in_manifest_and_does_not_block_ready(env):
    env.write("a.md", "本文A")
    env.write("empty.md", "")
    env.build()

    cache = env.cache()
    assert cache["files"]["empty.md"]["chunk_count"] == 0
    assert search.get_embed_index_status(MODEL)["state"] == "ready"
    # 検索経路は build_source_chunks() の結果を渡す。空ファイルでkeyword退避しない。
    chunks = search.build_source_chunks(env.src)
    status = search.get_embed_index_status(MODEL, chunks, cache=search.load_embed_cache())
    assert status["state"] == "ready"

    env.write("a.md", "")
    plan = search.plan_embed_index_update(MODEL)
    assert (plan["changed_files"], plan["generated_chunks"]) == (1, 0)
    assert plan["estimate_status"] == "no_embedding"
    env.build()
    assert env.calls == []
    assert env.cache()["entries"] == {}


def test_deleting_all_sources_removes_every_entry(env):
    env.write("a.md", "本文A")
    env.build()
    (env.src / "a.md").unlink()

    plan = search.plan_embed_index_update(MODEL)
    assert (plan["deleted_files"], plan["generated_chunks"]) == (1, 0)
    env.build()

    assert env.calls == []
    assert env.cache()["files"] == {}
    assert env.cache()["entries"] == {}
    assert search.get_embed_index_status(MODEL)["state"] == "ready"


def test_read_failure_is_not_treated_as_deletion(env, monkeypatch):
    env.write("a.md", "本文A")
    locked = env.write("locked.md", "読めない本文")
    env.build()
    before = env.cache_path.read_bytes()
    real_read_bytes = Path.read_bytes

    def read_bytes(self):
        if self == locked:
            raise PermissionError("simulated sharing violation")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    with pytest.raises(search.SourceSnapshotError) as excinfo:
        env.build()
    assert excinfo.value.failed_paths == ["locked.md"]
    with pytest.raises(search.SourceSnapshotError):
        search.plan_embed_index_update(MODEL)
    assert env.cache_path.read_bytes() == before

    # 検索read経路は止めないが、不完全snapshotを ready と判定しない。
    chunks = search.build_source_chunks(env.src)
    assert [chunk["path"] for chunk in chunks] == ["a.md"]
    assert chunks.source_complete is False
    status = search.get_embed_index_status(MODEL, chunks, cache=search.load_embed_cache())
    assert status["state"] == "stale"


def test_parser_failure_is_reported_as_snapshot_failure(env, monkeypatch):
    env.write("a.md", "本文A")
    broken = env.write("broken.md", "本文B")
    real_loader = search.load_metadata_sidecar

    def loader(path):
        if Path(path) == broken:
            raise ValueError("simulated parser failure")
        return real_loader(path)

    monkeypatch.setattr(search, "load_metadata_sidecar", loader)

    with pytest.raises(search.SourceSnapshotError) as excinfo:
        env.build()
    assert excinfo.value.failed_paths == ["broken.md"]
    assert not env.cache_path.exists()


def test_non_utf8_source_is_indexed_with_the_same_text_as_search(env):
    env.write("a.md", "本文A")
    (env.src / "legacy.md").write_bytes("旧文字コード".encode("cp932"))

    chunks = search.build_source_chunks(env.src)
    assert {chunk["path"] for chunk in chunks} == {"a.md", "legacy.md"}
    env.build()

    assert search.get_embed_index_status(MODEL)["state"] == "ready"
    legacy_text = next(chunk["text"] for chunk in chunks if chunk["path"] == "legacy.md")
    assert legacy_text in env.calls


# --- 内容同値性 ----------------------------------------------------------------


def test_incremental_result_equals_full_rebuild_for_the_same_final_sources(env):
    env.write("a.md", _long_text("a"))
    env.write("b.md", _long_text("b"))
    env.write("c.md", "削除予定")
    env.write("d.md", "")
    env.build()
    env.write("a.md", _long_text("a", sections=4))
    (env.src / "c.md").unlink()
    (env.src / "b.md").rename(env.src / "b2.md")
    env.write("e.md", "# 追加\n追加本文\n")
    incremental = _comparable(env.build())
    assert _comparable(env.cache()) == incremental

    full = _comparable(env.build(mode="full"))

    assert full == incremental
    assert len(env.calls) == len(search.build_source_chunks(env.src))


# --- キャッシュ互換性 ----------------------------------------------------------


@pytest.mark.parametrize("change", ["digest", "parser_version", "compatibility_version"])
def test_compatibility_change_recomputes_every_chunk(env, monkeypatch, change):
    env.write("a.md", "本文A")
    env.write("b.md", "本文B")
    env.build()

    if change == "digest":
        env.digest = "replaced-same-name-model"
    elif change == "parser_version":
        monkeypatch.setattr(search, "EMBED_SOURCE_PARSER_VERSION", search.EMBED_SOURCE_PARSER_VERSION + 1)
    else:
        monkeypatch.setattr(search, "EMBED_COMPATIBILITY_VERSION", search.EMBED_COMPATIBILITY_VERSION + 1)

    assert search.get_embed_index_status(MODEL)["state"] != "ready"
    plan = search.plan_embed_index_update(MODEL)
    assert plan["compatibility"] == "rebuild"
    assert plan["reason"] == "cache_incompatible"
    assert plan["generated_chunks"] == 2
    env.build()
    assert sorted(env.calls) == ["本文A", "本文B"]


def test_corrupted_entry_in_unchanged_file_recomputes_that_whole_file(env):
    env.write("big.md", _long_text("big"))
    env.write("other.md", "他の本文")
    env.build()
    cache = env.cache()
    big_ids = cache["files"]["big.md"]["chunk_ids"]
    assert len(big_ids) >= 2
    cache["entries"][big_ids[-1]]["embedding"] = [1.0, 0.0]  # 次元不一致
    env.cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    search._invalidate_embed_cache_memo()

    plan = search.plan_embed_index_update(MODEL)
    assert plan["generated_chunks"] == len(big_ids)
    env.build()

    assert sorted(env.calls) == sorted(env.chunk_texts("big.md"))


def test_legacy_cache_without_manifest_is_rebuilt_instead_of_reused(env):
    env.write("a.md", "本文A")
    env.build()
    cache = env.cache()
    cache["version"] = 4
    cache.pop("files")
    cache.pop("compatibility")
    env.cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    search._invalidate_embed_cache_memo()

    assert search.get_embed_index_status(MODEL)["state"] == "stale"
    env.build()
    assert env.calls == ["本文A"]


def test_plan_counts_file_delta_against_legacy_cache_without_manifest(env):
    """旧形式cacheでも、追加・変更・削除・未変更をentryから数える（全件追加扱いにしない）。"""
    env.write("same.md", "未変更本文")
    env.write("changed.md", "変更前本文")
    env.write("deleted.md", "削除する本文")
    env.build()
    cache = env.cache()
    cache["version"] = 4
    cache.pop("files")
    cache.pop("compatibility")
    env.cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    search._invalidate_embed_cache_memo()
    env.write("changed.md", "変更後本文")
    (env.src / "deleted.md").unlink()
    env.write("added.md", "追加本文")

    plan = search.plan_embed_index_update(MODEL)

    assert (
        plan["added_files"],
        plan["changed_files"],
        plan["deleted_files"],
        plan["unchanged_files"],
    ) == (1, 1, 1, 1)
    # 旧形式は互換性を証明できないため、件数表示と独立に全件再計算する。
    assert plan["compatibility"] == "rebuild"
    assert plan["reason"] == "cache_incompatible"
    assert (plan["generated_chunks"], plan["reused_chunks"]) == (3, 0)


def test_unknown_model_digest_never_reuses_but_can_become_ready(env):
    env.digest = None
    env.write("a.md", "本文A")
    env.write("b.md", "本文B")
    env.build()
    assert env.cache()["compatibility"]["model_digest"] is None
    assert search.get_embed_index_status(MODEL)["state"] == "ready"

    plan = search.plan_embed_index_update(MODEL)
    assert plan["reason"] == "model_digest_unavailable"
    assert plan["model_digest_available"] is False
    assert plan["generated_chunks"] == 2
    env.build()
    assert sorted(env.calls) == ["本文A", "本文B"]

    # digestを取得できるようになったら、同一モデルと証明できないcacheはreadyにしない。
    env.digest = "now-available"
    assert search.get_embed_index_status(MODEL)["state"] == "stale"


def test_promotion_invalidates_read_memo_and_plan_does_not_mutate_it(env):
    env.write("a.md", "本文A")
    env.build()
    memo_before = search.load_embed_cache()
    snapshot = json.dumps(memo_before, ensure_ascii=False, sort_keys=True)

    env.write("a.md", "本文Aを変更")
    search.plan_embed_index_update(MODEL)
    assert json.dumps(search.load_embed_cache(), ensure_ascii=False, sort_keys=True) == snapshot

    env.build()
    after = search.load_embed_cache()
    assert after is not memo_before
    assert after["generation"] != memo_before["generation"]
    assert next(iter(after["entries"].values()))["text"] == "本文Aを変更"


# --- 停止・障害 ----------------------------------------------------------------


def _changed_sources(env):
    env.write("a.md", "本文A")
    env.write("b.md", "本文B")
    env.build()
    env.write("a.md", "本文Aを変更")
    return env.cache_path.read_bytes()


def _no_cache_tmp(env) -> bool:
    return not list(env.tmp_path.glob("embed_cache.json.tmp*"))


def test_candidate_that_fails_reload_validation_is_never_promoted(env, monkeypatch):
    before = _changed_sources(env)
    real_load = search._load_embed_cache_file

    def corrupt_reload(path):
        loaded = real_load(path)
        if path != env.cache_path:
            loaded["entries"].popitem()
        return loaded

    monkeypatch.setattr(search, "_load_embed_cache_file", corrupt_reload)

    with pytest.raises(search.EmbedCachePersistenceError):
        env.build()

    assert env.cache_path.read_bytes() == before
    assert _no_cache_tmp(env)
    assert list(env.checkpoint_path.glob("batch-*.json"))
    assert search.get_embed_index_status(MODEL)["state"] == "stale"


def test_disk_full_while_writing_candidate_keeps_previous_cache(env, monkeypatch):
    before = _changed_sources(env)
    real_dump = search.json.dump

    def dump(payload, handle, *args, **kwargs):
        if "embed_cache.json.tmp" in str(getattr(handle, "name", "")):
            raise OSError(28, "No space left on device")
        return real_dump(payload, handle, *args, **kwargs)

    monkeypatch.setattr(search.json, "dump", dump)

    with pytest.raises(search.EmbedCachePersistenceError):
        env.build()

    assert env.cache_path.read_bytes() == before
    assert _no_cache_tmp(env)


def test_replace_failure_keeps_previous_cache(env, monkeypatch):
    before = _changed_sources(env)
    real_replace = search.os.replace

    def replace(source, destination):
        if Path(destination) == env.cache_path:
            raise PermissionError("simulated reader handle")
        return real_replace(source, destination)

    monkeypatch.setattr(search.os, "replace", replace)

    with pytest.raises(search.EmbedCachePersistenceError):
        env.build()

    assert env.cache_path.read_bytes() == before
    assert _no_cache_tmp(env)


def test_source_change_immediately_before_promotion_is_not_promoted(env, monkeypatch):
    before = _changed_sources(env)
    real_load = search._load_embed_cache_file

    def edit_source_then_reload(path):
        env.write("b.md", "昇格直前の外部編集")
        return real_load(path)

    monkeypatch.setattr(search, "_load_embed_cache_file", edit_source_then_reload)

    with pytest.raises(search.EmbedBuildError) as excinfo:
        env.build()

    assert excinfo.value.code == "SOURCE_CHANGED_DURING_BUILD"
    assert env.cache_path.read_bytes() == before
    assert _no_cache_tmp(env)


def test_cancel_immediately_before_promotion_keeps_previous_cache(env, monkeypatch):
    before = _changed_sources(env)
    armed = {"cancel": False}
    real_load = search._load_embed_cache_file

    class Cancelled(RuntimeError):
        pass

    def arm_then_reload(path):
        armed["cancel"] = True
        return real_load(path)

    def cancel_check():
        if armed["cancel"]:
            raise Cancelled("cancelled before promotion")

    monkeypatch.setattr(search, "_load_embed_cache_file", arm_then_reload)

    with pytest.raises(Cancelled):
        env.build(cancel_check=cancel_check)

    assert env.cache_path.read_bytes() == before
    assert _no_cache_tmp(env)


def test_full_mode_checkpoint_is_not_reused_by_incremental_build(env):
    for number in range(4):
        env.write(f"{number}.md", f"本文{number}")

    class Cancelled(RuntimeError):
        pass

    def cancel_after_two():
        if len(env.calls) >= 2:
            raise Cancelled("stop")

    with pytest.raises(Cancelled):
        env.build(mode="full", cancel_check=cancel_after_two)
    assert list(env.checkpoint_path.glob("batch-*.json"))

    plan = search.plan_embed_index_update(MODEL, mode="incremental")
    assert plan["checkpoint_chunks"] == 0
    env.build(mode="incremental")
    assert len(env.calls) == 4


def test_same_mode_resume_reuses_checkpoint_and_preview_counts_it(env):
    for number in range(4):
        env.write(f"{number}.md", f"本文{number}")

    class Cancelled(RuntimeError):
        pass

    def cancel_after_two_saved():
        # 3件目の生成直後に止めると、確定済みの2件だけがcheckpointに残る。
        if len(env.calls) >= 3:
            raise Cancelled("stop")

    with pytest.raises(Cancelled):
        env.build(mode="full", cancel_check=cancel_after_two_saved)

    plan = search.plan_embed_index_update(MODEL, mode="full")
    assert plan["checkpoint_chunks"] == 2
    assert plan["generated_chunks"] == 2
    env.build(mode="full")
    assert len(env.calls) == 2


# --- 事前解析・見積もり --------------------------------------------------------


def _tree_state(root: Path) -> dict:
    return {
        str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    }


def test_plan_is_side_effect_free_and_matches_the_following_build(env):
    env.write("a.md", "本文A")
    env.write("b.md", "本文B")
    env.write("c.md", "本文C")
    env.build()
    env.write("a.md", "本文Aを変更")
    (env.src / "c.md").unlink()
    env.write("d.md", "追加D")
    env.calls.clear()
    before = _tree_state(env.tmp_path)

    plan = search.plan_embed_index_update(MODEL)

    assert _tree_state(env.tmp_path) == before
    assert env.calls == []
    assert (
        plan["added_files"],
        plan["changed_files"],
        plan["deleted_files"],
        plan["unchanged_files"],
    ) == (1, 1, 1, 1)
    events = []
    env.build(emit_progress=events.append)
    assert events[-1]["generated"] == plan["generated_chunks"] == 2
    assert events[-1]["reused"] == plan["reused_chunks"] == 1
    assert plan["generation"] == env.cache()["generation"]


@pytest.mark.parametrize(
    ("status", "expected_status", "expected_estimate"),
    [
        ({}, "no_rate", None),
        ({"rate_per_second": 2.0}, "no_rate", None),  # 旧processed基準は流用しない
        ({"rate_per_second": 2.0, "rate_basis": "generated", "embed_model": "other"}, "no_rate", None),
        ({"rate_per_second": 2.0, "rate_basis": "generated", "embed_model": MODEL}, "estimated", 1.0),
    ],
)
def test_plan_estimate_uses_same_model_generated_rate_only(
    env, status, expected_status, expected_estimate
):
    env.write("a.md", "本文A")
    env.write("b.md", "本文B")
    env.status_path.write_text(json.dumps({"state": "stale", **status}), encoding="utf-8")

    plan = search.plan_embed_index_update(MODEL)

    assert plan["generated_chunks"] == 2
    assert plan["estimate_status"] == expected_status
    assert plan["estimated_seconds"] == expected_estimate


def test_plan_reports_missing_cache_and_full_request_reasons(env):
    env.write("a.md", "本文A")
    assert search.plan_embed_index_update(MODEL)["reason"] == "cache_missing"
    env.build()
    assert search.plan_embed_index_update(MODEL)["reason"] is None
    full = search.plan_embed_index_update(MODEL, mode="full")
    assert full["reason"] == "full_requested"
    assert (full["generated_chunks"], full["reused_chunks"]) == (1, 0)
    with pytest.raises(ValueError):
        search.plan_embed_index_update(MODEL, mode="partial")


# --- job状態 -------------------------------------------------------------------


def test_progress_rate_uses_embedding_time_and_protects_existing_rate():
    started = time.monotonic() - 100.0
    previous = {"rate_per_second": 10.0, "rate_basis": "generated"}
    fields = index_service.IndexCoordinator._progress_rate_fields

    small = fields(
        {"processed": 105, "generated": 5, "reused": 100, "embedding_seconds": 5.0},
        200,
        started,
        previous,
    )
    assert small["rate_per_second"] == 10.0  # 少量生成では既存実績を維持
    assert small["eta_seconds"] == 9.5

    large = fields(
        {"processed": 140, "generated": 40, "reused": 100, "embedding_seconds": 20.0},
        200,
        started,
        previous,
    )
    assert large["rate_per_second"] == 2.0  # 経過100秒ではなく生成20秒を分母にする
    assert large["rate_basis"] == "generated"

    first = fields(
        {"processed": 1, "generated": 1, "embedding_seconds": 0.5},
        2,
        started,
        {"rate_per_second": None, "rate_basis": None},
    )
    assert first["rate_per_second"] == 2.0

    none = fields({"processed": 0, "generated": 0}, 2, started, {"rate_per_second": None, "rate_basis": None})
    assert none["rate_per_second"] is None and none["eta_seconds"] is None


def test_blocking_build_records_job_counts_and_keeps_rate_for_next_estimate(env):
    for number in range(40):
        env.write(f"doc{number:02d}.md", f"本文{number}")
    env.embed_delay = 0.005
    coordinator = index_service.IndexCoordinator()

    assert coordinator.run_blocking() == 0
    first = search.load_index_status()
    assert first["state"] == "ready"
    assert (first["generated"], first["reused"]) == (40, 0)
    assert first["rate_basis"] == "generated" and first["rate_per_second"] > 0

    env.write("doc00.md", "変更した本文")
    plan = search.plan_embed_index_update(MODEL)
    assert plan["estimate_status"] == "estimated"
    assert plan["estimated_seconds"] == round(1 / first["rate_per_second"], 3)

    assert coordinator.run_blocking() == 0
    second = search.load_index_status()
    assert (second["generated"], second["reused"]) == (1, 39)
    assert second["rate_per_second"] == first["rate_per_second"]
    assert second["mode"] == "incremental"


def test_blocking_full_build_and_resume_mode_inheritance(env, monkeypatch):
    env.write("a.md", "本文A")
    coordinator = index_service.IndexCoordinator()
    assert coordinator.run_blocking(mode="full") == 0
    assert search.load_index_status()["mode"] == "full"
    assert coordinator.run_blocking(mode="partial") == 2

    monkeypatch.setattr(
        index_service,
        "load_index_status",
        lambda: {"state": "cancelled", "job_id": "saved", "mode": "full", "generation": "g"},
    )
    assert coordinator.run_blocking(resume=True, mode="incremental") == 2


def test_start_validates_mode_before_joining_and_reports_mode_conflicts(monkeypatch):
    coordinator = index_service.IndexCoordinator()

    class AliveWorker:
        def is_alive(self):
            return True

    coordinator._worker = AliveWorker()
    monkeypatch.setattr(
        index_service, "load_index_status", lambda: {"state": "building", "mode": "incremental"}
    )
    monkeypatch.setattr(coordinator, "status", lambda: {"state": "building"})

    with pytest.raises(ValueError):
        coordinator.start(mode="partial")
    with pytest.raises(index_service.IndexModeConflictError):
        coordinator.start(mode="full")
    assert coordinator.start(mode="incremental") == {"state": "building"}
    assert coordinator.start() == {"state": "building"}


# --- Web API / CLI ------------------------------------------------------------


class _FakeCoordinator:
    def __init__(self):
        self.plans: list[str] = []
        self.starts: list[dict] = []
        self.start_error: Exception | None = None

    def status(self):
        return {"state": "stale"}

    def plan(self, *, mode="incremental"):
        search._validate_embed_index_mode(mode)
        self.plans.append(mode)
        return {"mode": mode, "generation": "g1", "generated_chunks": 0}

    def start(self, **kwargs):
        if self.start_error:
            raise self.start_error
        self.starts.append(kwargs)
        return {"state": "building", "job_id": "job"}

    def shutdown(self):
        pass


@pytest.fixture
def web(tmp_path):
    server = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token="test-session-token",
        search_timeout=300,
        evidence_registry=source_view.EvidenceRegistry(tmp_path),
    )
    server.bind_port = server.server_address[1]
    server.index_coordinator.shutdown()
    fake = _FakeCoordinator()
    server.index_coordinator = fake
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def request(method, path, body=None, *, auth=True):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"}
        if auth:
            headers["Cookie"] = "offlineai_session=test-session-token"
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    try:
        yield fake, request
    finally:
        server.shutdown_jobs_once()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_index_plan_api_requires_auth_and_validates_mode(web):
    fake, request = web
    assert request("GET", "/api/index/plan", auth=False)[0] == 403
    assert request("GET", "/api/index/plan")[1]["mode"] == "incremental"
    assert request("GET", "/api/index/plan?mode=full")[1]["mode"] == "full"
    status, body = request("GET", "/api/index/plan?mode=partial")
    assert status == 400 and body["error"]["code"] == "invalid_mode"
    status, body = request("GET", "/api/index/plan?mode=full&mode=incremental")
    assert status == 400 and body["error"]["code"] == "invalid_mode"
    assert fake.plans == ["incremental", "full"]


def test_index_start_api_keeps_legacy_body_and_rejects_bad_requests(web):
    fake, request = web
    assert request("POST", "/api/index/start", auth=False)[0] == 403
    assert request("POST", "/api/index/start", "{}")[0] == 202
    assert request("POST", "/api/index/start", "")[0] == 202
    status, _ = request(
        "POST", "/api/index/start", json.dumps({"mode": "full", "expected_generation": "g1"})
    )
    assert status == 202
    assert fake.starts == [
        {"resume": False, "mode": None, "expected_generation": None},
        {"resume": False, "mode": None, "expected_generation": None},
        {"resume": False, "mode": "full", "expected_generation": "g1"},
    ]
    for body in ('{"mode": 1}', '{"expected_generation": 1}', "[]", "{broken"):
        status, payload = request("POST", "/api/index/start", body)
        assert status == 400 and payload["error"]["code"] == "invalid_mode"

    fake.start_error = ValueError("invalid index mode")
    assert request("POST", "/api/index/start", '{"mode": "partial"}')[1]["error"]["code"] == "invalid_mode"
    fake.start_error = index_service.IndexModeConflictError("resume mode differs")
    status, payload = request("POST", "/api/index/resume", '{"mode": "full"}')
    assert status == 409 and payload["error"]["code"] == "index_mode_conflict"
    fake.start_error = index_service.IndexGenerationChangedError("confirm again")
    status, payload = request("POST", "/api/index/start", '{"expected_generation": "old"}')
    assert status == 409 and payload["error"]["code"] == "index_source_changed"


def test_cli_full_flag_is_passed_only_to_build(monkeypatch):
    calls = []

    class Coordinator:
        def reconcile_orphaned_state(self):
            pass

        def run_blocking(self, **kwargs):
            calls.append(kwargs)
            return 0

    monkeypatch.setattr(index_cli, "IndexCoordinator", Coordinator)

    assert index_cli.main(["build", "--full"]) == 0
    assert index_cli.main(["build"]) == 0
    assert index_cli.main(["resume"]) == 0
    assert [(call["resume"], call["mode"]) for call in calls] == [
        (False, "full"),
        (False, None),
        (True, None),
    ]
    with pytest.raises(SystemExit):
        index_cli.main(["resume", "--full"])


# --- 検索・引用（差分更新後の検索経路） --------------------------------------


def _fake_query_embeddings(texts, _model, **_kwargs):
    return [_vector(text) for text in texts]


def _current_chunks_by_id(env) -> dict[str, dict]:
    return {chunk["chunk_id"]: chunk for chunk in search.build_source_chunks(env.src)}


def test_search_after_incremental_update_excludes_deleted_and_old_text(env, monkeypatch):
    """差分更新後のEmbedding検索に、削除した資料・変更前の本文が出ず、引用位置が現行資料と一致する。"""
    env.write("a.md", "# 見出しA\n旧本文キーワード甲\n")
    env.write("b.md", "# 見出しB\n削除予定キーワード乙\n")
    env.build()

    env.write("a.md", "# 前置き\n追加した前置き行\n\n# 見出しA\n新本文キーワード丙\n")
    (env.src / "b.md").unlink()
    env.build()

    monkeypatch.setattr(search, "_get_embeddings", _fake_query_embeddings)
    cache = search.load_embed_cache()
    index = search._build_embed_index(cache)
    current = _current_chunks_by_id(env)
    queries = [
        "# 見出しA\n旧本文キーワード甲",
        "# 見出しB\n削除予定キーワード乙",
        current["a.md#0002"]["text"],
    ]
    results = search.embedding_search_multi(queries, MODEL, index, top_k=10)

    returned = [match for matches in results.values() for match in matches]
    assert returned, "現行資料のentryは検索できる"
    for match in returned:
        assert match["path"] == "a.md"
        assert "旧本文" not in match["snippet"] and "削除予定" not in match["snippet"]
        chunk = current[match["chunk_id"]]
        assert (match["start_line"], match["end_line"], match["heading"]) == (
            chunk["start_line"],
            chunk["end_line"],
            chunk["heading"],
        )
    best = results[current["a.md#0002"]["text"]][0]
    assert best["chunk_id"] == "a.md#0002"
    assert (best["start_line"], best["heading"]) == (4, "見出しA")


def test_pipeline_uses_updated_index_and_falls_back_to_keyword_when_stale(env, monkeypatch):
    env.write("a.md", "# 手順\n差分更新で変更した本文\n")
    env.write("b.md", "# 別資料\n削除される資料の本文\n")
    env.build()
    (env.src / "b.md").unlink()
    env.build()
    monkeypatch.setattr(search, "_get_embeddings", _fake_query_embeddings)
    monkeypatch.setattr(search, "detect_embed_model", lambda: MODEL)
    monkeypatch.setattr(
        search, "_is_model_available", lambda *_a, **_k: search.ModelStatus.AVAILABLE
    )

    ready = search.run_retrieval_pipeline("差分更新で変更した本文", model="m", mode="search")

    assert ready.route == "hybrid"
    assert ready.matches and all(m["path"] == "a.md" for m in ready.matches)

    env.write("a.md", "# 手順\n再構築前に外部で編集した本文\n")
    search._invalidate_source_chunk_memo()
    stale = search.run_retrieval_pipeline("外部で編集した本文", model="m", mode="search")

    assert stale.route == "keyword"
    assert "stale" in stale.route_reason
    assert all("差分更新で変更した本文" not in (m.get("snippet") or "") for m in stale.matches)


# --- 停止・障害（昇格直後・状態保存失敗） ---------------------------------------


def test_stop_right_after_promotion_is_recovered_on_restart(env, monkeypatch):
    """置換直後・checkpoint整理前にプロセスが止まっても、再起動時に昇格済みcacheから状態を再整合する。"""
    _changed_sources(env)
    coordinator = index_service.IndexCoordinator()
    before_generation = env.cache()["generation"]
    coordinator._persist(
        {
            "state": "building",
            "job_id": "job-stopped",
            "_allow_restart": True,
            "_allow_generation_change": True,
            "generation": before_generation,
            "embed_model": MODEL,
            "mode": "incremental",
            "total": 2,
            "processed": 0,
        }
    )

    class ProcessStopped(BaseException):
        pass

    real_remove = search._remove_checkpoint_files

    def stop_after_replace(*, include_state=False):
        # 置換で昇格済みになった後の checkpoint 整理だけを「停止」させる
        # （構築開始時の整理呼び出しは通常どおり実行する）。
        if include_state and env.cache()["generation"] != before_generation:
            raise ProcessStopped("stopped right after promotion")
        return real_remove(include_state=include_state)

    monkeypatch.setattr(search, "_remove_checkpoint_files", stop_after_replace)
    with pytest.raises(ProcessStopped):
        env.build()
    monkeypatch.setattr(search, "_remove_checkpoint_files", real_remove)

    assert env.cache()["generation"] != before_generation, "検証済み候補は昇格済み"
    assert search.get_embed_index_status(MODEL)["state"] == "ready"
    assert search.load_index_status()["state"] == "building", "状態ファイルは古いまま"

    restarted = index_service.IndexCoordinator()
    restarted.reconcile_orphaned_state()

    assert search.load_index_status()["state"] == "ready"
    assert search.get_embed_index_status(MODEL)["state"] == "ready"
    env.build()
    assert env.calls == [], "昇格済みcacheを再計算しない"
    assert not list(env.checkpoint_path.glob("batch-*.json"))


def test_status_save_failure_after_promotion_is_reported_and_recoverable(env, monkeypatch):
    """昇格後の状態保存に失敗しても、検証済みcacheは保持され、失敗を記録し、次回更新で再計算しない。"""
    _changed_sources(env)
    real_save = index_service.save_index_status

    def fail_ready_status(status):
        if status.get("state") == "ready":
            return False
        return real_save(status)

    monkeypatch.setattr(index_service, "save_index_status", fail_ready_status)
    coordinator = index_service.IndexCoordinator()

    assert coordinator.run_blocking() == 1

    persisted = search.load_index_status()
    assert persisted["state"] == "failed"
    assert persisted["error_code"] == "INDEX_STATUS_PERSIST_FAILED"
    assert search.get_embed_index_status(MODEL)["state"] == "ready"
    assert "本文Aを変更" in {entry["text"] for entry in env.cache()["entries"].values()}

    monkeypatch.setattr(index_service, "save_index_status", real_save)
    env.calls.clear()
    assert coordinator.run_blocking() == 0
    assert env.calls == []
    assert search.load_index_status()["state"] == "ready"
