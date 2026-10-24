"""Embeddingインデックス開始確認UIの動的テスト。

`_internal/web/index.html` の script を Node.js の vm 上で評価し、DOMとfetchを
fakeへ置き換えて、確認ダイアログの分岐と開始要求の内容を実行検証する。
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX_HTML = Path(__file__).resolve().parents[2] / "_internal" / "web" / "index.html"

_HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const scenario = JSON.parse(process.argv[2]);
function element() {
    return {hidden: false, disabled: false, textContent: '', addEventListener() {}};
}
const confirms = [];
const context = {
    document: {addEventListener() {}, getElementById: () => element(), createElement: element},
    window: {confirm(message) { confirms.push(message); return scenario.confirm; }},
    console,
};
vm.createContext(context);
vm.runInContext(script + '\nglobalThis.fixture = {ui, state, api};', context);
const {ui, state, api} = context.fixture;
ui.els = Object.fromEntries(
    ['indexStatus', 'indexStartBtn', 'indexFullBtn', 'indexCancelBtn'].map(key => [key, element()])
);
const calls = {plans: [], starts: []};
let releasePlan = null;
api.indexStatus = async () => scenario.status;
api.indexPlan = mode => {
    calls.plans.push(mode);
    const plan = {...scenario.plan, mode};
    if (!scenario.holdPlan) return Promise.resolve(plan);
    return new Promise(resolve => { releasePlan = () => resolve(plan); });
};
api.startIndex = async options => { calls.starts.push(options); return {state: 'building'}; };
api.indexSSE = () => ({abort() {}});
(async () => {
    const first = ui.startIndex(scenario.requestedMode);
    let duringStart = null;
    if (scenario.holdPlan) {
        await new Promise(resolve => setImmediate(resolve));
        duringStart = {
            startDisabled: ui.els.indexStartBtn.disabled,
            fullDisabled: ui.els.indexFullBtn.disabled,
        };
        await ui.startIndex(scenario.requestedMode);
        releasePlan();
    }
    await first;
    process.stdout.write(JSON.stringify({
        calls,
        confirms,
        duringStart,
        startingAfter: state.indexStarting,
        startDisabledAfter: ui.els.indexStartBtn.disabled,
    }));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""

_PLAN = {
    "generation": "g-preview",
    "added_files": 2,
    "changed_files": 1,
    "deleted_files": 1,
    "unchanged_files": 7,
    "generated_chunks": 30,
    "reused_chunks": 400,
    "checkpoint_chunks": 0,
    "estimated_seconds": 150,
    "estimate_status": "estimated",
    "reason": None,
}


def _run(**scenario_overrides):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dynamic UI contract tests")
    scenario = {
        "requestedMode": "incremental",
        "status": {"state": "ready"},
        "plan": dict(_PLAN),
        "confirm": True,
        "holdPlan": False,
    }
    scenario.update(scenario_overrides)
    result = subprocess.run(
        [node, "-e", _HARNESS, str(INDEX_HTML), json.dumps(scenario, ensure_ascii=False)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_cancelled_confirmation_sends_no_start_request():
    result = _run(confirm=False)

    assert result["calls"]["plans"] == ["incremental"]
    assert result["calls"]["starts"] == []
    assert result["startingAfter"] is False
    assert result["startDisabledAfter"] is False


def test_confirmation_shows_delta_counts_and_binds_preview_generation():
    result = _run()

    message = result["confirms"][0]
    assert "差分更新: 追加2件・変更1件・削除1件・未変更7件。" in message
    assert "再計算30チャンク・再利用400チャンク。" in message
    assert "約0時間3分" in message
    assert result["calls"]["starts"] == [
        {"resume": False, "mode": "incremental", "expectedGeneration": "g-preview"}
    ]


@pytest.mark.parametrize(
    ("plan_overrides", "expected_text"),
    [
        (
            {"estimated_seconds": None, "estimate_status": "no_rate"},
            "生成実績がないため、所要時間は見積もりできません",
        ),
        (
            {"generated_chunks": 0, "estimated_seconds": None, "estimate_status": "no_embedding"},
            "Embeddingの生成はありません。資料の走査と保存に時間がかかる場合があります。",
        ),
        ({"reason": "model_digest_unavailable"}, "digest）を取得できないため"),
        ({"reason": "cache_incompatible"}, "既存インデックスと異なるため"),
        ({"estimated_seconds": 3 * 3600 + 61}, "約3時間2分"),
    ],
)
def test_confirmation_explains_estimate_basis_and_rebuild_reason(plan_overrides, expected_text):
    result = _run(plan={**_PLAN, **plan_overrides})

    assert expected_text in result["confirms"][0]


def test_full_button_on_interrupted_incremental_job_starts_new_full_job():
    result = _run(requestedMode="full", status={"state": "failed", "mode": "incremental"})

    assert result["calls"]["plans"] == ["full"]
    assert result["confirms"][0].startswith("全件再構築:")
    assert result["calls"]["starts"] == [
        {"resume": False, "mode": "full", "expectedGeneration": "g-preview"}
    ]


@pytest.mark.parametrize("requested_mode", ["incremental", "full"])
def test_interrupted_full_job_is_resumed_with_its_saved_mode(requested_mode):
    result = _run(requestedMode=requested_mode, status={"state": "cancelled", "mode": "full"})

    assert result["calls"]["starts"] == [
        {"resume": True, "mode": "full", "expectedGeneration": "g-preview"}
    ]


def test_building_job_is_not_previewed_or_started_again():
    result = _run(status={"state": "building", "mode": "incremental"})

    assert result["calls"]["plans"] == []
    assert result["calls"]["starts"] == []


def test_repeated_click_while_preview_is_pending_starts_only_once():
    result = _run(holdPlan=True)

    assert result["duringStart"] == {"startDisabled": True, "fullDisabled": True}
    assert result["calls"]["plans"] == ["incremental"]
    assert len(result["calls"]["starts"]) == 1
    assert result["startDisabledAfter"] is False
