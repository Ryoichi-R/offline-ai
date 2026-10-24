"""Web UI（`web/index.html`）の検索単位timeout入力・全体残り時間表示のテスト（Phase 7）。

対象:
- timeout入力欄（300〜600秒・step 30・既定300）とbudget表示領域の存在
- `budgetCountdown.computeRemainingSeconds` を Node.js 上で評価し、
  300→299 の単調減少、背景停止相当の複数秒経過、0下限、再同期をDOM非依存で検証する
- `budget` イベント受信・全終了経路（done/error/onError/onEnd/中止）での
  タイマー解除がソース上に存在すること
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

OFFLINE_AI = Path(__file__).resolve().parents[2]
INDEX_HTML = OFFLINE_AI / "_internal" / "web" / "index.html"

_BUDGET_BLOCK_RE = re.compile(
    r"// --- budgetCountdown: BEGIN.*?\n(.*?)// --- budgetCountdown: END ---",
    re.DOTALL,
)


def _html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def _extract_budget_countdown_source() -> str:
    match = _BUDGET_BLOCK_RE.search(_html())
    assert match, "budgetCountdown block markers not found in index.html"
    return match.group(1)


def _run_node(js_tail: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available in this environment")
    script = _extract_budget_countdown_source() + "\n" + js_tail
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


# --- 静的構造の存在確認 -------------------------------------------------------


def test_timeout_input_has_300_to_600_contract_with_default_300():
    html = _html()
    assert 'id="searchTimeoutInput"' in html
    assert 'min="300"' in html
    assert 'max="600"' in html
    assert 'step="30"' in html
    assert 'value="300"' in html


def test_budget_display_area_exists():
    assert 'id="searchBudget"' in _html()
    assert ".budget-msg" in _html()


def test_budget_label_says_zentai_zanri_to_distinguish_from_stall():
    html = _html()
    assert "全体残り" in html


def test_search_sse_forwards_timeout_seconds_query_param():
    html = _html()
    assert 'params.set("timeout_seconds"' in html
    assert "searchSSE(query, mode, reasoning, timeoutSeconds, callbacks)" in html


def test_health_response_syncs_timeout_input_bounds():
    html = _html()
    assert "h.searchTimeoutMin" in html
    assert "h.searchTimeoutMax" in html
    assert "h.searchTimeoutDefault" in html


def test_all_termination_paths_clear_budget_timer():
    html = _html()
    # done / error(SSE) / onError(fetch error) / onEnd / 利用者中止(cancelSearch)
    occurrences = html.count("this.clearBudgetTimer();")
    assert occurrences >= 5, (
        f"expected clearBudgetTimer() on done/error/onError/onEnd/cancelSearch, "
        f"found {occurrences} call sites"
    )


def test_terminal_event_finishes_once_and_old_search_cannot_reset_new_search():
    html = _html()
    assert "const searchId = ++state.searchSequence" in html
    assert "state.activeSearchId !== searchId" in html
    assert "state.finishSearch = finishSearch" in html
    assert html.count("finishSearch();") >= 3  # done / error / fetch end or cancel


# --- budgetCountdown ロジック（Node.js上でDOM非依存に評価） -------------------


def test_budget_countdown_counts_down_by_one_after_one_second():
    result = _run_node(
        "console.log(JSON.stringify(budgetCountdown.computeRemainingSeconds(300, 300, 0, 1000)));"
    )
    assert result == 299


def test_budget_countdown_no_drift_at_start():
    result = _run_node(
        "console.log(JSON.stringify(budgetCountdown.computeRemainingSeconds(300, 300, 0, 0)));"
    )
    assert result == 300


def test_budget_countdown_handles_background_suppression_multi_second_jump():
    # タブがバックグラウンドに置かれた後にタイマーが遅延して発火するケースを模す。
    # 単純な1ずつの減算ではなく経過時間から再計算するため、複数秒の飛びに対応する。
    result = _run_node(
        "console.log(JSON.stringify(budgetCountdown.computeRemainingSeconds(300, 250, 0, 45000)));"
    )
    assert result == 205


def test_budget_countdown_floors_at_zero_and_never_negative():
    result = _run_node(
        "console.log(JSON.stringify(budgetCountdown.computeRemainingSeconds(300, 5, 0, 30000)));"
    )
    assert result == 0


def test_budget_countdown_resyncs_from_new_subscription_snapshot():
    # 再購読でサーバーから新しいbudgetを受け取った場合、そのremainingSecondsと
    # 受信時刻を起点に再計算する（前回の起点を引きずらない）。
    resynced = _run_node(
        "console.log(JSON.stringify("
        "budgetCountdown.computeRemainingSeconds(300, 120, 5000, 6000)"
        "));"
    )
    assert resynced == 119


def test_budget_countdown_never_exceeds_timeout_seconds():
    result = _run_node(
        "console.log(JSON.stringify(budgetCountdown.computeRemainingSeconds(300, 300, 1000, 0)));"
    )
    assert result == 300


# --- deep診断(全体経過・段階別時間)の画面/保存表示 ------------------------------
# 2026-09-23の実資料受入で、保存Markdownと画面に段階別時間が出ず、時間の内訳を
# 確認できなかった。サーバーのdiagnosticsから数値だけを整形して両方へ出す。

_DIAG_BLOCK_RE = re.compile(
    r"// --- deepDiagnosticsFormat: BEGIN.*?\n(.*?)// --- deepDiagnosticsFormat: END ---",
    re.DOTALL,
)


def _run_diag_node(diag) -> list:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available in this environment")
    match = _DIAG_BLOCK_RE.search(_html())
    assert match, "deepDiagnosticsFormat block markers not found in index.html"
    script = match.group(1) + f"\nconsole.log(JSON.stringify(deepDiagnosticsFormat.lines({json.dumps(diag)})));"
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=15, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


def test_deep_diagnostics_lines_show_elapsed_stage_times_and_counts():
    lines = _run_diag_node(
        {
            "timeout_seconds": 300,
            "elapsed_seconds": 175.047,
            "stage_seconds": {"read": 150.2, "planning": 0.01, "search": 9.5, "integrate": 15.3},
            "rounds": 1,
            "documents": 2,
            "units": 3,
            "evidence_count": 1,
            "ledger_counts": {"candidate_limit": 18239},
        }
    )
    assert lines == [
        "全体経過（サーバー計測）: 175.0秒 / 上限300秒",
        "段階別時間: 計画 0.0秒 / 検索 9.5秒 / 読取（抽出・照合） 150.2秒 / 統合（最終回答・照合） 15.3秒",
        "処理件数: 巡回 1 / 資料 2 / 読取単位 3 / 根拠 1",
    ]


def test_deep_diagnostics_lines_show_final_answer_check_counts():
    lines = _run_diag_node(
        {
            "final_answer": {
                "result": "accepted",
                "attempts": [
                    {"normalized_citations": 1, "dropped_unconfirmed_lines": 3, "uncited_lines": 2, "unknown_id_lines": 0},
                    {"normalized_citations": 0, "dropped_unconfirmed_lines": 0, "uncited_lines": 0, "unknown_id_lines": 0},
                ],
            }
        }
    )
    assert lines == [
        "最終回答の検査: 合格 / 試行2回 / 根拠IDなしの行 0 / 不明な根拠ID 0 / 除いた未確認欄 3行 / 表記を直した引用 1件 / 限定の欠け 0行"
    ]


def test_deep_diagnostics_list_units_read_but_judged_irrelevant():
    ranges = [{"path": f"notice/doc-{i}.md", "start_line": 10 * i + 1, "end_line": 10 * i + 5} for i in range(7)]
    lines = _run_diag_node({"irrelevant_ranges": ranges})
    assert lines == [
        "読んだが質問と関係なしと判定した節: 7件（doc-0.md L1-5 / doc-1.md L11-15 / doc-2.md L21-25"
        " / doc-3.md L31-35 / doc-4.md L41-45 ほか2件）"
    ]


def test_deep_diagnostics_show_repaired_lines_only_when_present():
    lines = _run_diag_node(
        {"final_answer": {"result": "accepted", "repaired_lines": 2, "attempts": [{"limitation_lines": 2}, {"limitation_lines": 2}]}}
    )
    assert lines[0].endswith("限定の欠け 2行 / 原文に置き換えた行 2")


def test_deep_diagnostics_show_enumeration_gaps_and_added_items_only_when_present():
    lines = _run_diag_node(
        {
            "final_answer": {
                "result": "accepted",
                "added_items": 1,
                "attempts": [{"limitation_lines": 0, "enumeration_gaps": 2}, {"limitation_lines": 0, "enumeration_gaps": 1}],
            }
        }
    )
    assert lines[0].endswith("限定の欠け 0行 / 列挙の欠け 1項目 / 原文から補った列挙項目 1")

def test_deep_diagnostics_show_duplicate_lines_only_when_present():
    lines = _run_diag_node(
        {"final_answer": {"result": "accepted", "duplicate_lines": 2, "attempts": [{"limitation_lines": 0}]}}
    )
    assert lines[0].endswith("限定の欠け 0行 / 重複を除いた行 2")


def test_deep_diagnostics_show_line_by_line_check_counts():
    lines = _run_diag_node(
        {
            "final_answer": {
                "result": "accepted",
                "attempts": [{"limitation_lines": 0}],
                "line_checks": {"checked": 12, "unsupported": 2, "replaced": 1, "dropped": 1, "unchecked": 0},
            }
        }
    )
    assert lines[1] == "1行ずつの照合: 照合 12行 / 支持されない 2行（原文に置換 1・削除 1） / 未照合 0行"


def test_deep_diagnostics_show_background_lines_dropped_only_when_present():
    lines = _run_diag_node(
        {
            "final_answer": {
                "result": "accepted",
                "attempts": [{"limitation_lines": 0}],
                "line_checks": {"checked": 12, "unsupported": 2, "replaced": 0, "dropped": 2, "off_question": 1, "unchecked": 0},
            }
        }
    )
    assert lines[1] == "1行ずつの照合: 照合 12行 / 支持されない 2行（原文に置換 0・削除 2、うち質問外の背景 1） / 未照合 0行"

def test_deep_diagnostics_label_limitation_rejection():
    lines = _run_diag_node(
        {"final_answer": {"result": "rejected_limitations", "attempts": [{"limitation_lines": 2}]}}
    )
    assert lines[0].startswith("最終回答の検査: 根拠の限定の欠けで不合格 / 試行1回")
    assert lines[0].endswith("限定の欠け 2行")


@pytest.mark.parametrize("final_answer", [{"attempts": []}, {"result": "accepted"}, "x"])
def test_deep_diagnostics_skip_final_answer_line_without_attempts(final_answer):
    assert _run_diag_node({"final_answer": final_answer}) == []


@pytest.mark.parametrize("diag", [None, {}, "text", {"stage_seconds": {}}])
def test_deep_diagnostics_lines_tolerate_missing_diagnostics(diag):
    assert _run_diag_node(diag) == []


def test_deep_diagnostics_are_rendered_on_screen_and_in_saved_markdown():
    html = _html()
    assert 'const diagnostics = ev.route === "deep" ? deepDiagnosticsFormat.lines(ev.deepDiagnostics) : [];' in html
    assert "deepDiagnosticsFormat.lines(evidence.deepDiagnostics).map((line) => `- ${this._markdownMeta(line)}`)" in html


# Execute the production storage, transport and UI methods together. These tests
# use no live browser, corpus, Ollama, or network, and do not claim Chrome acceptance.
_RECONNECT_JS = r"""
const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
function extract(start, end) {
    const a = input.html.indexOf(start);
    const b = input.html.indexOf(end, a + start.length);
    assert(a >= 0 && b > a, start);
    return input.html.slice(a, b);
}
const storageCode = extract('const ACTIVE_SEARCH_STORAGE_KEY', '// --- budgetCountdown ---');
const apiCode = extract('    searchSSE(query,', '\n};\n\n// --- pagePresence ---');
const startCode = extract('    startSearch(resume = null)', '    startBudgetDisplay(');
const resumeCode = extract('    resumeActiveSearch() {', '    renderPageCloseHint(');
const cancelCode = extract('    async cancelSearch() {', '\n\n};');
const modeCode = extract('    updateSearchModeUI() {', '    startSearch(resume = null)');
const key = 'offlineai.activeSearch';
const flush = () => new Promise(resolve => setImmediate(resolve));
function page(storage, behavior, storageFails = false) {
    const requests = [];
    const reads = [];
    const els = new Proxy({}, {get(o, k) {
        return o[k] ||= {value:'', classList:{remove(){}}, replaceChildren(){}, focus(){}};
    }});
    Object.assign(els.searchInput, {value:'synthetic question'});
    els.searchModeSelect.value = 'deep';
    els.reasoningSelect.value = 'low';
    els.searchTimeoutInput.value = '300';
    const context = vm.createContext({
        state: {searching:false, searchSequence:0, features:{search:true,retrieval:true},
            deepTimeoutMin:300,deepTimeoutMax:1800,deepTimeoutDefault:1800},
        sessionStorage: {
            getItem:k => storage.get(k) ?? null,
            setItem(k,v){if(storageFails) throw Error('denied'); storage.set(k,v);},
            removeItem:k => storage.delete(k),
        },
        crypto: {randomUUID:()=>'synthetic-id'},
        AbortController, URLSearchParams, TextDecoder, setTimeout, clearTimeout,
        md:{render:t=>t}, els,
        fetch:async (url, options) => {
            requests.push({url,options});
            if (url === '/api/search/cancel') return behavior.cancel();
            if (behavior.kind === 'network_error' || behavior.kind === 'abort') {
                const e = Error('synthetic disconnect');
                e.name = behavior.kind === 'abort' ? 'AbortError' : 'TypeError';
                throw e;
            }
            if (behavior.status) return {ok:false,status:behavior.status,json:async()=>({error:{message:'denied'}})};
            let sent = false;
            return {ok:true,body:{getReader:()=>({read:async()=>{
                if(behavior.kind === 'pending') return new Promise(resolve=>reads.push(resolve));
                if(behavior.kind === 'eof' || sent) return {done:true};
                sent = true;
                return {done:false,value:new TextEncoder().encode('data: '+JSON.stringify(behavior.event)+'\n\n')};
            }})}};
        },
    });
    vm.runInContext(storageCode + '\nconst api = {' + apiCode + '};\nconst ui = {' +
        startCode + resumeCode + cancelCode + modeCode + `
        els, clearBudgetTimer(){}, closeEvidenceViewer(){},pollHealthUpdate(){},updateSaveButton(){}
    };`, context);
    return {requests,reads,els,context,run:code=>vm.runInContext(code,context)};
}
(async()=>{
    const storage = new Map();
    const scenario = input.scenario;
    const behavior = {kind:'pending',cancel:async()=>({ok:true,json:async()=>({cancelled:true})})};
    const p = page(storage, behavior, scenario === 'storage_denied');
    if (['eof','network_error','abort'].includes(scenario)) behavior.kind = scenario;
    if (scenario === 'done' || scenario === 'terminal_error') {
        behavior.kind = 'event';
        behavior.event = scenario === 'done' ? {type:'done',status:'partial',stopReason:'time_budget'} : {type:'error',message:'terminal'};
    }
    p.run('ui.startSearch()');
    assert.equal(p.requests.length,1);
    if(scenario !== 'storage_denied') assert(storage.has(key));
    await flush();
    if(['eof','network_error','abort'].includes(scenario)) {
        assert(storage.has(key));
        assert(p.run('state.searchDisconnected'));
        assert.equal(p.els.searchBtn.hidden,true);
        assert.equal(p.els.reconnectSearchBtn.hidden,false);
        p.run('ui.startSearch()');
        assert.equal(p.requests.length,1, 'disconnected job must block a new search');
        const saved = storage.get(key);
        behavior.kind = 'pending';
        const reloaded = page(storage,behavior);
        reloaded.run('ui.resumeActiveSearch(); ui.resumeActiveSearch();');
        assert.equal(reloaded.requests.length,1);
        const params = new URL(reloaded.requests[0].url,'http://local').searchParams;
        const original = new URL(p.requests[0].url,'http://local').searchParams;
        for(const field of ['request_id','q','mode','reasoning','timeout_seconds']) assert.equal(params.get(field),original.get(field));
        assert.equal(params.get('resume_only'),'1');
        assert.equal(storage.get(key),saved);
        assert.equal(reloaded.els.searchInput.value,'synthetic question');
        assert.equal(reloaded.els.searchModeSelect.value,'deep');
    } else if(scenario === 'done' || scenario === 'terminal_error') {
        assert(!storage.has(key));
        assert.equal(p.run('state.searching'),false);
        assert.equal(p.els.reconnectSearchBtn.hidden,true);
    } else if(scenario.startsWith('http_')) {
        const status = Number(scenario.slice(5));
        const resumed = page(storage,{kind:'pending',status});
        resumed.run('ui.resumeActiveSearch()');
        await flush();
        const transient = [408,429,503].includes(status);
        assert.equal(storage.has(key),transient);
        assert.equal(resumed.run('state.searching'),transient);
        assert.equal(resumed.requests.length,1,'no automatic retry');
    } else if(scenario === 'cancel_success' || scenario === 'cancel_failed') {
        let acknowledge;
        behavior.cancel = ()=>new Promise((resolve,reject)=>{acknowledge=scenario==='cancel_success'?resolve:reject;});
        const cancellation = p.run('ui.cancelSearch()');
        p.run('ui.cancelSearch(); ui.resumeActiveSearch(); ui.startSearch();');
        assert(storage.has(key),'unacknowledged cancellation must retain resume state');
        assert.equal(p.requests.length,2,'only one cancel, no new search');
        if(scenario==='cancel_success') acknowledge({ok:true,json:async()=>({cancelled:true})});
        else acknowledge(Error('offline'));
        await cancellation;
        assert.equal(storage.has(key),scenario==='cancel_failed');
        assert.equal(p.run('state.searching'),scenario==='cancel_failed');
        assert.equal(p.run('state.cancelPending'),false);
    } else if(scenario === 'stale_end') {
        // Old subscription completes after a retry started. It must not clear the new UI.
        p.run('state.searchDisconnected=true; ui.resumeActiveSearch();');
        await flush();
        assert.equal(p.requests.length,2);
        p.reads[0]({done:true});
        await flush();
        assert(storage.has(key));
        assert.equal(p.run('state.searchDisconnected'),false);
        assert.equal(p.run('state.searching'),true);
    } else if(scenario === 'cancel_done_race') {
        let rejectCancel;
        behavior.cancel=()=>new Promise((resolve,reject)=>{rejectCancel=reject;});
        const cancelling=p.run('ui.cancelSearch()');
        p.reads[0]({done:false,value:new TextEncoder().encode('data: {"type":"done"}\n\n')});
        await flush();
        p.run('ui.startSearch()');
        const current=storage.get(key);
        rejectCancel(Error('late cancel error'));
        await cancelling;
        assert.equal(storage.get(key),current);
        assert.equal(p.run('state.searching'),true);
        assert.equal(p.els.searchError.hidden,true);
    } else if(scenario === 'storage_denied') {
        assert(!storage.has(key));
        assert.equal(p.els.searchWarning.hidden,false);
        p.run('state.searchDisconnected=true; ui.resumeActiveSearch();');
        assert.equal(p.requests.length,2);
        assert(new URL(p.requests[1].url,'http://local').searchParams.has('resume_only'));
    }
    console.log(JSON.stringify({scenario,passed:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
"""


@pytest.mark.parametrize("scenario", [
    "eof", "network_error", "abort", "done", "terminal_error",
    "http_400", "http_403", "http_404", "http_409",
    "http_408", "http_429", "http_503",
    "cancel_success", "cancel_failed", "stale_end", "cancel_done_race", "storage_denied",
])
def test_search_resume_lifecycle(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available in this environment")
    result = subprocess.run(
        [node, "-e", _RECONNECT_JS],
        input=json.dumps({"html": _html(), "scenario": scenario}),
        capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"]
