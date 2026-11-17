"""確認案内のUI契約（本番の保存・通信・画面メソッドをNodeで実行。実ブラウザ受入ではない）。"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX_HTML = Path(__file__).resolve().parents[2] / "_internal" / "web" / "index.html"

_JS = r"""
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
const modeCode = extract('    updateSearchModeUI() {', '    startSearch(resume = null)');
const flush = () => new Promise(resolve => setImmediate(resolve));
function element() {
    const el = {value: '', hidden: false, disabled: false, textContent: '', className: '', children: [], listeners: {},
        classList: {remove() {}, add() {}}, focus() {},
        replaceChildren(...items) { this.children = items; },
        append(...items) { this.children.push(...items); },
        appendChild(item) { this.children.push(item); },
        addEventListener(type, fn) { this.listeners[type] = fn; },
        click() { this.listeners.click && this.listeners.click(); }};
    return el;
}
function page(storage, responses) {
    const requests = [];
    const els = new Proxy({}, {get(o, k) { return o[k] ||= element(); }});
    els.searchInput.value = '会計事務における押印の取扱いの内容詳細';
    els.searchModeSelect.value = 'deep';
    els.reasoningSelect.value = 'low';
    els.searchTimeoutInput.value = '300';
    const context = vm.createContext({
        state: {searching: false, searchSequence: 0, features: {search: true, retrieval: true},
            deepTimeoutMin: 300, deepTimeoutMax: 1800, deepTimeoutDefault: 1800},
        sessionStorage: {getItem: k => storage.get(k) ?? null, setItem(k, v) { storage.set(k, v); }, removeItem: k => storage.delete(k)},
        crypto: {randomUUID: () => 'id-' + (requests.length + 1)},
        document: {createElement: element},
        AbortController, URLSearchParams, TextDecoder, setTimeout, clearTimeout,
        md: {render: t => t}, els,
        fetch: async (url, options) => {
            requests.push({url, options});
            const events = responses.shift() || [];
            let index = 0;
            return {ok: true, body: {getReader: () => ({read: async () => {
                if (index >= events.length) return {done: true};
                return {done: false, value: new TextEncoder().encode('data: ' + JSON.stringify(events[index++]) + '\n\n')};
            }})}};
        },
    });
    vm.runInContext(storageCode + '\nconst api = {' + apiCode + '};\nconst ui = {' +
        startCode + resumeCode + modeCode + `
        els, clearBudgetTimer(){}, closeEvidenceViewer(){}, pollHealthUpdate(){}, updateSaveButton(){}, renderAnswerQuality(){}
    };`, context);
    return {requests, els, run: code => vm.runInContext(code, context)};
}
const clarification = {type: 'clarification', kind: 'topics', confirmationId: 'c'.repeat(24), state: 'document_fixed',
    optionsVersion: 3, message: '', document: {path: 'notice.md', title: '押印の取扱い'},
    points: [{kind: 'excerpt', heading: '1 趣旨', startLine: 5, endLine: 6, text: '押印を求めない。'}],
    options: [{kind: 'section', label: '1 趣旨', startLine: 5, endLine: 6}, {kind: 'overview', label: '全体をまとめて'}],
    moreSections: [], topic: {section: null, targets: [], aspects: []}};
(async () => {
    const out = {};
    if (input.scenario === 'clarify_then_select') {
        const storage = new Map();
        const p = page(storage, [[clarification, {type: 'done', stage: 'clarify'}], []]);
        p.run('ui.startSearch()');
        await flush(); await flush();
        out.panelHidden = p.els.guidedPanel.hidden;
        out.warning = p.els.searchWarning.textContent;
        out.result = p.els.searchResult.textContent || '';
        out.saveHidden = p.els.saveMarkdownBtn.hidden;
        out.searching = p.run('state.searching');
        out.guidedId = storage.get('offlineai.guidedConfirmation');
        out.optionLabels = p.els.guidedOptions.children.map(c => c.textContent);
        out.pointText = p.els.guidedPoints.children[0].children[0].textContent;
        p.els.guidedOptions.children[0].click();
        await flush();
        out.post = {url: p.requests[1].url, method: p.requests[1].options.method, body: JSON.parse(p.requests[1].options.body)};
        out.saved = JSON.parse(storage.get('offlineai.activeSearch'));
        // 再読み込み: 同じJSONに resume_only を付けて再購読する。
        const reloaded = page(storage, [[]]);
        reloaded.run('ui.resumeActiveSearch()');
        out.resume = JSON.parse(reloaded.requests[0].options.body);
    } else if (input.scenario === 'long_follow_up') {
        const p = page(new Map(), []);
        p.run(`state.guided = ${JSON.stringify(clarification)}`);
        p.els.guidedInput.value = 'あ'.repeat(1001);
        p.run('ui.sendGuidedText()');
        out.requests = p.requests.length;
        out.error = p.els.searchError.textContent;
        out.inputKept = p.els.guidedInput.value.length;
    } else if (input.scenario === 'legacy_done') {
        const p = page(new Map(), [[{type: 'chunk', text: '回答'}, {type: 'done'}]]);
        p.run('ui.startSearch()');
        await flush(); await flush();
        out.warning = p.els.searchWarning.textContent;
        out.panelHidden = p.els.guidedPanel.hidden;
    }
    console.log(JSON.stringify(out));
})().catch(error => { console.error(error); process.exit(1); });
"""


def _run(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dynamic UI contract tests")
    result = subprocess.run(
        [node, "-e", _JS],
        input=json.dumps(
            {
                "html": INDEX_HTML.read_text(encoding="utf-8").replace("\r\n", "\n"),
                "scenario": scenario,
            }
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_clarification_shows_panel_without_answer_or_deep_status_and_posts_selection():
    out = _run("clarify_then_select")
    assert out["panelHidden"] is False
    assert out["warning"] == ""  # 「状態未確認 / 理由未確認」を出さない
    assert out["result"] == ""
    assert out["saveHidden"] is True
    assert out["searching"] is False
    assert out["guidedId"] == "c" * 24
    assert out["optionLabels"] == ["1. 1 趣旨", "2. 全体をまとめて"]
    assert out["pointText"] == "押印を求めない。"
    assert out["post"]["url"] == "/api/search" and out["post"]["method"] == "POST"
    body = out["post"]["body"]
    assert body["confirmationId"] == "c" * 24 and body["action"] == "select_topic"
    assert body["choice"] == 0 and body["optionsVersion"] == 3 and body["mode"] == "deep"
    assert "resume_only" not in body
    assert out["saved"]["guided"]["action"] == "select_topic"
    assert out["resume"]["resume_only"] is True
    for key in ("confirmationId", "action", "choice", "optionsVersion", "requestId", "mode"):
        assert out["resume"][key] == body[key]


def test_long_follow_up_is_rejected_on_screen_and_input_is_kept():
    out = _run("long_follow_up")
    assert out == {"requests": 0, "error": "入力は1000文字以内にしてください。", "inputKept": 1001}


def test_done_without_stage_keeps_the_existing_deep_display():
    out = _run("legacy_done")
    assert out["warning"] == "状態未確認 / 理由未確認"
    assert out["panelHidden"] is True
