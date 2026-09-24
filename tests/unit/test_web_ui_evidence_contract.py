"""根拠ビューア・検索専用・Markdown保存のUI契約。"""

from pathlib import Path
import shutil
import subprocess

import pytest


INDEX_HTML = Path(__file__).resolve().parents[2] / "_internal" / "web" / "index.html"


def test_ui_contains_search_only_viewer_and_fixed_result_save_contract():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'id="searchModeSelect"' in html
    assert 'value="search">資料を探す' in html
    assert 'id="evidenceViewer"' in html
    assert "fetch(`/api/evidence/view?" in html
    assert 'text.textContent = String(line.text ?? "")' in html
    assert 'id="saveMarkdownBtn" type="button" hidden>Markdownで保存' in html
    assert 'new Blob([lines.join("\\n")], { type: "text/markdown;charset=utf-8" })' in html
    assert "state.resultMeta = JSON.parse(JSON.stringify(ev))" in html
    assert "state.resultEvidence = JSON.parse(JSON.stringify(ev))" in html


def test_markdown_ui_uses_content_dependent_fence_and_does_not_save_view_ids():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "const longest = runs.reduce" in html
    assert 'this._markdownTextBlock("検索時の抜粋", item.snippet)' in html
    save_start = html.index("    saveMarkdown()")
    save_end = html.index("    async cancelSearch()", save_start)
    assert "item.evidenceId" not in html[save_start:save_end]
    assert "document.body.appendChild(anchor)" in html


def test_index_ui_exposes_incremental_preview_and_explicit_full_rebuild():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'id="indexFullBtn"' in html
    assert 'fetch(`/api/index/plan?mode=${encodeURIComponent(mode)}`)' in html
    assert "expected_generation" in html
    assert "再計算${plan.generated_chunks}チャンク・再利用${plan.reused_chunks}チャンク" in html
    assert "全件再構築" in html


def test_evidence_list_and_markdown_mark_expanded_and_partial_evidence():
    """親子展開の由来と部分展開を、根拠一覧とMarkdown保存の両方に表示する。"""
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert (
        'return item.groupPartial ? "見出し配下の本文を展開（部分展開）" : "見出し配下の本文を展開";'
        in html
    )
    render_start = html.index("    renderEvidence(ev) {")
    render_end = html.index("\n    },", render_start)
    assert "if (item.groupId) metaParts.push(this._expansionLabel(item));" in html[render_start:render_end]
    save_start = html.index("    saveMarkdown()")
    save_end = html.index("    async cancelSearch()", save_start)
    assert 'item.groupId ? this._expansionLabel(item) : ""' in html[save_start:save_end]


@pytest.mark.parametrize("transition", ["new-view", "close", "new-search"])
@pytest.mark.parametrize("stale_error", [False, True])
def test_late_view_response_cannot_replace_current_view_or_focus(transition, stale_error):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dynamic UI contract tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[1], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const transition = process.argv[2];
const staleError = process.argv[3] === 'true';
let focused = null;
function element() {
    return {
        hidden: true, textContent: '', children: [], isConnected: true,
        classList: {add() {}},
        replaceChildren() { this.children = []; },
        append(...items) { this.children.push(...items); },
        appendChild(item) { this.children.push(item); },
        focus() { focused = this; },
    };
}
const context = {document: {addEventListener() {}, createElement: element}};
vm.createContext(context);
vm.runInContext(script + '\nglobalThis.fixture = {ui, state, api};', context);
const {ui, state, api} = context.fixture;
ui.els = Object.fromEntries([
    'evidenceViewer', 'evidenceViewerTitle', 'evidenceViewerLines',
    'evidenceViewerNote', 'evidenceViewerCloseBtn',
].map(key => [key, element()]));
const pending = new Map();
api.viewEvidence = id => new Promise((resolve, reject) => pending.set(id, {resolve, reject}));
const data = number => ({
    lines: [{lineNumber: number, text: `body-${number}`, highlighted: true}],
    highlightStartLine: number, highlightEndLine: number,
});
function snapshot() {
    return JSON.stringify({
        title: ui.els.evidenceViewerTitle.textContent,
        note: ui.els.evidenceViewerNote.textContent,
        lines: ui.els.evidenceViewerLines.children,
        hidden: ui.els.evidenceViewer.hidden,
    });
}
(async () => {
    state.activeSearchId = 1;
    const firstButton = element();
    const oldRequest = ui.openEvidence({evidenceId: 'old', sourceTitle: 'old'}, firstButton);
    let secondButton;
    if (transition === 'close') {
        ui.closeEvidenceViewer();
        assert.equal(focused, firstButton);
    } else {
        if (transition === 'new-search') {
            state.activeSearchId = 2;
            ui.closeEvidenceViewer(false);
        }
        secondButton = element();
        const currentRequest = ui.openEvidence({evidenceId: 'current', sourceTitle: 'current'}, secondButton);
        pending.get('current').resolve(data(2));
        await currentRequest;
        assert.equal(ui.els.evidenceViewerTitle.textContent, 'current');
        assert.equal(ui.els.evidenceViewerLines.children[0].children[1].textContent, 'body-2');
        assert.equal(focused, ui.els.evidenceViewerCloseBtn);
    }
    const before = snapshot();
    const previousFocus = focused;
    if (staleError) pending.get('old').reject(new Error('stale failure'));
    else pending.get('old').resolve(data(1));
    await oldRequest;
    assert.equal(snapshot(), before);
    assert.equal(focused, previousFocus);
    if (secondButton) {
        ui.closeEvidenceViewer();
        assert.equal(focused, secondButton);
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", script, str(INDEX_HTML), transition, str(stale_error).lower()],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
