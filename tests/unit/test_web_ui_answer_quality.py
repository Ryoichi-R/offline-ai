"""回答品質(v2)のUI契約: 要確認の表示、画面と保存Markdownの一致、generatedを合格表示にしない。"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX_HTML = Path(__file__).resolve().parents[2] / "_internal" / "web" / "index.html"

HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[1], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const saved = [];
function element() {
    return {
        hidden: true, textContent: '', className: '', children: [], isConnected: true,
        classList: {add() {}, remove() {}},
        replaceChildren() { this.children = []; },
        append(...items) { this.children.push(...items); },
        appendChild(item) { this.children.push(item); },
        focus() {}, click() {}, remove() {},
    };
}
class FakeBlob { constructor(parts) { saved.push(parts.join('')); } }
const context = {
    document: {addEventListener() {}, createElement: element, body: {appendChild() {}}},
    Blob: FakeBlob, URL: {createObjectURL() { return 'blob:x'; }, revokeObjectURL() {}},
    setTimeout(fn) { return 0; }, performance: {now() { return 0; }},
};
vm.createContext(context);
vm.runInContext(script + '\nglobalThis.fixture = {ui, state, answerQualityFormat, deepDiagnosticsFormat, md};', context);
const {ui, state, answerQualityFormat, deepDiagnosticsFormat, md} = context.fixture;
ui.els = {answerQuality: element(), saveMarkdownHint: element()};
const out = (value) => console.log(JSON.stringify(value));
"""


def run_node(tail: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dynamic UI contract tests")
    result = subprocess.run(
        [node, "-e", HARNESS + tail, str(INDEX_HTML)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def quality(state, **extra):
    base = {
        "policyVersion": "v2",
        "state": state,
        "label": "検査済み" if state == "passed" else "要確認",
        "message": "原文・意味・質問への適合・文章の検査をすべて通過しました。正確性を保証するものではありません。"
        if state == "passed"
        else "一部の観点は根拠を確認できませんでした。確認できた範囲だけを回答しています。",
        "gates": {
            "G1": {"label": "原文・引用", "status": "passed", "codes": []},
            "G2": {
                "label": "意味の支持",
                "status": "passed" if state == "passed" else "uncertain",
                "codes": [],
            },
            "G3": {"label": "質問への適合", "status": "passed", "codes": []},
            "G4": {"label": "文章の構成", "status": "not_run", "codes": []},
        },
        "unconfirmedAspects": [] if state == "passed" else ["別の会議体の役割との区別"],
        "removedClaims": 0 if state == "passed" else 2,
        "attempts": 2,
    }
    base.update(extra)
    return base


def test_static_structure_declares_the_banner_events_and_markdown_hook():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="answerQuality"' in html and "quality-review" in html
    assert "state.resultAnswerQuality = JSON.parse(JSON.stringify(ev.answerQuality))" in html
    save_start = html.index("    saveMarkdown()")
    save_end = html.index("    async cancelSearch()", save_start)
    assert "answerQualityFormat.lines(state.resultAnswerQuality)" in html[save_start:save_end]
    # 画面は textContent で描画し、サーバー文字列をHTMLとして解釈しない。
    render_start = html.index("    renderAnswerQuality()")
    assert (
        "innerHTML" not in html[render_start : html.index("    renderEvidence(ev)", render_start)]
    )


def test_lines_show_reason_unconfirmed_scope_gates_and_removed_claims():
    lines = run_node(
        f"out(answerQualityFormat.lines({json.dumps(quality('partial'), ensure_ascii=False)}));"
    )
    assert lines[0].startswith("品質状態: 要確認（一部の観点は根拠を確認できませんでした")
    assert lines[1] == "未確認の範囲: 別の会議体の役割との区別"
    assert (
        lines[2]
        == "検査: 原文・引用 通過 / 意味の支持 判定不能 / 質問への適合 通過 / 文章の構成 未実施"
    )
    assert lines[3] == "検査を通らず除いた主張: 2件"
    corrected = run_node(
        f"out(answerQualityFormat.lines({json.dumps(quality('needs_review', correctedQuotes=3), ensure_ascii=False)}));"
    )
    assert corrected[-1] == "原文の範囲へ補正した引用: 3件"
    assert all("補正" not in line for line in lines)


def test_passed_is_shown_as_checked_but_never_as_a_guarantee_and_empty_is_hidden():
    lines = run_node(
        f"out(answerQualityFormat.lines({json.dumps(quality('passed'), ensure_ascii=False)}));"
    )
    assert lines[0].startswith("品質状態: 検査済み")
    assert "保証するものではありません" in lines[0] and "正確" not in lines[0].replace(
        "正確性を保証するものではありません", ""
    )
    assert run_node(
        "out([answerQualityFormat.lines(null), answerQualityFormat.lines({}), answerQualityFormat.needsReview(null)]);"
    ) == [[], [], False]


@pytest.mark.parametrize("state", ["passed", "partial", "needs_review", "failed"])
def test_banner_marks_everything_but_passed_as_needing_review(state):
    data = quality(state)
    result = run_node(
        f"""
state.resultAnswerQuality = {json.dumps(data, ensure_ascii=False)};
ui.renderAnswerQuality();
const shown = {{hidden: ui.els.answerQuality.hidden, cls: ui.els.answerQuality.className, text: ui.els.answerQuality.textContent}};
state.resultAnswerQuality = null;
ui.renderAnswerQuality();
out([shown, {{hidden: ui.els.answerQuality.hidden, text: ui.els.answerQuality.textContent}}]);
"""
    )
    shown, cleared = result
    assert shown["hidden"] is False
    assert ("quality-review" in shown["cls"]) == (state != "passed")
    assert ("要確認" in shown["text"]) == (state != "passed")
    assert cleared == {"hidden": True, "text": ""}


def test_saved_markdown_carries_the_same_quality_lines_as_the_screen():
    data = quality("partial")
    result = run_node(
        f"""
state.resultAnswerQuality = {json.dumps(data, ensure_ascii=False)};
ui.renderAnswerQuality();
state.resultMeta = {{mode: 'deep', startedAt: '2026-10-01T00:00:00.000Z', requestId: 'r1', stopReason: 'scope_processed'}};
state.resultEvidence = {{items: [], evidenceStatus: 'partial', route: 'deep', indexState: 'deep', warnings: [], answerState: 'generated',
    answerStateMessage: '回答生成済み（要確認）', deepLedger: []}};
state.resultReceivedAt = '2026-10-01T00:05:00.000Z';
state.activeQuery = '質問';
state.rawSearchText = '結論です。[E1]';
ui.saveMarkdown();
out({{screen: ui.els.answerQuality.textContent, markdown: saved[0]}});
"""
    )
    unescaped = result["markdown"].replace("\\", "")
    for line in result["screen"].split("\n"):
        assert f"- {line}" in unescaped
    # 質問・回答の本文はMarkdownに同じ生の回答データで入る（画面も同じrawSearchTextから描画する）。
    assert "結論です。[E1]" in result["markdown"]
    assert "回答状態: 回答生成済み" in unescaped and "要確認" in unescaped


def test_a_new_search_clears_the_previous_quality_and_errors_do_not_keep_it():
    html = INDEX_HTML.read_text(encoding="utf-8")
    start = html.index(
        "state.resultAnswerEmpty = false;\n        state.resultAnswerQuality = null;"
    )
    assert "this.renderAnswerQuality();" in html[start : start + 200]
    assert html.count("state.resultAnswerQuality = null;") >= 3  # 新規検索・SSE error・onError


@pytest.mark.parametrize(("think_value", "display"), [("low", "low"), (False, "false")])
def test_search_stream_renders_model_metadata_and_answer(think_value, display):
    events = [
        {"type": "result_meta", "chatModel": "gpt-oss:120b", "apiThinkValue": think_value},
        {"type": "chunk", "text": "回答本文です。"},
        {"type": "done"},
    ]
    result = run_node(
        f"""
const events = {json.dumps(events, ensure_ascii=False)};
const payload = new TextEncoder().encode(events.map(event => 'data: ' + JSON.stringify(event) + '\\n\\n').join(''));
let sent = false;
Object.assign(context, {{
    AbortController, URLSearchParams, TextDecoder, clearTimeout,
    crypto: {{randomUUID: () => 'test-search'}},
    sessionStorage: {{setItem() {{}}, removeItem() {{}}}},
    fetch: async () => ({{
        ok: true,
        body: {{getReader: () => ({{read: async () => {{
            if (sent) return {{done: true}};
            sent = true;
            return {{done: false, value: payload}};
        }}}})}},
    }}),
}});
ui.els = new Proxy({{}}, {{get(elements, name) {{ return elements[name] ??= element(); }}}});
ui.els.searchInput.value = '質問';
ui.els.searchModeSelect.value = 'answer';
ui.els.searchTimeoutInput.value = '300';
ui.closeEvidenceViewer = () => {{}};
ui.pollHealthUpdate = () => {{}};
state.features.search = true;
ui.startSearch();
setImmediate(() => out({{
    summary: ui.els.resultMetaSummary.textContent,
    summaryHidden: ui.els.resultMetaSummary.hidden,
    answer: ui.els.searchResult.innerHTML,
    searching: state.searching,
}}));
"""
    )
    assert result["summaryHidden"] is False
    assert f"think {display} ・" in result["summary"]
    assert "回答本文です。" in result["answer"]
    assert result["searching"] is False


def test_deep_diagnostics_lines_describe_v2_calls_and_deferred_detail():
    diag = {
        "final_answer": {
            "attempts": [],
            "result": "v2_needs_review",
            "v2": {
                "calls": {
                    "counts": {
                        "draft": {"ok": 2},
                        "g2": {"ok": 1, "not_run": 3},
                        "g34": {"error": 1},
                    }
                },
                "reexplored": True,
            },
        },
        "answer_policy_v2": {"role_deferred": {"practical_detail": 3, "background": 1}},
    }
    lines = run_node(f"out(deepDiagnosticsFormat.lines({json.dumps(diag)}));")
    v2 = next(line for line in lines if line.startswith("最終回答の検査(v2)"))
    assert "検査の一部が未完了または不合格（要確認）" in v2
    assert "草稿 2回 / 意味照合 1回 / 評価 1回 / 再探索 あり" in v2
    assert any("展開しなかった詳細の範囲: 4件" in line for line in lines)
    # 旧経路の診断は従来どおり（v2の行を出さない）。
    legacy = run_node(
        "out(deepDiagnosticsFormat.lines({final_answer: {attempts: [{uncited_lines: 0}], result: 'accepted'}}));"
    )
    assert not any("(v2)" in line for line in legacy)


def test_original_item_lines_under_a_claim_are_shown_in_one_quote_box():
    # 回答方針v2: 限定句付きの列挙は、主張の下に原文の項目行を引用として示す（言い換えない）。
    answer = "- 審査する案件は次のとおりです。 [E1]\n> - 一　概算費用１，０００万円以上の設備更新\n> - 二　概算費用３００万円以上の設備改修（緊急修繕を除く。）\n- 次の主張です。 [E1]"
    html = run_node(f"out(md.render({json.dumps(answer, ensure_ascii=False)}));")
    assert html.count("<blockquote>") == 1
    assert "緊急修繕を除く。）</p></blockquote>" in html
    assert html.count("<li>") == 2
