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
