"""web.bat（非表示起動）でページを閉じたらサーバーを停止する機能のテスト。

- PagePresenceMonitor: 閉鎖猶予・再読み込み・複数タブ・失効・スリープ明け・構築中保留
- run_page_presence_watchdog: 索引構築/検索実行中の保留と1回だけの停止
- /api/page/heartbeat・/api/page/close と /api/health の契約
- UIの在席通知（Node.js vm 上で pagehide の sendBeacon を実行検証）
"""

import http.client
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

import source_view
import web_server
from web_services import PagePresenceMonitor

INDEX_HTML = Path(__file__).resolve().parents[2] / "_internal" / "web" / "index.html"
PAGE_A = "page-aaaaaaaa"
PAGE_B = "page-bbbbbbbb"


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _monitor(clock, **kwargs):
    options = {
        "stale_seconds": 300.0,
        "visible_stale_seconds": 45.0,
        "close_grace_seconds": 15.0,
        "initial_seconds": 600.0,
        "sleep_gap_seconds": 30.0,
        "clock": clock,
    }
    options.update(kwargs)
    return PagePresenceMonitor(**options)


def _advance(monitor, clock, seconds, *, step=2.0, busy=False):
    """watchdogと同じ間隔でtickしながら時間を進め、最後の判定を返す。"""
    result = False
    remaining = seconds
    while remaining > 0:
        delta = min(step, remaining)
        clock.now += delta
        remaining -= delta
        result = monitor.should_shutdown(busy=busy)
    return result


# --- PagePresenceMonitor ------------------------------------------------------


def test_closing_the_last_page_stops_after_grace_period():
    clock = FakeClock()
    monitor = _monitor(clock)
    monitor.heartbeat(PAGE_A)
    assert _advance(monitor, clock, 60) is False

    monitor.close(PAGE_A)
    assert _advance(monitor, clock, 14) is False
    assert _advance(monitor, clock, 2) is True


def test_reload_within_grace_period_keeps_server_running():
    clock = FakeClock()
    monitor = _monitor(clock)
    monitor.heartbeat(PAGE_A)
    monitor.close(PAGE_A)  # pagehide
    assert _advance(monitor, clock, 4) is False
    monitor.heartbeat(PAGE_B)  # 再読み込み後の新しいページ

    assert _advance(monitor, clock, 120) is False
    assert monitor.open_page_count() == 1


def test_other_open_tab_keeps_server_running():
    clock = FakeClock()
    monitor = _monitor(clock)
    monitor.heartbeat(PAGE_A)
    monitor.heartbeat(PAGE_B)
    monitor.close(PAGE_A)

    for _ in range(10):
        assert _advance(monitor, clock, 14) is False
        monitor.heartbeat(PAGE_B)
    assert monitor.open_page_count() == 1


def test_hidden_page_that_disappears_without_close_expires_after_long_timeout():
    clock = FakeClock()
    monitor = _monitor(clock)
    monitor.heartbeat(PAGE_A, visible=False)

    assert _advance(monitor, clock, 300) is False  # 失効時間ちょうどまでは在席扱い
    assert _advance(monitor, clock, 14) is False  # 失効後も猶予を置く
    assert _advance(monitor, clock, 4) is True


def test_visible_page_closed_without_pagehide_expires_quickly():
    """組み込みブラウザ等で close が届かなくても、表示中だったページは短時間で失効する。"""
    clock = FakeClock()
    monitor = _monitor(clock)
    for _ in range(4):
        monitor.heartbeat(PAGE_A, visible=True)
        assert _advance(monitor, clock, 15) is False

    assert _advance(monitor, clock, 44) is False
    assert _advance(monitor, clock, 16) is True  # 45秒失効 + 15秒猶予


def test_page_hidden_before_timer_throttling_uses_long_timeout():
    clock = FakeClock()
    monitor = _monitor(clock)
    monitor.heartbeat(PAGE_A, visible=True)
    monitor.heartbeat(PAGE_A, visible=False)  # visibilitychange で非表示を通知

    assert _advance(monitor, clock, 250) is False


def test_background_tab_throttled_to_one_heartbeat_per_minute_is_not_stopped():
    clock = FakeClock()
    monitor = _monitor(clock)
    for _ in range(30):
        monitor.heartbeat(PAGE_A, visible=False)
        assert _advance(monitor, clock, 60) is False


def test_resume_from_sleep_does_not_stop_before_page_can_heartbeat():
    clock = FakeClock()
    monitor = _monitor(clock)
    monitor.heartbeat(PAGE_A)
    clock.now += 3600  # PCのスリープ等でwatchdogも止まっていた

    assert monitor.should_shutdown() is False
    assert _advance(monitor, clock, 10) is False
    monitor.heartbeat(PAGE_A)
    assert _advance(monitor, clock, 100) is False


def test_server_without_any_page_stops_after_initial_window():
    clock = FakeClock()
    monitor = _monitor(clock)
    assert _advance(monitor, clock, 598) is False
    assert _advance(monitor, clock, 4) is True


def test_busy_server_defers_stop_until_work_finishes():
    clock = FakeClock()
    monitor = _monitor(clock)
    monitor.heartbeat(PAGE_A)
    monitor.close(PAGE_A)

    assert _advance(monitor, clock, 3600, busy=True) is False
    assert _advance(monitor, clock, 2, busy=False) is True


@pytest.mark.parametrize(
    "page_id", [None, 123, "", "short", "x" * 65, "has space!", "日本語のページID"]
)
def test_invalid_page_ids_are_rejected(page_id):
    monitor = PagePresenceMonitor()
    assert PagePresenceMonitor.valid_page_id(page_id) is False
    with pytest.raises(ValueError):
        monitor.heartbeat(page_id)
    with pytest.raises(ValueError):
        monitor.close(page_id)


def test_page_table_is_bounded():
    clock = FakeClock()
    monitor = _monitor(clock)
    for number in range(PagePresenceMonitor.MAX_PAGES + 5):
        clock.now += 1
        monitor.heartbeat(f"page-{number:08d}")
    assert monitor.open_page_count() == PagePresenceMonitor.MAX_PAGES


# --- watchdog -----------------------------------------------------------------


class _FakeIndex:
    def __init__(self):
        self.running = False

    def is_running(self):
        return self.running


class _FakeJobs:
    running_count = 0


class _FakeServer:
    def __init__(self):
        self.index_coordinator = _FakeIndex()
        self.job_table = _FakeJobs()


class _ScriptedMonitor:
    def __init__(self, server, script):
        self.server = server
        self.script = list(script)
        self.busy_seen = []

    def should_shutdown(self, *, busy):
        self.busy_seen.append(busy)
        step = self.script.pop(0)
        if callable(step):
            step()
            return False
        return step


def test_watchdog_reports_busy_work_and_stops_once():
    server = _FakeServer()

    def start_index():
        server.index_coordinator.running = True

    def finish_index_and_start_search():
        server.index_coordinator.running = False
        server.job_table.running_count = 1

    def finish_search():
        server.job_table.running_count = 0

    monitor = _ScriptedMonitor(
        server, [start_index, finish_index_and_start_search, finish_search, False, True]
    )
    stop = threading.Event()
    idle_calls = []

    web_server.run_page_presence_watchdog(
        server, monitor, stop, lambda: idle_calls.append(True), interval=0
    )

    assert idle_calls == [True]
    assert monitor.busy_seen == [False, True, True, False, False]


def test_watchdog_exits_without_stopping_when_server_is_already_shutting_down():
    stop = threading.Event()
    stop.set()
    idle_calls = []
    web_server.run_page_presence_watchdog(
        _FakeServer(), _ScriptedMonitor(None, []), stop, lambda: idle_calls.append(True), interval=0
    )
    assert idle_calls == []


# --- HTTP API -----------------------------------------------------------------


@pytest.fixture(params=[True, False], ids=["web-bat-hidden", "console"])
def http_server(request, tmp_path):
    monitor = PagePresenceMonitor() if request.param else None
    server = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token="test-session-token",
        search_timeout=300,
        evidence_registry=source_view.EvidenceRegistry(tmp_path),
        page_presence=monitor,
    )
    server.bind_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def call(method, path, body=None, *, auth=True, origin=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"}
        if auth:
            headers["Cookie"] = "offlineai_session=test-session-token"
        if origin:
            headers["Origin"] = origin
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    try:
        yield monitor, call
    finally:
        server.shutdown_jobs_once()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_page_presence_api_contract(http_server, monkeypatch):
    monitor, call = http_server
    monkeypatch.setattr(web_server, "check_health", lambda: {"status": "ok", "features": {}})
    heartbeat = json.dumps({"page_id": PAGE_A, "visible": True})

    assert call("POST", "/api/page/heartbeat", heartbeat, auth=False)[0] == 403
    assert call("POST", "/api/page/heartbeat", heartbeat, origin="http://evil.example")[0] == 403
    for body in ('{"page_id": "bad id"}', "{}", "[]", "{broken", '{"page_id": "page-aaaaaaaa", "visible": "yes"}'):
        status, payload = call("POST", "/api/page/heartbeat", body)
        assert status == 400 and payload["error"]["code"] == "invalid_query"

    status, payload = call("POST", "/api/page/heartbeat", heartbeat)
    assert status == 200
    status, health = call("GET", "/api/health")
    assert status == 200

    if monitor is None:
        assert payload == {"enabled": False}
        assert health["pageCloseStop"] is False
        return
    assert payload == {"enabled": True, "openPages": 1}
    assert health["pageCloseStop"] is True
    assert call("POST", "/api/page/heartbeat", json.dumps({"page_id": PAGE_B}))[1]["openPages"] == 2
    assert call("POST", "/api/page/close", heartbeat)[1] == {"enabled": True, "openPages": 1}
    assert monitor.open_page_count() == 1
    assert monitor._pages[PAGE_B][1] is False  # visible省略時は非表示扱い（長い失効時間）


# --- UI -----------------------------------------------------------------------

_UI_HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const scenario = JSON.parse(process.argv[2]);
const listeners = {};
const fetches = [];
const beacons = [];
const intervals = [];
const hint = {hidden: true, textContent: '', classList: {toggle(name, on) { hint.cls = on ? name : ''; }}};
const context = {
    console, JSON, Blob, Uint8Array, Array, Error, Promise,
    crypto: {randomUUID: () => '11111111-2222-3333-4444-555555555555'},
    document: {addEventListener(type, fn) { listeners['doc:' + type] = fn; }, visibilityState: 'visible'},
    window: {addEventListener(type, fn) { listeners[type] = fn; }},
    navigator: scenario.beacon ? {sendBeacon(url, blob) { beacons.push(url); return true; }} : {},
    setInterval(fn, ms) { intervals.push(ms); return 1; },
    fetch: async (url, options) => {
        fetches.push({url, body: options && options.body, keepalive: Boolean(options && options.keepalive)});
        if (scenario.fail) throw new Error('offline');
        return {ok: true, json: async () => ({enabled: scenario.enabled})};
    },
};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(script + '\nglobalThis.fixture = {ui, pagePresence};', context);
const {ui, pagePresence} = context.fixture;
ui.els = {pageCloseHint: hint};
(async () => {
    await pagePresence.start();
    if (scenario.fail) await pagePresence.beat();
    context.document.visibilityState = 'hidden';
    await listeners['doc:visibilitychange']();
    listeners.pagehide();
    await new Promise(resolve => setImmediate(resolve));
    process.stdout.write(JSON.stringify({fetches, beacons, intervals, hint: {hidden: hint.hidden, text: hint.textContent, cls: hint.cls || ''}}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def _run_ui(**scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dynamic UI contract tests")
    result = subprocess.run(
        [node, "-e", _UI_HARNESS, str(INDEX_HTML), json.dumps(scenario)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_ui_heartbeats_shows_hint_and_beacons_close_on_pagehide():
    result = _run_ui(enabled=True, beacon=True)

    assert result["fetches"][0]["url"] == "/api/page/heartbeat"
    assert json.loads(result["fetches"][0]["body"]) == {
        "page_id": "11111111-2222-3333-4444-555555555555",
        "visible": True,
    }
    assert result["intervals"] == [15000]
    assert json.loads(result["fetches"][1]["body"])["visible"] is False  # 非表示化も通知
    assert result["beacons"] == ["/api/page/close"]
    assert result["hint"]["hidden"] is False
    assert "閉じると、サーバーは自動で停止" in result["hint"]["text"]


def test_ui_without_beacon_falls_back_to_keepalive_fetch_and_hides_hint_in_console_mode():
    result = _run_ui(enabled=False, beacon=False)

    assert result["fetches"][-1]["url"] == "/api/page/close"
    assert result["fetches"][-1]["keepalive"] is True
    assert result["hint"]["hidden"] is True


def test_ui_reports_server_gone_after_repeated_heartbeat_failures():
    result = _run_ui(enabled=True, beacon=True, fail=True)

    assert result["hint"]["hidden"] is False
    assert result["hint"]["cls"] == "server-gone"
    assert "web.bat を起動し直してください" in result["hint"]["text"]
