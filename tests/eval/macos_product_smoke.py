"""Bounded CLI/HTTP smoke against real local models; no holdout or user material.

Default: prepare only. Model output, tokens and exception text stay in memory.
The real product entrypoints/handlers run in disposable, hash-bound copies.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_internal"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests/eval"))
import macos_large_model_compare as compare  # noqa: E402
import macos_runtime  # noqa: E402
import macos_smoke_diagnostics as diagnostics  # noqa: E402
from model_config import MAC_MODEL_DIGESTS, resolve_model_reasoning  # noqa: E402

CORPUS = "tests/eval/macos-smoke-corpus/handbook.md"
QUERY = "研修用電卓の貸与期間の上限と返却先を教えてください"
FACTS = {"duration": "7日", "returnOffice": "研修支援窓口"}
CASE_SECONDS = 300
TERMINALS = {"done", "error"}


def build_plan(run_id, interfaces, models, selected_cases=None):
    compare.validate_run_id(run_id)
    if (
        not interfaces
        or len(set(interfaces)) != len(interfaces)
        or set(interfaces) - {"cli", "web"}
    ):
        raise ValueError("invalid smoke interfaces")
    if not models or len(set(models)) != len(models) or set(models) - set(MAC_MODEL_DIGESTS):
        raise ValueError("invalid pinned models")
    names = [
        name
        for name in compare.snapshot_files(ROOT)
        if not name.startswith("tests/eval/macos-large-model-holdout/")
    ]
    names = sorted(set(names) | {CORPUS, "_internal/web/index.html"})
    cases = [
        {"model": model, "interface": interface, "mode": mode}
        for model in models
        for interface in interfaces
        for mode in ("answer", "deep")
    ]
    if selected_cases is not None:
        valid = {case["interface"] + ":" + case["mode"] for case in cases}
        if (
            not selected_cases
            or len(set(selected_cases)) != len(selected_cases)
            or set(selected_cases) - valid
        ):
            raise ValueError("invalid case selection")
        cases = [case for case in cases if case["interface"] + ":" + case["mode"] in selected_cases]
    return {
        "schemaVersion": "1.1",
        "runId": run_id,
        "status": "PREPARED_NOT_RUN",
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "synthetic-product-wiring-only-not-quality-acceptance",
        "product": {"files": {name: compare.sha256_file(ROOT / name) for name in names}},
        "corpusSha256": compare.sha256_file(ROOT / CORPUS),
        "querySha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "models": {model: MAC_MODEL_DIGESTS[model] for model in models},
        "cases": cases,
        "caseTimeoutSeconds": CASE_SECONDS,
        "maxGenerationSeconds": len(cases) * CASE_SECONDS,
        "lifecycleGraceSecondsPerCase": 10,
        "reasoning": "off",
        "embedding": "keyword-only",
        "productEnvironment": "OFFLINE_AI_* overrides removed in case subprocess",
        "inferenceCancellationVerified": False,
        "holdoutExecuted": False,
        "answerTextStored": False,
        "runs": [],
    }


def summarize_answer(answer):
    return {
        "answerSha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "answerChars": len(answer),
        "requiredFacts": {key: value in answer for key, value in FACTS.items()},
        "citationCount": len(re.findall(r"\[(?:E|L)?\d+\]", answer)),
    }


def parse_swap_bytes(text):
    match = re.search(r"used\s*=\s*([\d.]+)([KMGT])", text)
    return int(float(match[1]) * 1024 ** ("KMGT".index(match[2]) + 1)) if match else None


def process_rss_bytes(table, root_pid):
    rows = {}
    for line in table.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(field.isdigit() for field in fields):
            pid, parent, rss = map(int, fields)
            rows[pid] = (parent, rss * 1024)
    if root_pid not in rows:
        return None
    selected = {root_pid}
    while True:
        descendants = {pid for pid, (parent, _) in rows.items() if parent in selected}
        if descendants <= selected:
            return sum(rows[pid][1] for pid in selected)
        selected.update(descendants)


class ResourceSampler:
    """Sample numeric host/process counters; RSS sums may include shared pages."""

    def __init__(self, pid, ollama_pid):
        self.pid, self.ollama_pid = pid, ollama_pid
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _command(args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=2, check=False)
        return result.stdout if result.returncode == 0 else ""

    def sample(self):
        try:
            table = self._command(["ps", "-axo", "pid=,ppid=,rss="])
            swap = (
                self._command(["sysctl", "-n", "vm.swapusage"]) if sys.platform == "darwin" else ""
            )
            self.samples.append(
                {
                    "productProcessTreeRssBytes": process_rss_bytes(table, self.pid),
                    "ollamaProcessTreeRssBytes": process_rss_bytes(table, self.ollama_pid),
                    "systemSwapUsedBytes": parse_swap_bytes(swap),
                }
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=5)
        result = {
            "sampleCount": len(self.samples),
            "rssMeaning": "sampled-process-tree-sum-includes-shared-pages",
        }
        for key in (
            "productProcessTreeRssBytes",
            "ollamaProcessTreeRssBytes",
            "systemSwapUsedBytes",
        ):
            values = [sample[key] for sample in self.samples if sample[key] is not None]
            result[key] = (
                {"first": values[0], "maximum": max(values), "last": values[-1]} if values else None
            )
        return result


def cli_case(case):
    import deep_research
    import search

    captured = {}
    original_stream = search.stream_ollama_chat
    original_deep = deep_research.run_deep_research

    def stream(*args, **kwargs):
        answer = original_stream(*args, **kwargs)
        captured.update(summarize_answer(answer))
        captured["answerState"] = "generated" if answer.strip() else "empty"
        return answer

    def deep(*args, **kwargs):
        result = original_deep(*args, **kwargs)
        captured.update(summarize_answer(result.answer))
        captured.update(
            answerState=result.answer_state,
            deepStatus=result.status,
            stopReason=result.stop_reason,
        )
        captured["stageSeconds"] = result.diagnostics.get("stage_seconds", {})
        captured["sourceRepair"] = result.diagnostics.get("final_answer", {}).get("source_repair")
        return result

    previous = sys.argv
    search.stream_ollama_chat = stream
    deep_research.run_deep_research = deep
    sys.argv = [
        "search.py",
        QUERY,
        "--model",
        case["model"],
        "--mode",
        case["mode"],
        "--reasoning",
        "off",
        "--timeout-seconds",
        str(CASE_SECONDS),
    ]
    try:
        search.main()
    finally:
        sys.argv = previous
        search.stream_ollama_chat = original_stream
        deep_research.run_deep_research = original_deep
    return captured


def web_case(case):
    import search
    import source_view
    import web_server
    from model_config import model_call_timeout_seconds

    token = secrets.token_urlsafe(32)
    server = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token=token,
        search_timeout=CASE_SECONDS,
        evidence_registry=source_view.EvidenceRegistry(ROOT / "skill-source"),
        chat_model=case["model"],
        model_digest=MAC_MODEL_DIGESTS[case["model"]],
        model_selection_source="command-line",
        generation_stall_timeout=model_call_timeout_seconds(
            case["model"], search.GENERATION_STALL_TIMEOUT
        ),
    )
    server.bind_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=CASE_SECONDS + 5)
    host = f"127.0.0.1:{server.bind_port}"
    events = []
    first_content = None
    started = time.monotonic()
    try:
        conn.request(
            "POST",
            "/bootstrap",
            json.dumps({"token": token}),
            headers={
                "Host": host,
                "Origin": f"http://{host}",
                "Content-Type": "application/json",
            },
        )
        response = conn.getresponse()
        response.read()
        if response.status != 204:
            raise RuntimeError("bootstrap failed")
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        conn.request(
            "GET",
            "/api/search?"
            + urlencode(
                {
                    "q": QUERY,
                    "mode": case["mode"],
                    "reasoning": "off",
                    "timeout_seconds": CASE_SECONDS,
                    "request_id": "synthetic-smoke",
                }
            ),
            headers={"Host": host, "Cookie": cookie},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError("search HTTP failed")
        while True:
            line = response.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                event = json.loads(line[6:])
                # Thinking deltas are never retained in the observer.
                if event.get("type") == "thinking":
                    continue
                events.append(event)
                if event.get("type") == "chunk" and event.get("text") and first_content is None:
                    first_content = time.monotonic() - started
    finally:
        conn.close()
        server.shutdown_jobs_once()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    answer = "".join(str(event.get("text", "")) for event in events if event.get("type") == "chunk")
    meta = next((event for event in events if event.get("type") == "result_meta"), {})
    evidence = next((event for event in events if event.get("type") == "evidence"), {})
    return {
        **summarize_answer(answer),
        "httpStatus": response.status,
        "terminalCount": sum(event.get("type") in TERMINALS for event in events),
        "errorCount": sum(event.get("type") == "error" for event in events),
        "errorCodes": [
            diagnostics.safe_code(event.get("code"))
            for event in events
            if event.get("type") == "error"
        ],
        "answerState": evidence.get("answerState", "generated" if answer else "empty"),
        "firstContentSeconds": round(first_content, 3) if first_content is not None else None,
        "modelMatches": meta.get("chatModel") == case["model"],
        "digestMatches": meta.get("modelDigest") == MAC_MODEL_DIGESTS[case["model"]],
        "apiThinkValue": meta.get("apiThinkValue"),
        "deepStatus": meta.get("deepStatus"),
        "stopReason": meta.get("stopReason"),
        "jobsReleased": server.job_table.running_count == 0,
    }


def case_passed(case, result):
    passed = (
        result.get("answerState") == "generated"
        and all(result.get("requiredFacts", {}).get(key) is True for key in FACTS)
        and result.get("citationCount", 0) > 0
        and not result.get("failureType")
    )
    if case["mode"] == "deep":
        passed = passed and result.get("deepStatus") == "completed"
    if case["interface"] == "web":
        passed = (
            passed
            and result.get("terminalCount") == 1
            and result.get("errorCount") == 0
            and result.get("modelMatches") is True
            and result.get("digestMatches") is True
            and result.get("jobsReleased") is True
            and result.get("apiThinkValue")
            == resolve_model_reasoning(case["model"], "off")["think"]
        )
    return bool(passed)


def check_corpus(receipt):
    if compare.sha256_file(ROOT / "skill-source/handbook.md") != receipt["corpusSha256"]:
        raise ValueError("smoke corpus changed")


def child_case(case_path):
    payload = json.loads(case_path.read_text())
    case, receipt = payload["case"], payload["receipt"]
    compare.verify_snapshot(receipt, ROOT)
    check_corpus(receipt)
    started = time.monotonic()
    result = {}
    observed = []
    with (
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
        diagnostics.observe(observed),
    ):
        try:
            result = cli_case(case) if case["interface"] == "cli" else web_case(case)
        except (Exception, SystemExit) as exc:
            result = {"failureType": type(exc).__name__}
    compare.verify_snapshot(receipt, ROOT)
    check_corpus(receipt)
    result["observations"] = observed
    result.update(case)
    result["elapsedSeconds"] = round(time.monotonic() - started, 3)
    result["reasoning"] = resolve_model_reasoning(case["model"], "off")
    result["passed"] = case_passed(case, result)
    return result


def stop_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Only this disposable, explicitly created process group is affected.
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_case(case, receipt, isolated, ollama_pid):
    case_path = isolated / "smoke-case.json"
    compare._atomic_write(case_path, {"case": case, "receipt": receipt})
    env = {key: value for key, value in os.environ.items() if not key.startswith("OFFLINE_AI_")}
    env["OLLAMA_HOST"] = "http://127.0.0.1:11434"
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            str(isolated / "tests/eval/macos_product_smoke.py"),
            "--case",
            str(case_path),
        ],
        cwd=isolated,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    sampler = ResourceSampler(process.pid, ollama_pid)
    sampler.thread.start()
    try:
        try:
            stdout, _ = process.communicate(timeout=CASE_SECONDS + 10)
            if process.returncode:
                result = {**case, "passed": False, "failureType": "CaseProcessFailed"}
            else:
                result = json.loads(stdout)
        except subprocess.TimeoutExpired:
            stop_process_group(process)
            result = {**case, "passed": False, "failureType": "CaseDeadline"}
    finally:
        stop_process_group(process)
        resources = sampler.finish()
    result["resources"] = resources
    return result


def execute(receipt, output, runtime_config):
    if sys.platform != "darwin":
        raise ValueError("Mac smoke execution requires macOS")
    with tempfile.TemporaryDirectory(prefix="offline-ai-product-smoke-") as temporary:
        isolated = Path(temporary)
        compare.copy_snapshot(receipt, isolated)
        source = isolated / "skill-source/handbook.md"
        source.parent.mkdir()
        source.write_bytes((isolated / CORPUS).read_bytes())
        ready = isolated / "ollama-ready"
        launcher = subprocess.Popen(
            [
                "bash",
                str(isolated / "scripts/start-ollama-macos.sh"),
                "--runtime-config",
                str(runtime_config),
                "--model",
                next(iter(receipt["models"])),
                "--ready-file",
                str(ready),
            ],
            cwd=isolated,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        receipt["status"] = "RUNNING"
        compare._atomic_write(output, receipt)
        try:
            deadline = time.monotonic() + 140
            while not ready.exists():
                if launcher.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("owned runtime startup failed")
                time.sleep(0.2)
            compare.validate_owned_runtime(runtime_config)
            owner = macos_runtime._read_owner()
            compare.search.OLLAMA_HOST = "http://127.0.0.1:11434"
            for model in receipt["models"]:
                compare.search.get_model_identity(model)
            for case in receipt["cases"]:
                compare.verify_snapshot(receipt, isolated)
                result = run_case(case, receipt, isolated, int(owner["pid"]))
                receipt["runs"].append(result)
                compare._atomic_write(output, receipt)
                print(
                    json.dumps(
                        {
                            key: result.get(key)
                            for key in (
                                "model",
                                "interface",
                                "mode",
                                "passed",
                                "elapsedSeconds",
                                "failureType",
                            )
                        }
                    ),
                    flush=True,
                )
                if result.get("failureType") in {"CaseDeadline", "CaseProcessFailed"}:
                    break
            compare.verify_snapshot(receipt, isolated)
            receipt["status"] = (
                "PASS"
                if len(receipt["runs"]) == len(receipt["cases"])
                and all(item["passed"] for item in receipt["runs"])
                else "FAIL"
            )
        except BaseException as exc:
            receipt["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            receipt["failureType"] = type(exc).__name__
            raise
        finally:
            stop_process_group(launcher)
            receipt["finishedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            compare._atomic_write(output, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--interfaces", choices=("cli", "both"), default="both")
    parser.add_argument("--model", action="append", choices=tuple(MAC_MODEL_DIGESTS))
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / ".test-results/macos-large-model/host-runtime.json",
    )
    parser.add_argument(
        "--select-case",
        action="append",
        choices=("cli:answer", "cli:deep", "web:answer", "web:deep"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-case-runs", type=int, default=0)
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.case:

            def terminate(_signum, _frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, terminate)
            print(json.dumps(child_case(args.case), ensure_ascii=False))
            return 0
        receipt = build_plan(
            args.run_id,
            ["cli", "web"] if args.interfaces == "both" else ["cli"],
            args.model or list(MAC_MODEL_DIGESTS),
            selected_cases=args.select_case,
        )
        if args.execute and args.confirm_case_runs != len(receipt["cases"]):
            raise ValueError("case count confirmation is required")
        output = ROOT / ".test-results/macos-large-model" / receipt["runId"] / "smoke.json"
        if output.exists():
            raise ValueError("run-id already exists")
        compare._atomic_write(output, receipt)
        if args.execute:
            execute(receipt, output, args.runtime_config.resolve())
        print(
            json.dumps(
                {key: receipt[key] for key in ("runId", "status", "maxGenerationSeconds")},
                ensure_ascii=False,
            )
        )
        return 0 if receipt["status"] in {"PREPARED_NOT_RUN", "PASS"} else 1
    except (Exception, KeyboardInterrupt) as exc:
        print(
            json.dumps({"status": "BLOCKED", "failureType": type(exc).__name__}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
