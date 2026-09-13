import os
import json
import subprocess
import sys
from pathlib import Path


OFFLINE_AI = Path(__file__).resolve().parents[2]


def test_protocol_relative_links_are_rejected_by_renderer():
    html = (OFFLINE_AI / "_internal" / "web" / "index.html").read_text(encoding="utf-8")
    assert 'collapsed.replace(/\\\\/g, "/")' in html
    assert 'normalizedSlashes.startsWith("//")' in html


def test_default_web_log_does_not_include_raw_query():
    server = (OFFLINE_AI / "_internal" / "web_server.py").read_text(encoding="utf-8")
    assert "query_length=len(query)" in server
    assert "query=query[:50]" not in server


def test_public_documents_and_version_exist():
    for name in (
        "README.md",
        "SUPPORT.md",
        "UNINSTALL.md",
        "THIRD-PARTY-NOTICES.md",
        "VERSION",
        "CHANGELOG.md",
        "release-metadata.json",
    ):
        assert (OFFLINE_AI / name).is_file(), name


def test_web_server_rejects_remote_ollama_host_without_traceback():
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "http://example.com:11434"
    result = subprocess.run(
        [sys.executable, str(OFFLINE_AI / "_internal" / "web_server.py"), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=15,
    )

    assert result.returncode == 2
    assert "example.com" in result.stderr
    assert "Traceback" not in result.stderr


def test_bootstrap_failure_explains_one_time_token_recovery():
    server = (OFFLINE_AI / "_internal" / "web_server.py").read_text(encoding="utf-8")
    assert "Restart web.bat to issue a new one-time token." in server


def test_pdf_conversion_contract_is_absent():
    html = (OFFLINE_AI / "_internal" / "web" / "index.html").read_text(encoding="utf-8")
    server = (OFFLINE_AI / "_internal" / "web_server.py").read_text(encoding="utf-8")
    services = (OFFLINE_AI / "_internal" / "web_services.py").read_text(encoding="utf-8")
    manifest = json.loads(
        (OFFLINE_AI / "_internal" / "download-manifest.json").read_text(encoding="utf-8")
    )

    assert "/api/convert" not in html
    assert "PDF変換" not in html
    assert "--convert-timeout" not in server
    assert "OFFLINEAI_CONVERT_TIMEOUT" not in server
    assert '"convert"' not in services
    tool_ids = {item["id"] for item in manifest["tools"]}
    assert not {"pdftotext", "git-for-windows", "structured-parser"} & tool_ids


def test_search_only_web_ui_has_no_tab_or_panel_shell():
    html = (OFFLINE_AI / "_internal" / "web" / "index.html").read_text(encoding="utf-8")

    assert ".tabs {" not in html
    assert ".tab {" not in html
    assert ".tab:hover" not in html
    assert ".tab.active" not in html
    assert ".panel {" not in html
    assert ".panel.active" not in html
    assert 'class="panel active"' not in html
