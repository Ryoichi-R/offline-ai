#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
HELPER="$SCRIPT_DIR/macos_runtime.py"
RUNTIME_CONFIG="${OFFLINE_AI_MACOS_RUNTIME_CONFIG:-$REPO_ROOT/.test-results/macos-large-model/host-runtime.json}"
MODEL_OVERRIDE=""
MODEL_OVERRIDE_SET=0
MODE=""
OLLAMA_LAUNCHER_PID=""
READY_DIR=""

usage() {
    printf '%s\n' "Usage: bash scripts/start-macos.sh [--runtime-config FILE] [--model TAG] <web|cli> [-- APP_ARGS...]"
}

while (($#)); do
    case "$1" in
        --runtime-config) (($# >= 2)) || { usage >&2; exit 2; }; RUNTIME_CONFIG="$2"; shift 2 ;;
        --model) (($# >= 2)) || { usage >&2; exit 2; }; MODEL_OVERRIDE="$2"; MODEL_OVERRIDE_SET=1; shift 2 ;;
        web|cli) MODE="$1"; shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ -z "$MODE" ]]; then usage >&2; exit 2; fi
if (($#)) && [[ "$1" == "--" ]]; then shift; fi
for arg; do
    case "$arg" in
        --model|--model=*) printf '%s\n' "[ERROR] choose the model with the launcher --model option" >&2; exit 2 ;;
    esac
done

CONFIG_PYTHON="$(command -v python3)"
config_value() { "$CONFIG_PYTHON" "$HELPER" config-value --path "$RUNTIME_CONFIG" --key "$1"; }
PYTHON_BIN="$(config_value python_executable)"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "[ERROR] configured Python executable is unavailable" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    printf '%s\n' "[ERROR] Python 3.11 or later is required" >&2
    exit 1
fi
MODEL_FILE="$REPO_ROOT/_internal/.model"
if ((MODEL_OVERRIDE_SET)); then
    MODEL="$("$PYTHON_BIN" "$HELPER" resolve-model --model "$MODEL_OVERRIDE" --model-file "$MODEL_FILE")"
else
    MODEL="$("$PYTHON_BIN" "$HELPER" resolve-model --model-file "$MODEL_FILE")"
fi
case "$MODEL" in
    qwen3.5:122b|gpt-oss:120b) ;;
    *) printf '%s\n' "[ERROR] Mac runtime accepts only the pinned qwen3.5:122b and gpt-oss:120b candidates" >&2; exit 2 ;;
esac

export OFFLINE_AI_CHAT_MODEL="$MODEL"
export OLLAMA_HOST="http://127.0.0.1:11434"
export OFFLINE_AI_LARGE_MODEL_CALL_TIMEOUT="${OFFLINE_AI_LARGE_MODEL_CALL_TIMEOUT:-300}"

cleanup() {
    if [[ -n "$OLLAMA_LAUNCHER_PID" ]] && kill -0 "$OLLAMA_LAUNCHER_PID" 2>/dev/null; then
        kill -TERM "$OLLAMA_LAUNCHER_PID" 2>/dev/null || true
        wait "$OLLAMA_LAUNCHER_PID" 2>/dev/null || true
    fi
    if [[ -n "$READY_DIR" ]]; then
        rm -f -- "$READY_DIR/ready"
        rmdir -- "$READY_DIR"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

READY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/offline-ai-launch.XXXXXXXX")"
bash "$SCRIPT_DIR/start-ollama-macos.sh" --runtime-config "$RUNTIME_CONFIG" --model "$MODEL" --ready-file "$READY_DIR/ready" &
OLLAMA_LAUNCHER_PID=$!
READY=0
for _ in $(seq 1 120); do
    if [[ -f "$READY_DIR/ready" ]] && "$PYTHON_BIN" "$HELPER" server-ready \
        && "$PYTHON_BIN" "$HELPER" owner-matches \
            --models-directory "$(config_value models_directory)" \
            --volume-uuid "$(config_value volume_uuid)" \
            --ollama-executable "$(config_value ollama_executable)"; then
        READY=1
        break
    fi
    if ! kill -0 "$OLLAMA_LAUNCHER_PID" 2>/dev/null; then
        if ! wait "$OLLAMA_LAUNCHER_PID"; then
            OLLAMA_LAUNCHER_PID=""
            printf '%s\n' "[ERROR] Ollama startup validation failed" >&2
            exit 1
        fi
        OLLAMA_LAUNCHER_PID=""
        if [[ ! -f "$READY_DIR/ready" ]]; then
            printf '%s\n' "[ERROR] Ollama launcher exited without readiness confirmation" >&2
            exit 1
        fi
    fi
    sleep 1
done
if ((READY == 0)); then
    printf '%s\n' "[ERROR] the owned Ollama runtime did not become ready" >&2
    exit 1
fi
"$PYTHON_BIN" "$HELPER" verify-model --model "$MODEL" >/dev/null

if [[ "$MODE" == "web" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/_internal/web_server.py" --model "$MODEL" "$@"
else
    "$PYTHON_BIN" "$REPO_ROOT/_internal/search.py" --model "$MODEL" "$@"
fi
