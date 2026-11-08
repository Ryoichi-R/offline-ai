#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
HELPER="$SCRIPT_DIR/macos_runtime.py"
RUNTIME_CONFIG="${OFFLINE_AI_MACOS_RUNTIME_CONFIG:-$REPO_ROOT/.test-results/macos-large-model/host-runtime.json}"
MODEL_OVERRIDE=""
MODEL_OVERRIDE_SET=0
OLLAMA_PID=""
READY_FILE=""

usage() {
    printf '%s\n' "Usage: bash scripts/start-ollama-macos.sh [--runtime-config FILE] [--model TAG]"
}

while (($#)); do
    case "$1" in
        --runtime-config) (($# >= 2)) || { usage >&2; exit 2; }; RUNTIME_CONFIG="$2"; shift 2 ;;
        --ready-file) (($# >= 2)) || { usage >&2; exit 2; }; READY_FILE="$2"; shift 2 ;;
        --model) (($# >= 2)) || { usage >&2; exit 2; }; MODEL_OVERRIDE="$2"; MODEL_OVERRIDE_SET=1; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

CONFIG_PYTHON="$(command -v python3)"
config_value() { "$CONFIG_PYTHON" "$HELPER" config-value --path "$RUNTIME_CONFIG" --key "$1"; }
PYTHON_BIN="$(config_value python_executable)"
MODELS_DIR="$(config_value models_directory)"
VOLUME_UUID="$(config_value volume_uuid)"
MINIMUM_FREE_BYTES="$(config_value minimum_free_bytes)"
OLLAMA_BIN="$(config_value ollama_executable)"
EXPECTED_OLLAMA_VERSION="$(config_value ollama_version)"
MODEL_FILE="$REPO_ROOT/_internal/.model"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "[ERROR] configured Python executable is unavailable" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    printf '%s\n' "[ERROR] Python 3.11 or later is required" >&2
    exit 1
fi
if [[ ! -x "$OLLAMA_BIN" ]]; then
    printf '%s\n' "[ERROR] configured Ollama executable is unavailable" >&2
    exit 1
fi

if ((MODEL_OVERRIDE_SET)); then
    MODEL="$("$PYTHON_BIN" "$HELPER" resolve-model --model "$MODEL_OVERRIDE" --model-file "$MODEL_FILE")"
else
    MODEL="$("$PYTHON_BIN" "$HELPER" resolve-model --model-file "$MODEL_FILE")"
fi
case "$MODEL" in
    qwen3.5:122b|gpt-oss:120b) ;;
    *) printf '%s\n' "[ERROR] Mac runtime accepts only the pinned qwen3.5:122b and gpt-oss:120b candidates" >&2; exit 2 ;;
esac

VERSION_OUTPUT="$("$OLLAMA_BIN" --version 2>&1 || true)"
case "$VERSION_OUTPUT" in
    *"$EXPECTED_OLLAMA_VERSION"*) ;;
    *) printf '%s\n' "[ERROR] Ollama version does not match the local pinned runtime" >&2; exit 1 ;;
esac

export OLLAMA_HOST="http://127.0.0.1:11434"
export OFFLINE_AI_CHAT_MODEL="$MODEL"
export OFFLINE_AI_LARGE_MODEL_CALL_TIMEOUT="${OFFLINE_AI_LARGE_MODEL_CALL_TIMEOUT:-300}"
"$PYTHON_BIN" "$HELPER" validate-storage --models-directory "$MODELS_DIR" --volume-uuid "$VOLUME_UUID" --minimum-free-bytes "$MINIMUM_FREE_BYTES"

notify_ready() {
    if [[ -n "$READY_FILE" ]]; then printf '%s\n' ready >"$READY_FILE"; fi
}

if "$PYTHON_BIN" "$HELPER" server-ready; then
    if "$PYTHON_BIN" "$HELPER" owner-matches --models-directory "$MODELS_DIR" --volume-uuid "$VOLUME_UUID" --ollama-executable "$OLLAMA_BIN"; then
        "$PYTHON_BIN" "$HELPER" verify-model --model "$MODEL" >/dev/null
        notify_ready
        printf '%s\n' "Ollama is already running from the matching offline-ai launcher."
        exit 0
    fi
    printf '%s\n' "[ERROR] port 11434 is active but is not owned by the matching offline-ai launcher; it was left running" >&2
    exit 1
fi

cleanup() {
    if [[ -n "$OLLAMA_PID" ]]; then
        kill -TERM "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
        "$PYTHON_BIN" "$HELPER" remove-owner --pid "$OLLAMA_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OLLAMA_MODELS="$MODELS_DIR" \
OLLAMA_HOST="127.0.0.1:11434" \
OLLAMA_NO_CLOUD=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NUM_PARALLEL=1 \
    "$OLLAMA_BIN" serve >/dev/null 2>&1 &
OLLAMA_PID=$!
"$PYTHON_BIN" "$HELPER" write-owner --pid "$OLLAMA_PID" --models-directory "$MODELS_DIR" --volume-uuid "$VOLUME_UUID" --ollama-executable "$OLLAMA_BIN"

READY=0
for _ in $(seq 1 120); do
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        printf '%s\n' "[ERROR] owned Ollama process exited during startup" >&2
        exit 1
    fi
    if "$PYTHON_BIN" "$HELPER" server-ready && "$PYTHON_BIN" "$HELPER" owner-matches --models-directory "$MODELS_DIR" --volume-uuid "$VOLUME_UUID" --ollama-executable "$OLLAMA_BIN"; then
        READY=1
        break
    fi
    sleep 1
done
if ((READY == 0)); then
    printf '%s\n' "[ERROR] Ollama did not become ready within 120 seconds" >&2
    exit 1
fi
"$PYTHON_BIN" "$HELPER" verify-model --model "$MODEL" >/dev/null
notify_ready
printf '%s\n' "Ollama is ready on loopback with the configured external model directory and pinned model."
wait "$OLLAMA_PID"
