"""offline-ai のモデル既定値と .model 移行処理。"""

import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys

DEFAULT_MODEL = "gpt-oss:20b"
LEGACY_MODELS = {
    "qwen2.5:3b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen3.5:35b-a3b",
}


def migrate_model_file(model_path: Path, current_model: str) -> str:
    """旧モデル名が検出された場合、DEFAULT_MODEL に自動移行する。"""
    if current_model in LEGACY_MODELS:
        print(
            f"[WARN] 旧モデル '{current_model}' -> '{DEFAULT_MODEL}' に移行します",
            file=sys.stderr,
        )
        try:
            model_path.write_text(DEFAULT_MODEL, encoding="utf-8")
            print(
                f"[INFO] .model ファイルを '{DEFAULT_MODEL}' に更新しました",
                file=sys.stderr,
            )
        except OSError as e:
            print(
                f"[WARN] .model ファイルの更新に失敗しました: {e}",
                file=sys.stderr,
            )
        return DEFAULT_MODEL
    return current_model


def detect_model_config(model_path: Path) -> str:
    """MODEL_CONFIG (.model) からモデル名を読み取り、旧モデルは移行する。"""
    if model_path.exists():
        model = model_path.read_text(encoding="utf-8-sig").strip()
        if model:
            return migrate_model_file(model_path, model)
    return DEFAULT_MODEL


# Fixed Mac comparison candidates. Digests are recorded by the local model
# installation and are checked against Ollama /api/tags before use.
MAC_MODEL_DIGESTS = {
    "qwen3.5:122b": "8b9d11d807c57feb1e2ecb0d6cbf40334c37dcc3523bed5540af4f927f112a37",
    "gpt-oss:120b": "a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9",
}
MAC_OLLAMA_VERSION = "0.34.4"
MODEL_SELECTION_ENV = "OFFLINE_AI_CHAT_MODEL"


class ModelConfigurationError(ValueError):
    """モデル選択または固定モデルidentityの検証に失敗した。"""


@dataclass(frozen=True)
class ModelSelection:
    model: str
    source: str


def canonical_model_reference(model: str) -> str:
    """Ollamaの省略タグを ``:latest`` として正規化する。"""
    normalized = str(model).strip()
    last_segment = normalized.rsplit("/", 1)[-1]
    if ":" not in last_segment and "@" not in last_segment:
        return f"{normalized}:latest"
    return normalized


def _validated_model_name(value: object, source: str) -> str:
    model = str(value).strip()
    if not model or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9._-]+)?", model
    ):
        raise ModelConfigurationError(f"{source} のモデル名が不正です")
    return model


def resolve_model_selection(
    cli_argument: str | None = None,
    model_path: Path | None = None,
    *,
    environ=None,
) -> ModelSelection:
    """CLI > 専用環境変数 > .model > DEFAULT_MODELでモデルを固定する。"""
    env = os.environ if environ is None else environ
    if cli_argument is not None:
        return ModelSelection(
            _validated_model_name(cli_argument, "--model"), "command-line"
        )
    if MODEL_SELECTION_ENV in env:
        return ModelSelection(
            _validated_model_name(env[MODEL_SELECTION_ENV], MODEL_SELECTION_ENV),
            "environment",
        )
    if model_path is not None and model_path.exists():
        try:
            configured = model_path.read_text(encoding="utf-8-sig").strip()
        except OSError as exc:
            raise ModelConfigurationError(".model を読み取れません") from exc
        if configured:
            return ModelSelection(
                _validated_model_name(configured, ".model"), "model-file"
            )
    return ModelSelection(DEFAULT_MODEL, "default")


def resolve_model_reasoning(model: str, requested: str | None) -> dict:
    """モデル固有のthink値とUI表示契約を返す。思考本文は返さない。"""
    value = (
        "off"
        if requested is None or not str(requested).strip()
        else str(requested).strip().lower()
    )
    if value not in {"off", "low", "medium", "high"}:
        raise ModelConfigurationError("reasoning設定が不正です")
    family = model.rsplit("/", 1)[-1].split(":", 1)[0].lower()
    if family == "gpt-oss":
        think = "low" if value == "off" else value
        effective = think
    elif family.startswith("qwen3.5"):
        think = value != "off"
        effective = "enabled" if think else "off"
    else:
        think = value if value in {"low", "medium", "high"} else False
        effective = value
    return {
        "requested": value,
        "think": think,
        "effective": effective,
        "thinking_visible": value != "off",
    }


def validate_model_digest(model: str, model_entries) -> str:
    """固定候補のOllama digestを検証し、確認したdigestを返す。"""
    expected = MAC_MODEL_DIGESTS.get(model)
    if expected is None:
        raise ModelConfigurationError("固定digestが登録されていないモデルです")
    requested = canonical_model_reference(model)
    entry = next(
        (
            item
            for item in model_entries
            if isinstance(item, dict)
            and canonical_model_reference(item.get("name", "")) == requested
        ),
        None,
    )
    if entry is None:
        raise ModelConfigurationError(
            f"選択モデルがOllamaへ導入されていません: {model}"
        )
    observed = str(entry.get("digest", "")).removeprefix("sha256:").lower()
    if observed != expected:
        raise ModelConfigurationError(
            f"選択モデルのdigestが固定値と一致しません: {model}"
        )
    return observed


def model_call_timeout_seconds(model: str, default: float) -> float:
    """大型候補のstall上限を増やし、呼出し元の絶対期限で上限を絞る。"""
    if model not in MAC_MODEL_DIGESTS:
        return float(default)
    raw = os.environ.get("OFFLINE_AI_LARGE_MODEL_CALL_TIMEOUT")
    if raw is None or not raw.strip():
        configured = 300.0
    else:
        try:
            configured = float(raw)
        except ValueError as exc:
            raise ModelConfigurationError(
                "OFFLINE_AI_LARGE_MODEL_CALL_TIMEOUTは数値で指定してください"
            ) from exc
        if (
            configured != configured
            or configured in (float("inf"), float("-inf"))
            or not 1 <= configured <= 600
        ):
            raise ModelConfigurationError(
                "OFFLINE_AI_LARGE_MODEL_CALL_TIMEOUTは1〜600秒で指定してください"
            )
    return max(float(default), configured)
