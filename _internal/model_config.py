"""offline-ai のモデル既定値と .model 移行処理。"""

from pathlib import Path
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
