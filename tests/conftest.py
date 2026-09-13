import sys
from pathlib import Path

OFFLINE_AI_ROOT = Path(__file__).resolve().parent.parent
INTERNAL_DIR = OFFLINE_AI_ROOT / "_internal"
sys.path.insert(0, str(INTERNAL_DIR))

EVAL_DIR = OFFLINE_AI_ROOT / "tests" / "eval"
sys.path.insert(0, str(EVAL_DIR))
