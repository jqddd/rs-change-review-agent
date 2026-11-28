from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parents[1]
ROOT = MODEL_DIR.parent
sys.path.insert(0, str(ROOT / "scripts" / "stubs"))
sys.path.insert(0, str(MODEL_DIR))
from predict import Change_Perception


def load_model():
    return Change_Perception(device="cpu")
