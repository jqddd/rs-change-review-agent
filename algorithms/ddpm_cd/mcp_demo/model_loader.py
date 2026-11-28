from pathlib import Path
import sys

ALGORITHM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM.parents[1]))
from algorithms.ddpm_cd.inference import DDPMCD


def load_model() -> DDPMCD:
    return DDPMCD(device="cpu")
