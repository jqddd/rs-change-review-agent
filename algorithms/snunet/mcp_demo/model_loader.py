from pathlib import Path
import sys

ALGORITHM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM.parents[1]))
from algorithms.snunet.inference import SNUNet


def load_model() -> SNUNet:
    return SNUNet(device="cpu")
