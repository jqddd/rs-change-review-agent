from pathlib import Path
import sys

import numpy as np
import torch

ALGORITHM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM))
from inference import BIT


def load_model() -> BIT:
    # The published checkpoint includes the NumPy scalar validation score.
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]):
        return BIT(ALGORITHM / "checkpoints/best_ckpt.pt", device="cpu")
