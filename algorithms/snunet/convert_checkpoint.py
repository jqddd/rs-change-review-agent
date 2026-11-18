"""One-time conversion of a trusted upstream full-model pickle to state_dict.

Never invoke this from the Agent. Only convert a checkpoint obtained from a
trusted source. Class remapping is local; no global `models` alias is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "algorithms.snunet"

from .models import Models as upstream_model
from .inference import UPSTREAM_COMMIT


class _UpstreamUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "models.Models":
            if name not in {"SNUNet_ECAM", "conv_block_nested", "up", "ChannelAttention"}:
                raise pickle.UnpicklingError(f"Unexpected upstream class: {name}")
            return getattr(upstream_model, name)
        return super().find_class(module, name)


class _LegacyPickle:
    __name__ = "snunet_legacy_pickle"
    Unpickler = _UpstreamUnpickler
    load = staticmethod(pickle.load)


def convert(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    checkpoint = torch.load(source, map_location="cpu", pickle_module=_LegacyPickle, weights_only=False)
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module
    if not isinstance(checkpoint, upstream_model.SNUNet_ECAM):
        raise ValueError("Expected the official SNUNet_ECAM full-model checkpoint")
    state = checkpoint.state_dict()
    reference = upstream_model.SNUNet_ECAM(in_ch=3, out_ch=2)
    # The official CDD full-model checkpoint includes ca2/ca3/ca4 modules,
    # although neither its embedded original forward nor the pinned upstream
    # forward calls them (only ca and ca1 are used). Preserve these six tensors
    # separately; do not relax validation for any other key or shape mismatch.
    expected = set(reference.state_dict())
    allowed_unused = {f"ca{i}.fc{j}.weight" for i in (2, 3, 4) for j in (1, 2)}
    unexpected = set(state) - expected
    if unexpected - allowed_unused:
        raise ValueError(f"Unexpected checkpoint keys: {sorted(unexpected - allowed_unused)}")
    unused = {key: state.pop(key) for key in sorted(unexpected)}
    reference.load_state_dict(state, strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "source_sha256": digest.hexdigest(),
                "upstream_commit": UPSTREAM_COMMIT,
                "unused_legacy_state": unused}, output)
    print(f"Converted checkpoint: {output}\nSource SHA256: {digest.hexdigest()}")
    if unused:
        print(f"Preserved unused legacy tensors separately: {sorted(unused)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trust-source", action="store_true", help="Acknowledge that legacy pickle executes code; use only a trusted checkpoint")
    args = parser.parse_args()
    if not args.trust_source:
        parser.error("Conversion requires --trust-source for a checkpoint you trust")
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
