"""Extract the pinned Open-CD LEVIR C16 checkpoint without executing its pickle.

The upstream training checkpoint includes MMEngine and NumPy pickle objects.
This converter interprets pickle opcodes as inert data, then reconstructs only
the dense CPU tensors in ``state_dict`` from the ZIP's raw storage members.
It never imports a pickle GLOBAL or calls a pickle REDUCE/NEWOBJ/BUILD target.
The original CDD converter is deliberately independent of this conversion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import pickletools
import re
import sys
from typing import Any
import zipfile

import torch


SOURCE_SHA256 = "984bfd07e0a30bbea5cd5a2fec918c994208c63b6c69e67a6592e06871811572"
SOURCE_SIZE = 40_792_765
SOURCE_REVISION = "07de9dd7027827cb47bdbe9bcece7ff9b8da73ad"
SOURCE_URL = (
    "https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/"
    f"{SOURCE_REVISION}/LEVIR-CD/snunet_c16_256x256_40k_levircd.pth"
)
PROFILE = "levir_c16"
_MAX_PICKLE_BYTES = 16 * 1024 * 1024
_MAX_STORAGE_BYTES = 128 * 1024 * 1024
_MAX_OPS = 2_000_000
_MAX_MEMO = 500_000
_MARK = object()


@dataclass(frozen=True)
class _Global:
    module: str
    name: str


@dataclass
class _Expression:
    """A record of an operation, with no reference to executable payload code."""

    operation: str
    target: Any
    args: Any
    states: list = field(default_factory=list)


@dataclass(frozen=True)
class _Storage:
    kind: _Global
    key: str
    location: str
    numel: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_key(key: Any) -> bool:
    if type(key) in (str, int, bool, bytes, type(None)):
        return True
    return type(key) is tuple and all(_safe_key(part) for part in key)


def _parse_pickle(data: bytes) -> dict:
    """Evaluate a bounded subset of the pickle stack language as plain data.

    GLOBAL is a pair of strings. Except the empty OrderedDict constructor,
    REDUCE and NEWOBJ only create expression records; BUILD only records state.
    Unsupported opcodes fail closed. No ``pickle.Unpickler`` is involved.
    """
    _require(len(data) <= _MAX_PICKLE_BYTES, "Checkpoint pickle exceeds size limit")
    stack: list = []
    memo: dict = {}

    def pop_mark() -> list:
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is _MARK:
                result = stack[index + 1:]
                del stack[index:]
                return result
        raise ValueError("Pickle MARK is missing")

    def set_items(target: Any, items: list) -> None:
        _require(type(target) is dict and len(items) % 2 == 0,
                 "Pickle SETITEM(S) requires a dictionary and key/value pairs")
        for index in range(0, len(items), 2):
            key = items[index]
            _require(_safe_key(key), "Unsupported dictionary key in pickle")
            _require(key not in target, "Duplicate dictionary key in pickle")
            target[key] = items[index + 1]

    for count, (opcode, arg, position) in enumerate(pickletools.genops(data)):
        _require(count < _MAX_OPS, "Checkpoint pickle exceeds opcode limit")
        name = opcode.name
        if name == "PROTO":
            _require(arg == 2, "Only the pinned checkpoint's pickle protocol 2 is supported")
        elif name == "MARK":
            stack.append(_MARK)
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4",
                      "BINFLOAT", "BINUNICODE", "SHORT_BINSTRING", "BINSTRING"}:
            stack.append(arg)
        elif name == "NONE":
            stack.append(None)
        elif name == "NEWTRUE":
            stack.append(True)
        elif name == "NEWFALSE":
            stack.append(False)
        elif name == "TUPLE":
            stack.append(tuple(pop_mark()))
        elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
            length = int(name[-1])
            _require(len(stack) >= length, "Pickle tuple stack underflow")
            values = tuple(stack[-length:])
            del stack[-length:]
            stack.append(values)
        elif name in {"BINPUT", "LONG_BINPUT"}:
            _require(type(arg) is int and 0 <= arg < _MAX_MEMO and stack,
                     "Invalid pickle memo write")
            _require(arg not in memo, "Duplicate pickle memo index")
            memo[arg] = stack[-1]
        elif name in {"BINGET", "LONG_BINGET"}:
            _require(arg in memo, "Invalid pickle memo read")
            stack.append(memo[arg])
        elif name == "GLOBAL":
            module, global_name = arg.split(" ", 1)
            stack.append(_Global(module, global_name))
        elif name in {"REDUCE", "NEWOBJ"}:
            arguments, target = stack.pop(), stack.pop()
            _require(type(arguments) is tuple, "Pickle operation arguments must be a tuple")
            if name == "REDUCE" and target == _Global("collections", "OrderedDict"):
                _require(arguments == (), "Only empty OrderedDict construction is supported")
                stack.append({})
            else:
                stack.append(_Expression(name, target, arguments))
        elif name == "BUILD":
            state = stack.pop()
            target = stack[-1]
            if type(target) is _Expression:
                target.states.append(state)
            else:
                # PyTorch attaches module version metadata to its OrderedDict.
                # It is not needed to reconstruct the tensor mapping.
                _require(type(target) is dict and type(state) is dict,
                         "Unsupported pickle BUILD target")
        elif name in {"SETITEM", "SETITEMS"}:
            if name == "SETITEM":
                value, key = stack.pop(), stack.pop()
                items = [key, value]
            else:
                items = pop_mark()
            set_items(stack[-1], items)
        elif name in {"APPEND", "APPENDS"}:
            items = [stack.pop()] if name == "APPEND" else pop_mark()
            _require(type(stack[-1]) is list, "Pickle APPEND(S) requires a list")
            stack[-1].extend(items)
        elif name == "BINPERSID":
            value = stack.pop()
            _require(type(value) is tuple and len(value) == 5 and value[0] == "storage",
                     "Unsupported pickle persistent ID")
            _, kind, key, location, numel = value
            _require(type(kind) is _Global and kind.module == "torch"
                     and kind.name in {"FloatStorage", "LongStorage"},
                     "Only float32/int64 storages are supported")
            _require(type(key) is str and re.fullmatch(r"[0-9]+", key) is not None,
                     "Invalid storage member key")
            _require(type(location) is str and
                     (location == "cpu" or re.fullmatch(r"cuda:[0-9]+", location) is not None),
                     "Unsupported storage device marker")
            _require(type(numel) is int and 0 <= numel <= _MAX_STORAGE_BYTES // 4,
                     "Invalid storage element count")
            stack.append(_Storage(kind, key, location, numel))
        elif name == "STOP":
            _require(position == len(data) - 1 and len(stack) == 1
                     and type(stack[0]) is dict, "Invalid pickle result or trailing data")
            return stack[0]
        else:
            raise ValueError(f"Unsupported pickle opcode: {name}")
    raise ValueError("Checkpoint pickle has no STOP opcode")


def _plain_meta(value: Any, depth: int = 0) -> Any:
    _require(depth <= 12, "Checkpoint metadata nesting limit exceeded")
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float:
        _require(math.isfinite(value), "Nonfinite checkpoint metadata")
        return value
    if type(value) in (list, tuple):
        return [_plain_meta(item, depth + 1) for item in value]
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "Metadata keys must be strings")
        return {key: _plain_meta(item, depth + 1) for key, item in value.items()}
    raise ValueError("Requested checkpoint metadata is not plain data")


def _extract(source: Path) -> tuple:
    """Internal CPU extraction; the public converter additionally pins the hash."""
    _require(sys.byteorder == "little", "This checkpoint requires a little-endian host")
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        _require(0 < len(infos) <= 10_000, "Invalid checkpoint archive member count")
        names = [info.filename for info in infos]
        _require(len(set(names)) == len(names), "Duplicate checkpoint archive members")
        for info in infos:
            _require(info.compress_type == zipfile.ZIP_STORED and not info.flag_bits & 1,
                     "Only unencrypted stored ZIP members are supported")
            _require(0 <= info.file_size <= _MAX_STORAGE_BYTES,
                     "Checkpoint archive member exceeds size limit")
        pickle_names = [name for name in names if name.endswith("/data.pkl")]
        _require(len(pickle_names) == 1, "Expected one checkpoint data.pkl member")
        pickle_name = pickle_names[0]
        root = pickle_name[:-len("data.pkl")]
        _require(re.fullmatch(r"[A-Za-z0-9_-]+/", root) is not None,
                 "Unsupported checkpoint archive prefix")
        _require(all(name.startswith(root) and ".." not in name.split("/")
                     for name in names), "Invalid checkpoint archive member path")
        if root + "byteorder" in names:
            _require(archive.read(root + "byteorder") == b"little",
                     "Unsupported checkpoint byte order")
        _require(archive.getinfo(pickle_name).file_size <= _MAX_PICKLE_BYTES,
                 "Checkpoint pickle exceeds size limit")
        checkpoint = _parse_pickle(archive.read(pickle_name))
        state = checkpoint.get("state_dict")
        _require(type(state) is dict and 0 < len(state) <= 10_000,
                 "Checkpoint must contain a tensor state_dict")
        tensors: dict = {}
        storages: dict = {}
        storage_specs: dict = {}
        for key, value in state.items():
            _require(type(key) is str and len(key) <= 512,
                     "Invalid state_dict tensor name")
            _require(type(value) is _Expression and value.operation == "REDUCE"
                     and value.target == _Global("torch._utils", "_rebuild_tensor_v2")
                     and not value.states and len(value.args) in (6, 7),
                     f"Unsupported tensor construction: {key}")
            storage, offset, shape, stride, requires_grad, hooks = value.args[:6]
            _require(type(storage) is _Storage and type(offset) is int and offset >= 0,
                     f"Invalid tensor storage: {key}")
            _require(type(shape) is tuple and type(stride) is tuple
                     and len(shape) == len(stride) and len(shape) <= 8,
                     f"Invalid tensor dimensions: {key}")
            _require(all(type(item) is int and 0 <= item <= 10_000_000 for item in shape)
                     and all(type(item) is int and 0 <= item <= 100_000_000 for item in stride),
                     f"Invalid tensor shape/stride: {key}")
            _require(type(requires_grad) is bool and
                     (hooks is None or type(hooks) is dict and not hooks)
                     and (len(value.args) == 6 or value.args[6] is None),
                     f"Unsupported tensor attributes: {key}")
            numel = math.prod(shape)
            _require(numel <= _MAX_STORAGE_BYTES // 4, f"Tensor exceeds size limit: {key}")
            if numel:
                last = offset + sum((size - 1) * step for size, step in zip(shape, stride))
                _require(last < storage.numel, f"Tensor view exceeds its storage: {key}")
            else:
                _require(offset <= storage.numel, f"Empty tensor offset exceeds storage: {key}")
            dtype = torch.float32 if storage.kind.name == "FloatStorage" else torch.int64
            itemsize = 4 if dtype == torch.float32 else 8
            storage_bytes = storage.numel * itemsize
            _require(storage_bytes <= _MAX_STORAGE_BYTES, "Storage exceeds size limit")
            spec = (storage.kind, storage.numel, storage.location)
            _require(storage.key not in storage_specs or storage_specs[storage.key] == spec,
                     "Conflicting storage specifications")
            if storage.key not in storages:
                member = root + "data/" + storage.key
                _require(member in names and archive.getinfo(member).file_size == storage_bytes,
                         f"Storage size mismatch: {key}")
                raw = bytearray(archive.read(member))
                storages[storage.key] = (torch.frombuffer(raw, dtype=dtype)
                                         if storage.numel else torch.empty(0, dtype=dtype))
                storage_specs[storage.key] = spec
            tensor = torch.as_strided(storages[storage.key], shape, stride, offset).clone()
            _require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
                     f"Nonfinite tensor values: {key}")
            tensors[key] = tensor
        meta = checkpoint.get("meta")
        _require(type(meta) is dict and type(meta.get("cfg")) is str,
                 "Checkpoint must include its original configuration string")
        config = meta["cfg"]
        selected_meta = {
            key: _plain_meta(meta[key])
            for key in ("epoch", "iter", "seed", "experiment_name", "mmengine_version",
                        "dataset_meta") if key in meta
        }
    return tensors, config, selected_meta


def convert(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    _require(source.stat().st_size == SOURCE_SIZE, "Source is not the pinned Open-CD checkpoint size")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    _require(digest.hexdigest() == SOURCE_SHA256,
             "Source SHA256 does not match the pinned Open-CD LEVIR C16 checkpoint")
    state, config, meta = _extract(source)
    result = {
        "profile": PROFILE,
        "state_dict": state,
        "source_sha256": SOURCE_SHA256,
        "source_revision": SOURCE_REVISION,
        "source_url": SOURCE_URL,
        "upstream_config": config,
        "meta": meta,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing file even if another process
    # creates it during extraction. The result contains tensors and primitives.
    with output.open("xb") as stream:
        torch.save(result, stream)
    print(f"Converted {len(state)} CPU tensors: {output}")
    print(f"Source SHA256: {SOURCE_SHA256}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
