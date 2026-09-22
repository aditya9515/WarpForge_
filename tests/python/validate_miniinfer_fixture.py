#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path


EXPECTED_WEIGHTS = {
    "attention_norm_weight",
    "ffn_norm_weight",
    "query_weight",
    "key_weight",
    "value_weight",
    "output_weight",
    "gate_weight",
    "up_weight",
    "down_weight",
}
EXPECTED_INTERMEDIATES = {
    "input_norm",
    "query",
    "key",
    "value",
    "query_rope",
    "key_rope",
    "attention_scores",
    "masked_scores",
    "attention_probabilities",
    "attention_context",
    "attention_output",
    "attention_residual",
    "post_attention_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_swiglu",
    "mlp_output",
    "output",
}


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        assert isinstance(value, int) and value > 0
        result *= value
    return result


def validate(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["seed"] == 2027
    assert manifest["generator"]["numpy"] == "2.3.3"
    assert manifest["generator"]["torch"] == "2.14.0+cpu"
    assert manifest["generator"]["torch_cuda"] is None
    config = manifest["config"]
    assert config["hidden_size"] == config["attention_heads"] * config["head_dimension"]
    assert config["head_dimension"] % 2 == 0
    assert config["rms_norm_epsilon"] > 0.0
    assert config["rope_base"] > 0.0

    tensors = manifest["tensors"]
    assert set(tensors) == EXPECTED_WEIGHTS | EXPECTED_INTERMEDIATES | {"input"}
    for name, metadata in tensors.items():
        assert metadata["dtype"] == "fp32-le"
        assert metadata["element_count"] == product(metadata["shape"])
        assert metadata["byte_count"] == metadata["element_count"] * 4
        payload = (directory / metadata["file"]).read_bytes()
        assert len(payload) == metadata["byte_count"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        if name != "masked_scores":
            values = struct.iter_unpack("<f", payload)
            assert all(math.isfinite(value[0]) for value in values)

    probability_metadata = tensors["attention_probabilities"]
    payload = (directory / probability_metadata["file"]).read_bytes()
    values = [value[0] for value in struct.iter_unpack("<f", payload)]
    sequence = config["sequence"]
    rows = config["batch"] * config["attention_heads"] * sequence
    assert len(values) == rows * sequence
    for row in range(rows):
        row_sum = sum(values[row * sequence : (row + 1) * sequence])
        assert abs(row_sum - 1.0) <= 2.0e-5


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_miniinfer_fixture.py <fixture-directory>")
    validate(Path(sys.argv[1]))
