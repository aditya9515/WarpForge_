#!/usr/bin/env python3
"""Generate deterministic FP32 MiniInfer fixtures with a PyTorch reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


PRESETS = {
    "small": {
        "batch": 1,
        "sequence": 8,
        "hidden_size": 32,
        "attention_heads": 4,
        "head_dimension": 8,
        "intermediate_size": 64,
        "rms_norm_epsilon": 1.0e-5,
        "rope_base": 10000.0,
        "position_offset": 0,
    },
    "full": {
        "batch": 1,
        "sequence": 128,
        "hidden_size": 512,
        "attention_heads": 8,
        "head_dimension": 64,
        "intermediate_size": 1536,
        "rms_norm_epsilon": 1.0e-5,
        "rope_base": 10000.0,
        "position_offset": 0,
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    inverse_rms = torch.rsqrt(torch.mean(value * value, dim=-1, keepdim=True) + epsilon)
    return value * inverse_rms * weight


def rope(value: torch.Tensor, base: float, position_offset: int) -> torch.Tensor:
    head_dimension = value.shape[-1]
    pair = torch.arange(head_dimension // 2, dtype=torch.float32)
    inverse_frequency = torch.pow(
        torch.tensor(base, dtype=torch.float32),
        -2.0 * pair / float(head_dimension),
    )
    positions = torch.arange(
        position_offset,
        position_offset + value.shape[1],
        dtype=torch.float32,
    )
    angle = positions[:, None] * inverse_frequency[None, :]
    cosine = torch.cos(angle)[None, :, None, :]
    sine = torch.sin(angle)[None, :, None, :]
    first = value[..., 0::2]
    second = value[..., 1::2]
    output = torch.empty_like(value)
    output[..., 0::2] = first * cosine - second * sine
    output[..., 1::2] = first * sine + second * cosine
    return output


def random_uniform(
    generator: torch.Generator,
    shape: tuple[int, ...],
    low: float,
    high: float,
) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=generator)


def create_tensors(config: dict[str, int | float], seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    batch = int(config["batch"])
    sequence = int(config["sequence"])
    hidden = int(config["hidden_size"])
    heads = int(config["attention_heads"])
    head_dimension = int(config["head_dimension"])
    intermediate = int(config["intermediate_size"])
    epsilon = float(config["rms_norm_epsilon"])
    rope_base = float(config["rope_base"])
    position_offset = int(config["position_offset"])
    tokens = batch * sequence

    tensors: dict[str, torch.Tensor] = {}
    tensors["input"] = random_uniform(generator, (batch, sequence, hidden), -0.5, 0.5)
    tensors["attention_norm_weight"] = random_uniform(generator, (hidden,), 0.75, 1.25)
    tensors["ffn_norm_weight"] = random_uniform(generator, (hidden,), 0.75, 1.25)

    def weight(rows: int, columns: int) -> torch.Tensor:
        bound = 1.0 / math.sqrt(float(rows))
        return random_uniform(generator, (rows, columns), -bound, bound)

    tensors["query_weight"] = weight(hidden, hidden)
    tensors["key_weight"] = weight(hidden, hidden)
    tensors["value_weight"] = weight(hidden, hidden)
    tensors["output_weight"] = weight(hidden, hidden)
    tensors["gate_weight"] = weight(hidden, intermediate)
    tensors["up_weight"] = weight(hidden, intermediate)
    tensors["down_weight"] = weight(intermediate, hidden)

    input_tensor = tensors["input"]
    tensors["input_norm"] = rms_norm(
        input_tensor, tensors["attention_norm_weight"], epsilon
    )
    normalized_2d = tensors["input_norm"].reshape(tokens, hidden)
    tensors["query"] = (normalized_2d @ tensors["query_weight"]).reshape(
        batch, sequence, heads, head_dimension
    )
    tensors["key"] = (normalized_2d @ tensors["key_weight"]).reshape(
        batch, sequence, heads, head_dimension
    )
    tensors["value"] = (normalized_2d @ tensors["value_weight"]).reshape(
        batch, sequence, heads, head_dimension
    )
    tensors["query_rope"] = rope(tensors["query"], rope_base, position_offset)
    tensors["key_rope"] = rope(tensors["key"], rope_base, position_offset)

    query_heads = tensors["query_rope"].permute(0, 2, 1, 3)
    key_heads = tensors["key_rope"].permute(0, 2, 1, 3)
    scale = torch.tensor(1.0 / math.sqrt(float(head_dimension)), dtype=torch.float32)
    tensors["attention_scores"] = torch.matmul(
        query_heads, key_heads.transpose(-2, -1)
    ) * scale
    causal_mask = torch.triu(
        torch.ones((sequence, sequence), dtype=torch.bool), diagonal=1
    )
    tensors["masked_scores"] = tensors["attention_scores"].masked_fill(
        causal_mask[None, None, :, :], -torch.inf
    )
    tensors["attention_probabilities"] = torch.softmax(tensors["masked_scores"], dim=-1)
    value_heads = tensors["value"].permute(0, 2, 1, 3)
    tensors["attention_context"] = torch.matmul(
        tensors["attention_probabilities"], value_heads
    ).permute(0, 2, 1, 3).contiguous()
    tensors["attention_output"] = (
        tensors["attention_context"].reshape(tokens, hidden) @ tensors["output_weight"]
    ).reshape(batch, sequence, hidden)
    tensors["attention_residual"] = input_tensor + tensors["attention_output"]
    tensors["post_attention_norm"] = rms_norm(
        tensors["attention_residual"], tensors["ffn_norm_weight"], epsilon
    )
    post_attention_2d = tensors["post_attention_norm"].reshape(tokens, hidden)
    tensors["mlp_gate"] = (
        post_attention_2d @ tensors["gate_weight"]
    ).reshape(batch, sequence, intermediate)
    tensors["mlp_up"] = (
        post_attention_2d @ tensors["up_weight"]
    ).reshape(batch, sequence, intermediate)
    tensors["mlp_swiglu"] = functional.silu(tensors["mlp_gate"]) * tensors["mlp_up"]
    tensors["mlp_output"] = (
        tensors["mlp_swiglu"].reshape(tokens, intermediate) @ tensors["down_weight"]
    ).reshape(batch, sequence, hidden)
    tensors["output"] = tensors["attention_residual"] + tensors["mlp_output"]
    return tensors


def write_fixture(
    output_directory: Path,
    preset: str,
    config: dict[str, int | float],
    seed: int,
    tensors: dict[str, torch.Tensor],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    weights = {
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
    manifest_tensors: dict[str, dict[str, object]] = {}
    for name, tensor in tensors.items():
        contiguous = tensor.detach().cpu().contiguous()
        array = contiguous.numpy().astype("<f4", copy=False)
        payload = array.tobytes(order="C")
        filename = f"{name}.bin"
        (output_directory / filename).write_bytes(payload)
        role = "weight" if name in weights else ("input" if name == "input" else "intermediate")
        manifest_tensors[name] = {
            "file": filename,
            "role": role,
            "dtype": "fp32-le",
            "shape": list(contiguous.shape),
            "element_count": contiguous.numel(),
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    manifest = {
        "schema_version": 1,
        "preset": preset,
        "seed": seed,
        "generator": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "config": config,
        "tensors": manifest_tensors,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    arguments = parse_arguments()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = PRESETS[arguments.preset].copy()
    tensors = create_tensors(config, arguments.seed)
    write_fixture(
        arguments.output_dir,
        arguments.preset,
        config,
        arguments.seed,
        tensors,
    )
    print(f"Wrote {len(tensors)} tensors to {arguments.output_dir}")


if __name__ == "__main__":
    main()
