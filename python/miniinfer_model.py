"""Shared PyTorch definition and fixture loading for the MiniInfer block."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


INTERMEDIATE_NAMES = (
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
)

WEIGHT_NAMES = (
    "attention_norm_weight",
    "query_weight",
    "key_weight",
    "value_weight",
    "output_weight",
    "ffn_norm_weight",
    "gate_weight",
    "up_weight",
    "down_weight",
)


@dataclass(frozen=True)
class MiniInferConfig:
    batch: int
    sequence: int
    hidden_size: int
    attention_heads: int
    head_dimension: int
    intermediate_size: int
    rms_norm_epsilon: float
    rope_base: float
    position_offset: int


@dataclass(frozen=True)
class MiniInferFixture:
    directory: Path
    manifest: dict[str, object]
    config: MiniInferConfig
    tensors: dict[str, np.ndarray]


def load_fixture(directory: Path) -> MiniInferFixture:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported MiniInfer fixture schema")
    config = MiniInferConfig(**manifest["config"])
    if config.hidden_size != config.attention_heads * config.head_dimension:
        raise ValueError("fixture attention width does not equal hidden size")
    if config.head_dimension % 2 != 0:
        raise ValueError("fixture head dimension must be even")

    tensors: dict[str, np.ndarray] = {}
    for name, metadata in manifest["tensors"].items():
        path = directory / metadata["file"]
        payload = path.read_bytes()
        if len(payload) != metadata["byte_count"]:
            raise ValueError(f"fixture byte count mismatch for {name}")
        if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
            raise ValueError(f"fixture SHA-256 mismatch for {name}")
        array = np.frombuffer(payload, dtype="<f4").copy()
        if array.size != metadata["element_count"]:
            raise ValueError(f"fixture element count mismatch for {name}")
        tensors[name] = array.reshape(metadata["shape"])
    return MiniInferFixture(directory, manifest, config, tensors)


class MiniInferBlock(torch.nn.Module):
    """Fixed-shape, bias-free, pre-norm LLaMA-style reference block."""

    def __init__(self, fixture: MiniInferFixture) -> None:
        super().__init__()
        self.config = fixture.config
        for name in WEIGHT_NAMES:
            self.register_buffer(name, torch.from_numpy(fixture.tensors[name]).clone())

        config = self.config
        pair = torch.arange(config.head_dimension // 2, dtype=torch.float32)
        inverse_frequency = torch.pow(
            torch.tensor(config.rope_base, dtype=torch.float32),
            -2.0 * pair / float(config.head_dimension),
        )
        positions = torch.arange(
            config.position_offset,
            config.position_offset + config.sequence,
            dtype=torch.float32,
        )
        angle = positions[:, None] * inverse_frequency[None, :]
        self.register_buffer("rope_cosine", torch.cos(angle)[None, :, None, :])
        self.register_buffer("rope_sine", torch.sin(angle)[None, :, None, :])
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.sequence, config.sequence, dtype=torch.bool),
                diagonal=1,
            )[None, None, :, :],
        )

    def _rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(
            torch.mean(value * value, dim=-1, keepdim=True)
            + self.config.rms_norm_epsilon
        )
        return value * inverse_rms * weight

    def _rope(self, value: torch.Tensor) -> torch.Tensor:
        config = self.config
        pairs = value.reshape(
            config.batch,
            config.sequence,
            config.attention_heads,
            config.head_dimension // 2,
            2,
        )
        first = pairs[..., 0]
        second = pairs[..., 1]
        rotated = torch.stack(
            (
                first * self.rope_cosine - second * self.rope_sine,
                first * self.rope_sine + second * self.rope_cosine,
            ),
            dim=-1,
        )
        return rotated.flatten(-2)

    def _execute(
        self,
        input_tensor: torch.Tensor,
        capture_intermediates: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        config = self.config
        if not torch.jit.is_tracing() and tuple(input_tensor.shape) != (
            config.batch,
            config.sequence,
            config.hidden_size,
        ):
            raise ValueError("MiniInfer input shape does not match the fixed configuration")
        intermediates: dict[str, torch.Tensor] | None = (
            {} if capture_intermediates else None
        )

        def capture(name: str, value: torch.Tensor) -> torch.Tensor:
            if intermediates is not None:
                intermediates[name] = value
            return value

        input_norm = capture(
            "input_norm", self._rms_norm(input_tensor, self.attention_norm_weight)
        )
        tokens = config.batch * config.sequence
        normalized_2d = input_norm.reshape(tokens, config.hidden_size)
        query = capture(
            "query",
            (normalized_2d @ self.query_weight).reshape(
                config.batch,
                config.sequence,
                config.attention_heads,
                config.head_dimension,
            ),
        )
        key = capture(
            "key",
            (normalized_2d @ self.key_weight).reshape(
                config.batch,
                config.sequence,
                config.attention_heads,
                config.head_dimension,
            ),
        )
        value = capture(
            "value",
            (normalized_2d @ self.value_weight).reshape(
                config.batch,
                config.sequence,
                config.attention_heads,
                config.head_dimension,
            ),
        )
        query_rope = capture("query_rope", self._rope(query))
        key_rope = capture("key_rope", self._rope(key))

        query_heads = query_rope.permute(0, 2, 1, 3)
        key_heads = key_rope.permute(0, 2, 1, 3)
        attention_scores = capture(
            "attention_scores",
            torch.matmul(query_heads, key_heads.transpose(-2, -1))
            * (1.0 / math.sqrt(float(config.head_dimension))),
        )
        masked_scores = capture(
            "masked_scores", attention_scores.masked_fill(self.causal_mask, -torch.inf)
        )
        probabilities = capture(
            "attention_probabilities", torch.softmax(masked_scores, dim=-1)
        )
        value_heads = value.permute(0, 2, 1, 3)
        context = capture(
            "attention_context",
            torch.matmul(probabilities, value_heads)
            .permute(0, 2, 1, 3)
            .contiguous(),
        )
        attention_output = capture(
            "attention_output",
            (context.reshape(tokens, config.hidden_size) @ self.output_weight).reshape(
                config.batch, config.sequence, config.hidden_size
            ),
        )
        attention_residual = capture(
            "attention_residual", input_tensor + attention_output
        )
        post_attention_norm = capture(
            "post_attention_norm",
            self._rms_norm(attention_residual, self.ffn_norm_weight),
        )
        post_attention_2d = post_attention_norm.reshape(tokens, config.hidden_size)
        mlp_gate = capture(
            "mlp_gate",
            (post_attention_2d @ self.gate_weight).reshape(
                config.batch, config.sequence, config.intermediate_size
            ),
        )
        mlp_up = capture(
            "mlp_up",
            (post_attention_2d @ self.up_weight).reshape(
                config.batch, config.sequence, config.intermediate_size
            ),
        )
        mlp_swiglu = capture("mlp_swiglu", functional.silu(mlp_gate) * mlp_up)
        mlp_output = capture(
            "mlp_output",
            (mlp_swiglu.reshape(tokens, config.intermediate_size) @ self.down_weight).reshape(
                config.batch, config.sequence, config.hidden_size
            ),
        )
        output = capture("output", attention_residual + mlp_output)
        return output, intermediates

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output, _ = self._execute(input_tensor, False)
        return output

    def forward_with_intermediates(
        self, input_tensor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        _, intermediates = self._execute(input_tensor, True)
        assert intermediates is not None
        return intermediates
