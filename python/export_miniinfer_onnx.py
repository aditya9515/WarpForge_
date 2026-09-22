#!/usr/bin/env python3
"""Export the fixed-shape MiniInfer reference block to standard ONNX operators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch

from miniinfer_model import MiniInferBlock, load_fixture


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("benchmarks/fixtures/miniinfer_full"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/generated/stage10/miniinfer_fp32.onnx"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Optional manifest path (defaults beside the ONNX model)",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    fixture = load_fixture(arguments.fixture_dir)
    model = MiniInferBlock(fixture).eval()
    input_tensor = torch.from_numpy(fixture.tensors["input"]).clone()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (input_tensor,),
            arguments.output,
            export_params=True,
            opset_version=arguments.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
            dynamo=False,
        )

    model_proto = onnx.load(arguments.output)
    onnx.checker.check_model(model_proto)
    custom_domains = sorted(
        {node.domain for node in model_proto.graph.node if node.domain not in {"", "ai.onnx"}}
    )
    if custom_domains:
        raise RuntimeError(f"ONNX graph contains custom domains: {custom_domains}")
    input_shape = [dimension.dim_value for dimension in model_proto.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in model_proto.graph.output[0].type.tensor_type.shape.dim]
    expected_shape = [
        fixture.config.batch,
        fixture.config.sequence,
        fixture.config.hidden_size,
    ]
    if input_shape != expected_shape or output_shape != expected_shape:
        raise RuntimeError("ONNX input/output shape is not the required fixed shape")

    payload = arguments.output.read_bytes()
    metadata = {
        "schema_version": 1,
        "onnx_file": arguments.output.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "opset": arguments.opset,
        "input_name": model_proto.graph.input[0].name,
        "input_shape": input_shape,
        "output_name": model_proto.graph.output[0].name,
        "output_shape": output_shape,
        "node_count": len(model_proto.graph.node),
        "operator_types": sorted({node.op_type for node in model_proto.graph.node}),
        "custom_domains": custom_domains,
    }
    metadata_path = (
        arguments.manifest_output
        if arguments.manifest_output is not None
        else arguments.output.with_suffix(".manifest.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"Exported {arguments.output} ({len(payload)} bytes, "
        f"{metadata['node_count']} nodes, SHA-256 {metadata['sha256']})"
    )


if __name__ == "__main__":
    main()
