#!/usr/bin/env python3
"""Execute a TensorRT MiniInfer engine and validate its final output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

_tensorrt_dll_directory = None
if os.name == "nt":
    tensorrt_lib_dir = os.environ.get("WARPFORGE_TENSORRT_LIB_DIR")
    if tensorrt_lib_dir:
        os.environ["PATH"] = tensorrt_lib_dir + os.pathsep + os.environ["PATH"]
        _tensorrt_dll_directory = os.add_dll_directory(tensorrt_lib_dir)

import tensorrt as trt

from benchmark_miniinfer_pytorch import tolerance, validate_tensor
from miniinfer_model import load_fixture


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("benchmarks/fixtures/miniinfer_full"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def torch_dtype(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TensorRT tensor dtype: {dtype}")
    return mapping[dtype]


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    fixture = load_fixture(arguments.fixture_dir)
    engine_payload = arguments.engine.read_bytes()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_payload)
    if engine is None:
        raise RuntimeError("failed to deserialize TensorRT engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")

    input_names = []
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    if input_names != ["input"] or output_names != ["output"]:
        raise RuntimeError(
            f"unexpected TensorRT I/O tensors: inputs={input_names}, outputs={output_names}"
        )

    expected_shape = (
        fixture.config.batch,
        fixture.config.sequence,
        fixture.config.hidden_size,
    )
    input_shape = tuple(engine.get_tensor_shape("input"))
    output_shape = tuple(engine.get_tensor_shape("output"))
    if input_shape != expected_shape or output_shape != expected_shape:
        raise RuntimeError(
            f"unexpected TensorRT shapes: input={input_shape}, output={output_shape}"
        )

    input_dtype = torch_dtype(engine.get_tensor_dtype("input"))
    output_dtype = torch_dtype(engine.get_tensor_dtype("output"))
    input_tensor = torch.from_numpy(fixture.tensors["input"]).to(
        device="cuda", dtype=input_dtype
    )
    output_tensor = torch.empty(expected_shape, device="cuda", dtype=output_dtype)
    context.set_tensor_address("input", input_tensor.data_ptr())
    context.set_tensor_address("output", output_tensor.data_ptr())
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT execute_async_v3 returned false")
    stream.synchronize()

    if arguments.precision == "fp16":
        absolute_tolerance, relative_tolerance = 1.0e-2, 1.0e-2
    else:
        absolute_tolerance, relative_tolerance = tolerance("output", fixture.config)
    validation = validate_tensor(
        fixture.tensors["output"],
        output_tensor.float().cpu().numpy(),
        absolute_tolerance,
        relative_tolerance,
    )
    if not validation["passed"]:
        raise RuntimeError(
            "TensorRT output failed validation: "
            f"{validation['failure_count']} failures, "
            f"max error {validation['max_absolute_error']}"
        )

    result = {
        "schema_version": 1,
        "precision": arguments.precision,
        "tensorrt_version": trt.__version__,
        "engine_file": arguments.engine.name,
        "engine_sha256": hashlib.sha256(engine_payload).hexdigest(),
        "engine_bytes": len(engine_payload),
        "engine_device_memory_bytes": int(engine.device_memory_size_v2),
        "input_name": "input",
        "input_shape": list(input_shape),
        "input_dtype": str(engine.get_tensor_dtype("input")),
        "output_name": "output",
        "output_shape": list(output_shape),
        "output_dtype": str(engine.get_tensor_dtype("output")),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "correctness": validation,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"TensorRT {arguments.precision} validation: PASS; "
        f"max error {validation['max_absolute_error']:.9g}"
    )


if __name__ == "__main__":
    main()
