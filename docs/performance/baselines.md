# MiniInfer Library and Framework Baselines

## Scope

Stage 10 compares the fixed Stage 9 MiniInfer block with three external execution
paths while retaining the custom and cuBLAS native baselines. Every path uses batch
1, sequence length 128, hidden size 512, 8 heads, head dimension 64, intermediate
size 1536, RMSNorm epsilon `1e-5`, and the deterministic seed-2027 fixture.

The compared paths are:

- WarpForge with custom register-blocked projection GEMMs;
- WarpForge with cuBLAS projection GEMMs;
- eager PyTorch CUDA in inference mode; and
- TensorRT 10.7 through a fixed-shape, standard-operator ONNX graph.

This stage adds no TensorRT plugin and no direct cuDNN benchmark. The exported
graph parsed successfully without a plugin, and no standalone cuDNN primitive has
the same whole-block semantics, so either addition would answer a different
question.

The Stage 5 operation-level custom/cuBLAS GEMM ladder and its committed evidence
remain unchanged. Stage 10 adds a fresh whole-block custom-versus-cuBLAS comparison
from the same clean commit used for the framework baselines.

## Audited Dependency Stack

The approved Stage 10 environment is isolated in ignored, repository-local
directories. It does not modify the global Python or CUDA installation.

| Component | Version or observation |
| --- | --- |
| Python | CPython 3.12.13 in `.venv-stage10` |
| NumPy | 2.3.3 |
| PyTorch | 2.14.0+cu126 from the official CUDA 12.6 wheel index |
| PyTorch CUDA runtime | 12.6 |
| ONNX | 1.16.0 |
| TensorRT | 10.7.0.23 Windows CUDA 12.6 archive; Python API 10.7.0 |
| `trtexec` | TensorRT 10.7.0 (`v100700`) |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, compute capability 8.6, 4 GB |
| Driver / Toolkit | 616.92 / CUDA Toolkit 12.6.85 |

Exact Python packages are in `python/requirements-stage10.lock`. The TensorRT ZIP
was downloaded from NVIDIA's TensorRT 10.7 release link. The 1,289,086,747-byte
archive had SHA-256
`fbdef004578e7ccd5ee51fe7f846b57422364a743372fd8f9f1d7dbd33f62879`.
The extracted SDK, Python environment, ONNX model, and serialized engines remain
ignored because they are platform-specific generated dependencies or artifacts.

Primary compatibility references:

- [PyTorch CUDA 12.6 wheel index](https://download.pytorch.org/whl/cu126/torch/)
- [TensorRT 10.7 release README](https://github.com/NVIDIA/TensorRT/blob/release/10.7/README.md)
- [TensorRT 10.7 support matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1070/support-matrix/index.html)

## Equivalent Model and ONNX Export

`python/miniinfer_model.py` reconstructs the Stage 9 operation order and loads the
same checked fixture bytes. It disables TF32 and validates all 18 named
intermediates on CPU and CUDA before measuring PyTorch. The model is in evaluation
and inference mode; it contains no dropout, autograd work, compilation, or CUDA
Graph capture.

The fixed-shape opset-17 export has one `input` and one `output`, both shaped
`[1, 128, 512]`. ONNX 1.16 validates the graph. It contains 111 nodes drawn from
18 standard operator types and no custom operator domain. The 13,694,342-byte
model has SHA-256
`c2805b95e0af22ff1c121830a7bbfbbfe1e2e48d0045f011c8bb4d92c49a1884`.

TensorRT builds two engines from that graph:

- FP32 with TF32 disabled; and
- mixed FP32/FP16 with FP16 tactics enabled and FP32 graph input/output.

TensorRT may retain FP32 for numerically sensitive layers. In particular, its
builder warned about FP16 layer-normalization reductions. “TensorRT FP16” below
therefore means the supported mixed-precision engine, not an assertion that every
layer and tensor is FP16.

## Correctness Gate

Correctness was completed before `trtexec` timing.

| Path | Validation extent | Maximum absolute error | Failing elements |
| --- | --- | ---: | ---: |
| PyTorch CPU | all 18 intermediates | 0 | 0 |
| WarpForge custom | all 18 intermediates | 3.099442e-6 | 0 |
| WarpForge cuBLAS | all 18 intermediates | 2.145767e-6 | 0 |
| PyTorch CUDA eager | all 18 intermediates | 1.788139e-6 | 0 |
| TensorRT FP32 | final output, 65,536 elements | 7.152557e-7 | 0 |
| TensorRT FP16 mixed | final output, 65,536 elements | 5.970299e-4 | 0 |

The native and PyTorch paths use the same operation-aware Stage 9 tolerances.
TensorRT FP32 uses the Stage 9 output tolerance (`atol=0.0055`, `rtol=0.001`), and
the mixed FP16 path uses `atol=0.01`, `rtol=0.01`. Serialized engine validation is
recorded separately from performance so a fast but incorrect engine cannot enter
the comparison.

## Measurement Boundaries

All committed results were generated from clean implementation commit `f637452`.
Native and PyTorch runs use 50 warmup iterations and 500 measured samples per
scope. `trtexec` accepts warmup duration rather than an iteration count, so it uses
200 ms, which produced at least 304 warmup queries in each reported run, followed
by exactly 500 exported timing records. Clocks were not locked.

The boundaries are deliberately explicit:

- Native and PyTorch `kernel-sequence-device-resident` records use CUDA events
  around a complete forward pass with input, weights, and output already on the
  GPU. Allocation, transfers, fixture loading, validation, and serialization are
  excluded.
- Native and PyTorch `end-to-end-h2d-forward-d2h` records use a host steady clock
  around a pageable input H2D copy, forward pass, pageable output D2H copy, and
  completion. Allocation, fixture loading, model/weight upload, and validation are
  excluded.
- TensorRT `gpu-compute-device-resident` records use each `trtexec` trace's
  `computeMs` with data transfers disabled.
- TensorRT `gpu-h2d-compute-d2h` records use `latencyMs`, the sum of the tool's GPU
  H2D, compute, and D2H intervals. This excludes host enqueue and engine startup,
  so it is not directly equivalent to the native/PyTorch host-clock end-to-end
  boundary.

All paths use one inference stream and no CUDA Graph. TensorRT's serialized-engine
load/deserialization is outside measurement.

## Measured Results

### Device-resident GPU work

| Path | Precision | Median ms | p95 ms | Tokens/s | Max abs. error |
| --- | --- | ---: | ---: | ---: | ---: |
| WarpForge custom | FP32 | 1.350656 | 1.760472 | 94,769 | 3.099e-6 |
| WarpForge cuBLAS | FP32 | 0.746400 | 0.848013 | 171,490 | 2.146e-6 |
| PyTorch eager | FP32 | 1.298944 | 2.028168 | 98,542 | 1.788e-6 |
| TensorRT | FP32 | 0.311310 | 0.607546 | 411,166 | 7.153e-7 |
| TensorRT | mixed FP16 | 0.233490 | 0.324615 | 548,203 | 5.970e-4 |

At this one shape and run, cuBLAS makes the native block `1.810x` faster than the
custom-GEMM path. PyTorch eager is `1.040x` faster than the native custom path but
slower than the native cuBLAS path. TensorRT FP32 is `2.398x` faster than the
native cuBLAS sequence by median GPU work, and enabling supported FP16 tactics is
another `1.333x` faster than TensorRT FP32.

These ratios describe complete execution paths, not just GEMM quality. TensorRT
can optimize and fuse the standard graph globally, while the native and eager
paths launch their explicit sequences.

### Transfer-inclusive records

| Path | Boundary | Median ms | p95 ms | Tokens/s |
| --- | --- | ---: | ---: | ---: |
| WarpForge custom FP32 | host H2D + forward + D2H + completion | 1.512300 | 2.020210 | 84,639 |
| WarpForge cuBLAS FP32 | host H2D + forward + D2H + completion | 0.899000 | 1.111050 | 142,380 |
| PyTorch eager FP32 | host H2D + forward + D2H + completion | 1.298050 | 2.118970 | 98,609 |
| TensorRT FP32 | GPU H2D + compute + D2H | 0.348389 | 0.404862 | 367,405 |
| TensorRT mixed FP16 | GPU H2D + compute + D2H | 0.281502 | 0.325329 | 454,705 |

PyTorch's host-clock transfer-inclusive median is 0.000894 ms below its separately
run device-event median. That difference is smaller than the run's dispersion and
does not imply negative copy cost. The two scopes were executed sequentially on a
laptop GPU with unlocked clocks; the p95 and standard deviations preserve the
observed variability.

For TensorRT's transfer-inclusive traces, mean FP32 H2D/compute/D2H intervals are
0.028131/0.304571/0.023278 ms. Mixed FP16 reports
0.027846/0.237423/0.023291 ms. These components explain the tool-local latency but
remain distinct from process or host-dispatch latency.

## Memory Observations

Memory numbers expose different ownership boundaries and must not be treated as a
single apples-to-apples peak metric:

- WarpForge records 13,635,584 parameter bytes and a 6,820,079-byte activation
  workspace capacity (6,815,744 bytes used), excluding CUDA/cuBLAS runtime state.
- PyTorch reports 29,544,448 allocator bytes before forward, 36,884,480 peak bytes,
  and a 7,340,032-byte incremental forward peak. Its model buffers occupy
  13,684,736 bytes.
- TensorRT's FP32 and mixed-FP16 engine files are 14,194,508 and 14,384,316 bytes;
  each execution context reports 3,145,728 device-memory bytes. Engine weights and
  TensorRT/CUDA runtime allocations are not a comprehensive process peak.

## Reproduction

After creating `.venv-stage10` from CPython 3.12 and installing the locked Python
packages plus the matching TensorRT SDK wheel, the core commands are:

```powershell
.venv-stage10\Scripts\python.exe python\benchmark_miniinfer_pytorch.py `
  --fixture-dir benchmarks\fixtures\miniinfer_full --warmups 50 `
  --iterations 500 --output-dir benchmarks\results\stage10\pytorch

.venv-stage10\Scripts\python.exe python\export_miniinfer_onnx.py `
  --fixture-dir benchmarks\fixtures\miniinfer_full `
  --output benchmarks\generated\stage10\miniinfer_fp32.onnx `
  --manifest-output benchmarks\results\stage10\onnx_manifest.json
```

Build each engine with TensorRT 10.7 `trtexec`, using `--noTF32` for FP32 or
`--fp16` for mixed precision, then run `python/validate_tensorrt_engine.py` before
timing. Report traces use `--warmUp=200 --duration=0 --iterations=500
--avgRuns=500 --percentile=95`; the device-resident run additionally uses
`--noDataTransfers`. `python/summarize_trtexec.py` converts each exported trace,
and `python/assemble_stage10_results.py benchmarks/results/stage10` checks all ten
records and writes the combined table.

The authoritative summary and per-sample JSON are in
[`benchmarks/results/stage10/`](../../benchmarks/results/stage10/summary.csv).
Generated ONNX and engine binaries are intentionally excluded from Git.

## Verification and Conclusions

- All 31 Release CTests pass after the clean-commit rebuild.
- The PyTorch CPU and CUDA paths validate every named intermediate.
- ONNX checker passes; the graph has no custom domain.
- TensorRT FP32 and mixed-FP16 engines validate before any report timing.
- The Stage 10 assembler verifies 10 records, 500 samples per record, seed 2027,
  fixed dimensions, passing correctness, and one shared clean code commit.
- `git diff --check` passes.

Stage 10 is a complete baseline, not a profiler diagnosis. The RTX 3050 Laptop GPU
showed substantial p95 variability with unlocked clocks, especially in some
device-only runs. Stage 11 will use Nsight Compute and Nsight Systems to connect
the observed execution differences to kernel, launch, memory, and synchronization
evidence rather than inferring causes from latency alone.
