# MiniInfer Single Transformer Block

## Scope

Stage 9 composes the validated WarpForge runtime and kernels into one deterministic,
pre-norm LLaMA-style Transformer block. The report configuration is FP32 with batch
1, sequence length 128, hidden size 512, 8 attention heads, head dimension 64,
intermediate size 1536, and RMSNorm epsilon `1e-5`.

The block executes:

```text
input
  -> RMSNorm
  -> Q/K/V projections
  -> interleaved RoPE
  -> scaled QK^T
  -> causal mask
  -> row-wise Softmax
  -> attention probabilities times V
  -> output projection
  -> residual add
  -> RMSNorm
  -> gate/up projections
  -> fused SwiGLU
  -> down projection
  -> residual add
```

`custom` and `cublas` select the projection-GEMM backend. The remaining operations
use the same WarpForge CUDA kernels in both paths. This distinction matters: the
custom result is a custom-GEMM block path, not a claim that every operation has a
separate vendor-library implementation.

## Runtime and Interface

`MiniInferConfig` describes the fixed block dimensions and numerical constants.
`MiniInferWeights` owns the nine device-resident parameter tensors and caches their
views. `MiniInferWorkspace` carves all 17 internal activation tensors from one
256-byte-aligned `DeviceWorkspace` allocation. `MiniInferBlock` owns both objects,
uploads weights once, and accepts caller-owned input/output views, a GEMM backend,
an explicit cuBLAS handle, and an explicit CUDA stream.

The full configuration contains 3,408,896 FP32 parameters (13,635,584 bytes). Its
activation workspace uses 6,815,744 bytes; the allocation capacity is 6,820,079
bytes including worst-case alignment slack. Input/output allocations, weights, and
workspace are reused across validation, warmups, and measured iterations.

## PyTorch Reference and Fixtures

The approved repository-local `.venv` uses CPython 3.12.13, NumPy 2.3.3, and CPU
PyTorch 2.14.0. Exact direct and transitive versions are recorded in
`python/requirements-stage9.lock`. No model or external weights are downloaded.

`python/generate_miniinfer_fixture.py` uses seed `2027`, one CPU thread, and
deterministic PyTorch algorithms. It creates little-endian FP32 binaries and a JSON
manifest containing shapes, byte counts, roles, and SHA-256 hashes. Git marks
`*.bin` as binary so line-ending conversion cannot alter fixture bytes.

Two presets are committed:

- `small`: batch 1, sequence 8, hidden 32, 4 heads, head dimension 8,
  intermediate 64; used by CTest.
- `full`: the Stage 9 report configuration; used by the executable and report run.

Both fixtures contain the input, nine weights, and these 18 named intermediates:

```text
input_norm, query, key, value, query_rope, key_rope,
attention_scores, masked_scores, attention_probabilities,
attention_context, attention_output, attention_residual,
post_attention_norm, mlp_gate, mlp_up, mlp_swiglu,
mlp_output, output
```

The Python fixture validator checks the schema, pinned generator versions, exact
tensor set, shapes, element/byte counts, hashes, finite values, and Softmax row
sums. The C++ test and report executable copy and compare every named intermediate
for both GEMM backends before timing.

## Correctness Contract

Validation uses the shared combined absolute/relative criterion. Tolerances are
named and operation-aware: early normalization/projection tensors begin at
`2e-4` absolute and relative tolerance, RoPE at `3e-4`/`2e-4`, and attention
scores/probabilities/context at `5e-4`/`5e-4`. Later accumulation paths scale the
absolute tolerance with the logarithm of the largest accumulation dimension.

All 36 backend/intermediate comparisons pass with zero failing elements. The
largest absolute error is `3.099442e-6` for the custom path and `2.145767e-6` for
the cuBLAS path. Relative-error maxima can be large near zero; acceptance is based
on the documented combined tolerance, while the CSV preserves absolute, mean, and
meaningful relative-error statistics for each tensor.

Focused attention tests also cover irregular layouts, non-divisible launch sizes,
CPU/GPU score and value products, invalid dimensions and block sizes, overflow,
null pointers, and forbidden output aliasing. MiniInfer tests cover both backends,
invalid configuration, and stable workspace pointer/capacity across repeated runs.

## Measurement Method

The report was generated from clean code commit `4c4225a` on the RTX 3050 Laptop
GPU with seed `2027`, 50 warmups, and 500 measured samples per backend and scope.

- `kernel-sequence-device-resident` uses CUDA events around the complete device
  forward sequence. Inputs and weights are already resident; allocation, fixture
  loading, host/device transfers, validation, and serialization are excluded.
- `end-to-end-h2d-forward-d2h` uses a host steady clock around one pageable input
  H2D copy, the forward sequence, one pageable output D2H copy, and stream
  completion. Allocation, fixture loading, weight upload, and validation remain
  excluded.
- The cuBLAS handle uses `CUBLAS_PEDANTIC_MATH` so the FP32 reference comparison
  does not silently become a reduced-precision Tensor Core experiment.

The device-resident scope is a sequence of CUDA kernel launches and cuBLAS calls,
not a single fused kernel. Stage 11 will profile that sequence; Stage 9 does not
infer a bottleneck from timing alone.

## Measured Results

| Backend | Timing scope | Minimum ms | Mean ms | Median ms | p95 ms | Std. dev. ms | Tokens/s | Max abs. error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| custom | device-resident sequence | 1.308320 | 1.354734 | 1.321984 | 1.664000 | 0.100304 | 96,824 | 3.099e-6 |
| cuBLAS | device-resident sequence | 0.731168 | 0.741181 | 0.737280 | 0.764642 | 0.010453 | 173,611 | 2.146e-6 |
| custom | H2D + forward + D2H | 1.438000 | 1.474943 | 1.467500 | 1.529260 | 0.024087 | 87,223 | 3.099e-6 |
| cuBLAS | H2D + forward + D2H | 0.846300 | 0.874930 | 0.871500 | 0.927205 | 0.027840 | 146,873 | 2.146e-6 |

At this shape, cuBLAS is 1.793x faster by device-resident median and 1.684x faster
by end-to-end median. The custom path reaches 55.8% of cuBLAS device-resident
throughput. These are observed block-level results for this machine and run, not a
claim about all shapes or GPUs.

The authoritative per-sample JSON, compact summary, and per-intermediate validation
CSV are in [`benchmarks/results/stage9/`](../../benchmarks/results/stage9/summary.csv).

## Verification

- All 31 Release CTests pass, including attention, both MiniInfer backends, the
  small fixture, benchmark smoke run, and result-contract validation.
- Both committed fixture directories pass independent manifest/hash validation.
- Compute Sanitizer 2024.3.0 reports zero errors and zero leaked bytes for the
  focused MiniInfer test across both backends.
- `git diff --check` passes.

## Decisions and Limitations

- Retain both GEMM backends. cuBLAS is the faster practical option; the custom path
  keeps the optimization stack inspectable and independently executable.
- Keep attention score and value products as simple FP32 kernels in Stage 9. Their
  optimization and sequence-level profiler analysis belong to later evidence-driven
  work.
- Keep the block fixed to FP32 and one pre-norm layer. There is no FP16 block,
  batching study, multi-block model, KV cache, tokenizer, text generation, model
  loader, or model download.
- The PyTorch dependency is CPU-only and is used to generate deterministic fixtures,
  not as the Stage 10 CUDA performance baseline.
- Results cover one Windows laptop GPU with unlocked clocks and one report shape.
- End-to-end timing uses pageable host memory and excludes fixture loading, weight
  upload, and allocation; it is an explicit inference-call boundary, not process
  startup latency.

Stage 10 will add equivalent PyTorch CUDA and block-level framework/library
baselines, and will consider TensorRT only after a separate compatibility/install
approval. It will not begin until explicitly requested.
