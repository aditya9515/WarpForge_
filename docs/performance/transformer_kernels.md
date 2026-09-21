# Transformer-Oriented CUDA Kernels

## Scope and Semantics

Stage 6 adds FP32 CPU references and CUDA implementations for the primitives needed by a later LLaMA-style block without assembling that block early:

- numerically stable row-wise Softmax using `exp(x - row_max)`;
- RMSNorm using FP32 device accumulation and configurable epsilon;
- interleaved-pair RoPE over contiguous `[batch, sequence, heads, head_dimension]` storage;
- SiLU, add, multiply, and scale;
- an explicitly unfused two-kernel SwiGLU path;
- a causal mask over contiguous `[batch, heads, query, key]` scores.

Softmax exposes naive, block-reduction, and warp-shuffle variants. RMSNorm exposes naive and block-reduction variants. RoPE uses base 10,000 by default and accepts a position offset. Exact input/output aliasing is supported where documented by the tests; the unfused SwiGLU intermediate must be distinct from its inputs and output.

## Correctness

Every operation has an independent CPU reference and CUDA test. The suite covers empty inputs, tiny shapes, warp/block boundaries, irregular widths and tails, in-place paths, and invalid arguments. Operation-specific coverage includes:

- Softmax logits from `-1000` through `1000`, finite outputs, and row sums near one;
- zero and near-zero RMSNorm inputs, alternate epsilon values, and output/weight alias rejection;
- RoPE layout, multiple batches/heads, position offsets, alternate bases, and odd-dimension rejection;
- SiLU inputs from `-100` through `100`, elementwise aliasing, and invalid SwiGLU intermediates;
- causal-mask boundaries, nonzero query offsets, rectangular tails, and exact negative-infinity masking.

The initial report-sized run exposed a RoPE difference at position 2,014 that the short tests did not exercise. The CPU reference deliberately retains double-precision angle/trigonometric evaluation. The CUDA path uses FP32 inverse frequencies and `sincosf`, so frequency rounding is amplified at long positions before range reduction. A 2,048-token regression case now runs in CTest. That case and the report declare `atol=2.5e-4`, `rtol=1e-5`; the observed maximum absolute error is `1.9608438e-4`. Short-position tests retain `1e-5`/`1e-5`.

All 22 Release CTests pass, including the five Stage 6 correctness executables, the unified benchmark smoke run, and the Python JSON contract check. Every report record has zero validation failures.

## Measurement Method

The report was generated from clean code commit `16c59a5` on the RTX 3050 Laptop GPU with seed `2027`, 50 warmups, and 500 measured samples per case. CUDA events enclose only the selected kernel launch or, for unfused SwiGLU, both required launches. Allocation, deterministic input construction, host/device copies, CPU reference work, validation, and serialization are outside the interval.

The measured shapes are:

- Softmax/RMSNorm: 4,096 rows × 1,024 columns;
- RoPE: batch 1, sequence 2,048, 8 heads, head dimension 64, position offset 17;
- elementwise/SwiGLU: 16,777,216 elements;
- causal mask: batch 1, 8 heads, 512 queries × 512 keys.

Effective bandwidth counts logical algorithmic FP32 reads and writes, not measured DRAM transactions. RMSNorm counts input, weight, and output traffic per element; actual cache reuse is hardware-dependent. SwiGLU counts both kernels' logical intermediate traffic. The authoritative per-sample records and compact index are in [`benchmarks/results/stage6/`](../../benchmarks/results/stage6/summary.csv).

## Measured Results

| Operation | Implementation | Median ms | p95 ms | Logical GB/s | Speedup | Maximum absolute error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Softmax | naive | 1.074176 | 1.270784 | 31.237 | 1.000× | 5.402e-8 |
| Softmax | block | 0.222208 | 0.229376 | 151.005 | 4.834× | 5.588e-9 |
| Softmax | warp | 0.200672 | 0.209920 | 167.210 | 5.353× | 5.588e-9 |
| RMSNorm | naive | 0.908288 | 0.921600 | 55.414 | 1.000× | 2.384e-6 |
| RMSNorm | block | 0.197504 | 0.206973 | 254.839 | 4.599× | 4.768e-7 |
| RoPE | interleaved pairs | 0.053248 | 0.056320 | 157.538 | 1.000× | 1.961e-4 |
| SiLU | baseline | 0.738112 | 0.750592 | 181.839 | 1.000× | 4.768e-7 |
| Add | baseline | 1.075200 | 1.093632 | 187.246 | 1.000× | 0 |
| Multiply | baseline | 1.079072 | 1.114163 | 186.574 | 1.000× | 0 |
| Scale | baseline | 0.736256 | 0.751654 | 182.298 | 1.000× | 0 |
| SwiGLU | unfused | 1.828864 | 1.845291 | 183.471 | 1.000× | 1.907e-6 |
| Causal mask | baseline | 0.090112 | 0.097280 | 186.182 | 1.000× | 0 |

## Operation Notes

### Softmax

The naive path assigns one thread to an entire row, leaving little parallelism within each 1,024-element reduction. The block variant cooperatively reduces row maximum and sum in shared memory. The strongest warp variant uses warp shuffles within the reduction and only shared warp summaries between warps. It reduces median latency by `5.353×` and also lowers the numerical error relative to the serial FP32 GPU path.

### RMSNorm

The naive path serially accumulates one row per thread. The block path performs the sum-of-squares reduction cooperatively and then applies the reciprocal root-mean-square and weights in parallel. It improves median latency by `4.599×`. The width-aware tolerance is `atol=1e-5 * max(1, ceil(log2(width)))`, `rtol=1e-5`; the report error remains well below it.

### RoPE

One CUDA thread rotates one adjacent feature pair. The implementation preserves the declared interleaved layout and position offset and reaches `157.538` logical GB/s for the report shape. The long-position precision behavior is now tested and explicitly bounded rather than hidden by changing the CPU reference to mimic device rounding.

### Elementwise and SwiGLU

The one-kernel SiLU, add, multiply, and scale paths cluster around `181.8–187.2` logical GB/s. Unfused SwiGLU measures both SiLU and multiply launches plus the explicit intermediate read/write. It is intentionally retained as the Stage 7 fusion baseline; Stage 6 makes no claim that the two-launch path is optimal.

### Causal Mask

The mask maps each score independently and writes exact negative infinity above the permitted key boundary. The 512 × 512 × 8 report shape measures `0.090112 ms` median with exact agreement to the CPU reference.

## Run Conditions and Limitations

Before the report, `nvidia-smi` showed 80 C, about 5.15 W, and idle 210/405 MHz graphics/memory clocks. Immediately afterward it showed 83 C and about 5.39 W. Clocks were not locked, the laptop was already thermally warm, and these are endpoint snapshots rather than a continuous power/clock trace.

- Results cover FP32, one Windows laptop GPU, and one report shape per operation family.
- No FP16 Transformer primitive is included because Stage 6 requires a reliable FP32 baseline first.
- Softmax and RMSNorm variant conclusions are benchmark-backed but not yet Nsight Compute-backed; focused profiling belongs to Stage 11.
- Logical bandwidth is a comparable accounting metric, not a claim about physical DRAM traffic.
- RoPE is interleaved-pair only; split-half layouts are not implemented.
- Causal masking is a standalone materialized-score operation, not yet fused into Softmax or attention.
- SwiGLU remains deliberately unfused. Fusion and resource-pressure analysis belong to Stage 7.

Stage 7 may compare fused residual + RMSNorm and fused SiLU + multiply against these validated baselines. It will begin only after explicit user authorization.
