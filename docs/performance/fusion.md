# Fusion and Asynchronous Execution

## Scope

Stage 7 asks two separate questions:

1. Does eliminating a materialized intermediate and one kernel launch improve residual + RMSNorm and SiLU + multiply (SwiGLU) on the audited GPU?
2. Does a pinned, two-stream, chunked H2D → SiLU → D2H pipeline create useful overlap compared with the same work in one stream?

The fused kernels remain narrow FP32 operations over explicit pointers, dimensions, and streams. The pipeline remains benchmark/test orchestration rather than becoming a premature runtime abstraction.

## Correctness and Ordering

`residual_rmsnorm_cpu` uses double precision for its sum-of-squares reference. The fused CUDA kernel computes the residual sum inside the row reduction and again during normalized output, avoiding a global intermediate. It supports exact output aliasing with either data input but rejects output aliasing with the shared weight vector.

`swiglu_fused_cpu` is the trusted `SiLU(gate) * up` reference. The fused CUDA kernel computes that expression in one elementwise pass and supports exact output aliasing with either input.

Focused tests cover empty, tiny, irregular, warp/block-boundary, and large cases; extreme activation values; input/output aliasing; invalid epsilon, weight alias, and block size; and both fused and separate CUDA paths against the same CPU result.

The pipeline test uses pinned host buffers, two nonblocking streams, two device buffers, and one completion event per stream. It enqueues H2D, in-place SiLU, and D2H per chunk, records each completion event after the stream's last chunk, and synchronizes those events. No device-wide synchronization is used for ordering.

All 25 Release CTests pass. All six report records have zero validation failures.

## Logical Memory Traffic

Logical reads/writes count the tensors required by the mathematical composition, not physical cache/DRAM transactions or repeated implementation loads inside a reduction.

| Operation | Path | Reads/element | Writes/element | Total accesses | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| residual + RMSNorm | separate | 4 | 2 | 6 | baseline |
| residual + RMSNorm | fused | 3 | 1 | 4 | 33.3% |
| SwiGLU | separate | 3 | 2 | 5 | baseline |
| SwiGLU | fused | 2 | 1 | 3 | 40.0% |

Residual fusion removes the intermediate residual-sum write and later RMSNorm read. SwiGLU fusion removes the SiLU intermediate write/read. Both also remove one kernel launch.

## Measurement Method

The report was generated from clean code commit `b21cdf0` on the RTX 3050 Laptop GPU with seed `2027`, 50 warmups, and 500 samples per case.

- Residual + RMSNorm uses 4,096 rows × 1,024 columns.
- SwiGLU uses 16,777,216 FP32 elements.
- The pipeline uses 16,777,216 elements split into eight pinned chunks.
- CUDA events measure only the one- or two-kernel fusion cases.
- A host steady clock measures the pipeline from enqueue start through explicit completion-event synchronization, including H2D, kernel, D2H, launch/API overhead, and event completion.
- Allocation, deterministic initialization, CPU references, and result serialization remain outside every interval.

Authoritative per-sample JSON and the compact summary are in [`benchmarks/results/stage7/`](../../benchmarks/results/stage7/summary.csv).

## Measured Results

| Operation | Path | Timing scope | Median ms | p95 ms | Speedup | Maximum absolute error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| residual + RMSNorm | separate | kernel-only | 0.468992 | 0.482304 | 1.000× | 7.153e-7 |
| residual + RMSNorm | fused | kernel-only | 0.317440 | 0.328643 | 1.477× | 7.153e-7 |
| SwiGLU | separate | kernel-only | 1.808384 | 1.833051 | 1.000× | 1.907e-6 |
| SwiGLU | fused | kernel-only | 1.108992 | 1.134470 | 1.631× | 1.907e-6 |
| pinned SiLU pipeline | single stream | end-to-end | 14.078150 | 15.678615 | 1.000× | 4.768e-7 |
| pinned SiLU pipeline | two streams | end-to-end | 14.680550 | 16.675055 | 0.959× | 4.768e-7 |

Residual + RMSNorm fusion lowers median latency by 32.3%; SwiGLU fusion lowers it by 38.7%. Both preserve exactly the same recorded maximum error as their separate baselines, so the measured gains justify retaining the fused implementations.

The two-stream pipeline is 4.28% slower by median and has a higher p95. Correct asynchronous code is not automatically concurrent or faster; this experiment is retained as a measured negative result.

## Nsight Compute Evidence

Nsight Compute 2024.3.2 captured each separate kernel and both fused kernels. Instrumented durations are explanatory and are not substituted for the report measurements.

| Path/kernel | Instrumented duration | Memory throughput | DRAM peak | Registers/thread | Shared memory | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| residual add | 272.160 us | 184.858 GB/s | 94.60% | 16 | 0 B | 84.18% |
| RMSNorm block | 217.696 us | 153.649 GB/s | 78.61% | 18 | 1,024 B | 93.97% |
| fused residual + RMSNorm | 325.600 us | 163.412 GB/s | 83.61% | 18 | 1,024 B | 86.74% |
| SiLU | 756.704 us | 177.204 GB/s | 90.65% | 16 | 0 B | 81.14% |
| multiply | 1,079.232 us | 186.496 GB/s | 95.40% | 16 | 0 B | 83.62% |
| fused SwiGLU | 1,121.248 us | 179.664 GB/s | 91.91% | 16 | 0 B | 85.99% |

All six kernels have 100% theoretical occupancy. Residual fusion matches the RMSNorm kernel's 18 registers and shared-memory allocation; SwiGLU fusion matches its separate kernels' 16 registers and zero shared memory. Fusion therefore does not introduce a resource-pressure penalty at these shapes.

The fused kernels reach 83.61% and 91.91% of peak DRAM throughput while using little FP32 compute, supporting a memory-bound classification. Removing logical intermediate traffic and a launch addresses that measured bottleneck directly.

## Nsight Systems Evidence

Nsight Systems 2025.6.3 captured both pipeline variants. The two-stream portions use streams 15 and 16, but none of the 48 H2D, SiLU, and D2H intervals overlap; the smallest gap between consecutive GPU operations is 1,728 ns. The device reports one asynchronous copy engine.

The timeline remains serialized H2D → kernel → D2H even as chunks alternate streams. This explains why two streams do not improve latency and agrees with the uninstrumented 4.28% regression. Stream count alone is not evidence of overlap.

Commands, compact metrics, and a representative timeline excerpt are in [`benchmarks/profiles/stage7/`](../../benchmarks/profiles/stage7/README.md). Raw `.ncu-rep`, `.nsys-rep`, and SQLite files remain ignored.

## Decisions and Limitations

- Retain both fused kernels: each is correct, materially faster, and does not increase the measured register/shared-memory footprint.
- Retain the two-stream implementation as reproducible evidence, but prefer the single-stream pipeline on the audited machine.
- Results cover FP32, one Windows laptop GPU, one report shape per operation, and unlocked clocks.
- Endpoint temperature rose from 78 C to 81 C during the report run.
- Logical access counts describe the operation graph; Nsight metrics describe measured physical behavior. They are not interchangeable.
- No fusion spans GEMM, attention Softmax, causal masking, or whole Transformer sublayers.
- General stream/event ownership and reusable buffers remain Stage 8 runtime work.

Stage 8 will introduce the minimal move-only GPU runtime and migrate these explicit-stream kernels without hiding their costs. It will begin only after explicit user authorization.
