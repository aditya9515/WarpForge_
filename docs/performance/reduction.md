# Parallel Reduction Engine

## Problem

Stage 4 implements FP32 sum and maximum reductions as an explicit optimization ladder. A reduction repeatedly maps an arbitrary-length input to smaller arrays of block results until one value remains. WarpForge uses this deterministic multi-pass structure instead of making a nondeterministic global atomic accumulation the primary path.

The public dispatcher accepts an operation, implementation variant, input length, caller-owned ping-pong workspaces, block size, output pointer, and CUDA stream. Every pass is included in the kernel-only timing interval.

## Implementations

| Variant | First-pass input per block | Main technique |
| --- | ---: | --- |
| `naive_interleaved` | 256 elements | Interleaved shared-memory indexing and a barrier at every tree level |
| `shared_memory` | 256 elements | Sequential shared-memory tree with reduced indexing overhead |
| `reduced_divergence` | 512 elements | Two inputs per thread before the sequential shared-memory tree |
| `unrolled` | 512 elements | Compile-time unrolled shared-memory tree and warp tail |
| `warp_shuffle` | 512 elements | Register/shuffle warp reductions plus one shared value per warp |

All variants support power-of-two block sizes from 32 through 1024. The report uses 256 threads. Neutral steps remain in the ladder: the sequential shared-memory variant is not removed merely because it did not improve this measured workload.

## Correctness

The sum reference accumulates FP32 inputs in CPU `double`. GPU sum validation uses:

```text
atol = 1e-5 * (1 + ceil(log2(N)))
rtol = 2e-5
abs(gpu - cpu) <= atol + rtol * abs(cpu)
```

For the report size, `N = 16,777,216`, so `atol = 0.00025`. The CPU sum was `3190.3831573724747`. GPU absolute error was between `0.00010073184967041016` and `0.00014340877532958984`, depending on the tree order. The maximum reference and every GPU variant returned exactly `7.25`.

Tests cover lengths 1, 2, 17, 31, 32, 33, 127, 255, 256, 257, 511, 512, 513, 1,003, 4,097, and 1,048,576; all five variants; both operations; block sizes 32, 64, 128, 256, and 512; negative-only maximum inputs; invalid configurations; undersized workspaces; and CPU double-precision accumulation. All 12 Release CTests pass.

## Measurement Method

Report data was collected on 2026-09-20 from clean implementation commit `26ac9d9`, using deterministic seed `2027`, 50 warmups, and 500 measured samples per case. The input contains 16,777,216 FP32 values and each launch uses 256 threads.

CUDA events delimit every pass needed to produce the scalar. Allocation, deterministic input construction, H2D setup, scalar D2H validation, reference work, and result serialization are outside the interval. Effective input bandwidth divides the original input's 67,108,864 logical bytes by median latency; it deliberately does not claim physical DRAM traffic and does not add intermediate-pass traffic.

## Sum Results

| Variant | Passes | Median | P95 | Logical input GB/s | Speedup | Absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive interleaved | 4 | 0.990176 ms | 1.321893 ms | 67.775 | 1.000× | 0.000100732 |
| Shared memory | 4 | 0.991232 ms | 1.013861 ms | 67.702 | 0.999× | 0.000143409 |
| Reduced divergence | 3 | 0.518144 ms | 0.546634 ms | 129.518 | 1.911× | 0.000143409 |
| Unrolled | 3 | 0.367616 ms | 0.387123 ms | 182.552 | 2.694× | 0.000143409 |
| Warp shuffle | 3 | 0.365536 ms | 0.381952 ms | 183.590 | 2.709× | 0.000100732 |

## Maximum Results

| Variant | Passes | Median | P95 | Logical input GB/s | Speedup | Absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive interleaved | 4 | 0.990208 ms | 1.017021 ms | 67.772 | 1.000× | 0 |
| Shared memory | 4 | 0.992256 ms | 1.019904 ms | 67.633 | 0.998× | 0 |
| Reduced divergence | 3 | 0.518144 ms | 0.541795 ms | 129.518 | 1.911× | 0 |
| Unrolled | 3 | 0.367616 ms | 0.392994 ms | 182.552 | 2.694× | 0 |
| Warp shuffle | 3 | 0.364544 ms | 0.380581 ms | 184.090 | 2.716× | 0 |

Two inputs per thread nearly halves the median by halving the first-pass grid and removing a whole reduction pass at this size. Compile-time unrolling supplies the next large improvement. Warp shuffle is only slightly faster than the already-unrolled variant, but it sharply reduces shared-memory use and barrier work.

## Nsight Compute Evidence

Nsight Compute 2024.3.2 profiled the first sum pass separately from report timing. Performance-counter access required an elevated profiler process on this Windows machine; no driver setting was changed. Instrumented durations are evidence only and are not substituted for the 500-sample report timings.

| Metric | Naive interleaved | Warp shuffle |
| --- | ---: | ---: |
| First-pass grid | 65,536 × 256 | 32,768 × 256 |
| Instrumented duration | 1,572.672 µs | 401.568 µs |
| DRAM bandwidth | 43.264 GB/s | 169.265 GB/s |
| DRAM throughput | 22.13% | 86.60% |
| SM throughput | 88.41% | 65.84% |
| Achieved occupancy | 89.92% | 74.80% |
| Registers per thread | 16 | 16 |
| Dynamic shared memory per block | 1,024 B | 32 B |
| Warp cycles per issued instruction | 31.53 | 28.27 |
| Barrier stall, cycles/instruction | 8.65 | 2.56 |
| MIO throttle stall, cycles/instruction | 5.75 | 0.56 |
| Long-scoreboard stall, cycles/instruction | 5.50 | 14.75 |
| Excessive shared-memory wavefronts | 6,356,992 | 0 |
| FP32 adds per cycle | 8.59 | 111.02 |

The profiler reports no divergent branches for the baseline because the compiler predicates its reduction control flow. The cost is still visible: only a shrinking subset of lanes performs useful work, and the report shows high barrier/MIO stalls plus 6.36 million excessive shared-memory wavefronts. The baseline is therefore constrained by its shared-memory access pattern, synchronization, and inactive-lane work rather than a branch-divergence counter.

The optimized first pass loads two adjacent values per thread, uses shuffle instructions inside each warp, stores only one partial per warp, and performs one final warp reduction. That eliminates the shared-memory wavefront warning, reduces dynamic shared memory by 32×, and increases measured DRAM bandwidth by 3.91×. Its lower achieved occupancy did not prevent the speedup; occupancy is not an optimization objective by itself. At 86.60% DRAM throughput with long-scoreboard stalls now dominant, this first pass is memory-bound on the audited workload.

This is the evidence chain:

```text
interleaved shared tree
  -> excessive shared wavefronts + barrier/MIO stalls
  -> reduce blocks/passes and keep the warp tail in registers
  -> zero excessive shared wavefronts + 169.265 GB/s profiled DRAM bandwidth
  -> correctness preserved + 2.709x/2.716x full-reduction median speedup
```

Profiler commands and the exported metric summary are in [`benchmarks/profiles/stage4/`](../../benchmarks/profiles/stage4/README.md). Raw `.ncu-rep` files remain ignored under `build/profiles/`.

## Run Conditions

Before the report suite, `nvidia-smi` reported 74 C, P8, 0% utilization, 4.59 W, 210 MHz SM clock, and 405 MHz memory clock. Immediately afterward it reported 86 C, P0, 97% utilization, 75.68 W, 1,987 MHz SM clock, and 6,121 MHz memory clock. These endpoint snapshots are not continuous telemetry, and clocks were not locked.

The complete results are in [`benchmarks/results/stage4/summary.csv`](../../benchmarks/results/stage4/summary.csv), with all 500 samples and metadata preserved in the adjacent JSON files.

## Conclusions

- Multi-pass ping-pong reduction handles arbitrary lengths without using global atomics as the primary accumulation path.
- Sequential shared-memory indexing alone was neutral; it did not address the dominant cost at this scale.
- Loading two elements per thread cut grid and pass overhead and produced the first major speedup.
- Unrolling removed loop/control overhead, while shuffle reduction nearly eliminated shared-memory coordination.
- The strongest measured variants are warp shuffle: `2.709×` for sum and `2.716×` for maximum versus naive interleaved.
- The optimized first pass moved the limiting behavior from shared-memory/synchronization overhead toward DRAM latency and bandwidth.

## Limitations and Next Experiment

- Results cover one Windows laptop GPU, one driver/toolkit combination, one report size, and one block size.
- The report run heated the GPU to an 86 C endpoint; unlocked clocks and thermal behavior can influence absolute timing.
- FP32 tree order is deterministic for a given launch but not numerically identical to serial double accumulation; no compensated summation is implemented.
- Logical input bandwidth excludes partial-pass traffic, while Nsight Compute DRAM bandwidth is a separate profiler measurement.
- Nsight Compute replay and instrumentation change execution time, so profiler duration is not a headline benchmark.

Stage 5 will build a validated GEMM optimization ladder and compare custom implementations with cuBLAS. It will not begin until explicitly requested.
