# VectorAdd Validation Baseline

## Problem

Stage 2 needs a simple workload that can verify the complete reference → CUDA → validation → benchmark → export path without making kernel optimization the feature. VectorAdd exposes indexing, launch boundaries, asynchronous execution, and memory traffic while keeping the mathematical reference unambiguous.

## Mathematical Operation

For `i` in `[0, N)`:

```text
C[i] = A[i] + B[i]
```

The logical traffic model counts two FP32 reads and one FP32 write: 12 bytes per element.

## Baseline Implementation

- CPU: one serial loop over FP32 arrays.
- CUDA: one thread per element with a bounds check.
- Launch: 256 threads per block and `ceil(N / 256)` blocks.
- Input seed: `2027`, uniformly distributed over `[-1, 1]`.

## Correctness Method

The CUDA output is copied to the host and compared with the CPU result using:

```text
abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)
```

The report records failure count, maximum and mean absolute error, maximum relative error, worst index, and the values at that index.

## Baseline Benchmark

Measured on 2026-09-20 from code commit `04ff8c9`:

| Property | Value |
| --- | ---: |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| CUDA runtime | 12.6 |
| Compiler | MSVC 19.44.35228.0 / nvcc 12.6.85 |
| Elements | 1,048,576 |
| Logical bytes | 12,582,912 |
| Grid | 4,096 blocks |
| Block | 256 threads |
| Warmups | 50 |
| Samples | 500 |
| Minimum | 0.071680 ms |
| Mean | 0.073380 ms |
| Median | 0.072704 ms |
| P95 | 0.074405 ms |
| Sample standard deviation | 0.003197 ms |
| Effective bandwidth from median | 173.070 GB/s |
| Maximum absolute error | 0 |

The complete samples and metadata are stored in [`benchmarks/results/stage2/vector_add_1048576.json`](../../benchmarks/results/stage2/vector_add_1048576.json).

Before the run, `nvidia-smi` reported 68 C, P8, 0% utilization, 3.96 W, 210 MHz SM clock, and 405 MHz memory clock. The immediate post-run snapshot reported 68 C, P0, 22% utilization, 4.29 W, 1500 MHz SM clock, and 6121 MHz memory clock. These snapshots describe the run surroundings but are not continuous telemetry.

## Profiler Observations

No Nsight profile was collected in Stage 2. Profiling this validation workload before defining the Stage 3 memory experiments would add data without answering a specific optimization question.

## Bottleneck

VectorAdd has low arithmetic intensity and is expected to be memory-oriented, but Stage 2 does not classify the measured kernel as memory-bound without profiler evidence. Stage 3 will test that hypothesis through block-size, access-pattern, and bandwidth experiments.

## Optimization Hypothesis

No optimization hypothesis is tested in Stage 2. The controlled baseline establishes the measurement path that later experiments will reuse.

## Implementation Change

Not applicable. This document records the first correct CUDA baseline.

## New Benchmark

Not applicable until a controlled Stage 3 variant is compared with the same timing methodology.

## Correctness After Optimization

No optimization was applied. The baseline produced zero failures and zero maximum absolute error for the measured input.

## Nsight Metrics

Not collected in Stage 2.

## Explanation

Allocation, input initialization, H2D copies, the validation D2H copy, CPU computation, and JSON serialization are outside the event interval. Each sample records a start event, launches VectorAdd in the same stream, records a stop event, and synchronizes the stop event before reading elapsed time.

One retained sample was materially higher than the central cluster, which raises the mean and standard deviation relative to the median. The sample was not discarded because no invalidating system event was observed. Reporting the complete distribution preserves that evidence.

## Remaining Bottlenecks

- No block-size sweep has been performed.
- Logical bandwidth has not been compared with profiler-reported DRAM traffic.
- Host/device transfer behavior is excluded from this kernel-only result.
- No coalesced-versus-strided comparison exists yet.

## Next Experiment

Stage 3 will reuse this workload for controlled block-size sweeps, then compare coalesced and strided access, pageable and pinned transfers, synchronous and asynchronous paths, matrix transpose variants, and stream overlap.
