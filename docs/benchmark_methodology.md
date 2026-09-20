# WarpForge Benchmark and Validation Methodology

## Purpose

This document is the measurement contract for WarpForge. All implementations in a comparison must use equivalent inputs, timing boundaries, synchronization, build configuration, and statistical treatment. Results that do not satisfy the contract are exploratory data, not project conclusions.

## Correctness before timing

Every custom CUDA implementation must be compared with a trusted reference before its performance result is accepted. The normal sequence is:

```text
Reference -> CUDA implementation -> validation -> benchmark -> profile
```

After every meaningful optimization, validation runs again before the new benchmark result is accepted.

## Numerical comparison

Floating-point values use the combined absolute and relative rule:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

Initial defaults are starting points, not permission to ignore algorithm-specific error growth:

| Datatype | Initial `atol` | Initial `rtol` | Use |
| --- | ---: | ---: | --- |
| FP32 | `1e-5` | `1e-5` | Elementwise operations with limited accumulation |
| FP16 | `1e-2` | `1e-2` | FP16 storage/compute experiments |

Reductions, GEMM, Softmax, RMSNorm, mixed-precision operations, and long accumulation chains must define tolerances based on operation size, accumulation type, reference precision, and expected conditioning. Any override is recorded with the result and applied equally to compared implementations.

Validation should report:

- total element count;
- failing element count;
- maximum absolute error;
- mean absolute error when useful;
- maximum relative error where the reference magnitude makes it meaningful;
- the tolerance and reference used;
- representative mismatch locations on failure.

Inputs use deterministic seeds. Tests include representative random data plus operation-specific edge cases such as zeros, negative values, non-power-of-two dimensions, extreme logits for Softmax, and shapes that exercise partial tiles. NaN and infinity behavior must be defined by the operation rather than silently discarded.

## Timing categories

### Kernel-only latency

Kernel-only timing measures the GPU work under study. Unless the experiment says otherwise, it excludes:

- host and device allocation;
- initialization;
- host-to-device and device-to-host transfers;
- reference computation;
- validation;
- result serialization;
- profiler startup overhead.

CUDA events are recorded in the same stream as the measured work. The stop event is synchronized before elapsed time is read. If a logical implementation launches multiple kernels, all required launches are inside the event interval and the result identifies that it is a multi-kernel measurement.

### End-to-end latency

End-to-end timing measures an explicitly named application path and may include allocation, transfers, dispatch, synchronization, or result retrieval. Its boundary must be written beside the result. Host timing uses an appropriate monotonic clock and includes a final synchronization that guarantees the named GPU work has completed.

Kernel-only and end-to-end results are never presented as directly interchangeable.

## Benchmark procedure

Each benchmark must:

1. Construct deterministic inputs outside the measured region unless input construction is the subject of the experiment.
2. Run the same correctness path used by tests.
3. Perform warmup iterations until one-time initialization is excluded from steady-state measurements.
4. Use separate measured iterations without allocation or transfers unless those costs are in scope.
5. Synchronize at the declared timing boundary.
6. Retain per-sample measurements rather than only an aggregate.
7. Report sample count and summary statistics.
8. Record enough metadata to reproduce the run.

Warmup and measurement counts are configuration values. Early stages should begin with at least 10 warmups and 100 measured iterations for short kernels, then increase counts when timer resolution or variance makes that inadequate. Long-running workloads may use fewer iterations only when the reason is recorded.

## Statistics

Reported summaries should include:

- minimum;
- arithmetic mean;
- median;
- p95;
- sample standard deviation;
- sample count.

The minimum can approximate a low-interference execution, while median and p95 reveal typical behavior and tail variation. No single statistic is treated as the complete result. Outliers are retained unless a documented system event invalidates the run; discarded samples and the reason must be recorded.

WarpForge computes median and p95 with linear interpolation between adjacent sorted samples. Standard deviation is the sample standard deviation (`N - 1` denominator) and is zero for a single sample.

## Derived metrics

Derived metrics must state their formulas and byte/operation counting conventions.

- **Effective bandwidth:** logical bytes read and written divided by elapsed time. The reported byte count must distinguish useful algorithmic traffic from estimated physical DRAM traffic.
- **GEMM throughput:** for dense `M x K` by `K x N` multiplication, use `2 * M * N * K` floating-point operations unless an experiment requires a different convention.
- **Speedup:** baseline latency divided by candidate latency using the same statistic and timing boundary.
- **Library-relative performance:** custom throughput divided by the matched library throughput, reported as a percentage with identical shapes, datatype, math mode, and timing protocol.

## Required metadata

Each persisted benchmark result should contain, where applicable:

- operation and implementation/version;
- datatype and accumulation type;
- tensor dimensions and strides;
- grid, block, tile, and dynamic shared-memory configuration;
- warmup and measured iteration counts;
- timing category and included work;
- GPU name, compute capability, and VRAM;
- driver and CUDA Toolkit versions;
- compiler, build type, and relevant flags;
- benchmark statistics and units;
- correctness tolerance and error summary;
- deterministic seed;
- timestamp and Git commit;
- profiler status and notes.

Stage 2 defines JSON schema version 1 at [`benchmarks/schema/v1.json`](../benchmarks/schema/v1.json). A result contains metadata, benchmark configuration, millisecond statistics, correctness, derived metrics, and every measured sample. Non-finite correctness diagnostics are serialized as `null`; timing samples must always be finite and non-negative. CSV export remains a later extension.

## Stage 2 implementation

- `validate_fp32` applies the combined absolute/relative rule and reports failures, worst index, maximum/mean absolute error, and maximum relative error.
- `measure_cuda_kernel` allocates its CUDA events before warmup, synchronizes after warmup, and records one event-delimited sample per measured iteration.
- Benchmark callbacks contain only the GPU work under study. VectorAdd allocation, initialization, H2D copies, validation D2H copy, and JSON output remain outside the measured interval.
- The first workload uses seed `2027`, 256 threads per block, CUDA launch-error checking, and a CPU reference.

## Comparison rules

A performance comparison is valid only when:

- implementations compute the same mathematical operation;
- shapes, layouts, datatypes, and input values match;
- build type and relevant compiler settings match;
- timing boundaries match;
- warmup and iteration policies match;
- synchronization is equivalent;
- correctness passes under the declared policy;
- power, thermal, and competing-workload conditions are noted when they materially affect interpretation.

Changing several optimization factors at once should be avoided because it weakens the causal conclusion. If unavoidable, the result must be described as a combined change.

## Profiling workflow

### Nsight Systems

Use Nsight Systems to investigate application-level behavior:

- CPU/GPU timeline;
- CUDA API overhead;
- launch cadence and gaps;
- memory copies;
- synchronization;
- stream concurrency and overlap.

### Nsight Compute

Use Nsight Compute to investigate selected kernel behavior:

- achieved and theoretical occupancy;
- SM and memory utilization;
- DRAM and cache behavior;
- memory-access efficiency;
- warp stalls;
- register and shared-memory use;
- instruction behavior;
- roofline position where supported and meaningful.

Profiling runs and timing runs are separate. Instrumentation overhead must not be mixed into headline latency results. Occupancy is evidence about resource use, not a score that must always be maximized.

## Performance journal

Each optimized kernel's report under `docs/performance/` will use this structure:

1. Problem and mathematical operation
2. Baseline implementation
3. Correctness method
4. Baseline benchmark
5. Profiler observations and bottleneck
6. Optimization hypothesis
7. Implementation change
8. New benchmark and correctness result
9. Relevant Nsight metrics
10. Explanation, remaining bottlenecks, and next experiment

## Evidence policy

- Never fabricate benchmark, throughput, occupancy, profiler, compiler, or framework results.
- Label unexecuted expectations as hypotheses.
- Preserve failed or neutral experiments when they teach something relevant.
- Do not claim that WarpForge beats an NVIDIA library unless a controlled, reproducible measurement supports that exact workload and boundary.
- Historical results remain tied to their hardware, software versions, configuration, and commit; they are not universal performance claims.
