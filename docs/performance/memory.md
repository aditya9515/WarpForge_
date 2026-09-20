# CUDA Execution and Memory Engineering

## Problem

Stage 3 measures how launch shape, access order, host-memory type, copy API, transpose tiling, and stream count affect execution on the audited RTX 3050 Laptop GPU. The goal is controlled evidence, including neutral results, rather than a claim that every additional mechanism is faster.

## Implemented Workloads

- SAXPY: `y[i] = alpha * x[i] + y[i]`
- Custom device-to-device logical FP32 copy
- Strided gather: `output[i] = input[i * stride]`
- Row-major out-of-place rectangular matrix transpose
- H2D + D2H transfer round trips
- Chunked H2D(x, y) → SAXPY → D2H pipeline

All CUDA launches perform an immediate launch-error check. Empty inputs are accepted by the kernel entry points, while benchmark runs require non-empty dimensions.

## Correctness

CPU references validate SAXPY, strided copy, and transpose. Copy and transpose use exact FP32 comparison; SAXPY uses `atol=1e-5`, `rtol=1e-5`. Tests cover:

- empty, tiny, warp-boundary, block-boundary, non-divisible, and large 1D sizes;
- block sizes 32, 64, 128, 256, and 512;
- strides 1, 2, 4, 8, 16, and 32;
- empty, square, rectangular, tile-boundary, and irregular transposes;
- odd-sized, five-chunk work distributed across two non-blocking streams;
- invalid block, stride, and transpose launch configurations.

All nine Release CTests pass. Every one of the 24 report records has zero validation failures.

## Measurement Method

Report data was collected on 2026-09-20 from clean code commit `3c30bc7`, using deterministic seed `2027`, 50 warmups, and 500 measured samples per case.

Kernel results use CUDA events. Allocation, initialization, H2D setup, D2H validation, and serialization are outside the event interval. Transfer and pipeline results use host steady-clock timing around the complete operation and its required stream synchronization, so those records are explicitly labeled `end-to-end`.

The 1D block and transfer experiments use 16,777,216 FP32 elements. Stride experiments keep 1,048,576 logical outputs fixed and expand the source span with stride. Transpose uses a rectangular 2,048 × 1,536 matrix. The pipeline divides the 16,777,216 elements into eight chunks and transfers two inputs, executes SAXPY, and returns one output.

Logical bandwidth counts algorithmic bytes, not profiler-measured physical DRAM transactions.

## Block-Size Sweep

| Threads | SAXPY median | SAXPY GB/s | Copy median | Copy GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.321984 ms | 152.291 | 1.346304 ms | 99.693 |
| 64 | 1.080320 ms | 186.358 | 0.756736 ms | 177.364 |
| 128 | 1.074864 ms | 187.304 | 0.731136 ms | 183.574 |
| 256 | 1.074176 ms | 187.424 | 0.732160 ms | 183.317 |
| 512 | 1.074176 ms | 187.424 | 0.740064 ms | 181.360 |

SAXPY plateaus at 128–512 threads, while custom copy is strongest at 128 threads. A single 32-thread warp per block leaves both materially slower on this workload. More threads are not monotonically better: copy at 512 threads is slightly slower than at 128 or 256.

## Coalesced Versus Strided Access

| Stride | Median | Logical GB/s | Relative latency vs stride 1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.051200 ms | 163.840 | 1.00× |
| 2 | 0.073408 ms | 114.274 | 1.43× |
| 4 | 0.118496 ms | 70.792 | 2.31× |
| 8 | 0.206768 ms | 40.570 | 4.04× |
| 16 | 0.206624 ms | 40.598 | 4.04× |
| 32 | 0.206688 ms | 40.586 | 4.04× |

Stride 1 gives adjacent threads adjacent FP32 addresses. Increasing stride spreads a warp's reads across more memory sectors, so useful logical bytes per transaction fall. The measured penalty saturates near stride 8 for this access pattern and data size; Stage 3 does not claim that this threshold is universal.

## Matrix Transpose

| Variant | Median | P95 | Logical GB/s | Correct |
| --- | ---: | ---: | ---: | --- |
| Naive | 0.424960 ms | 0.439235 ms | 59.219 | PASS |
| Tiled 32×32, 32×8 threads | 0.141312 ms | 0.148262 ms | 178.087 | PASS |

The tiled transpose is `3.007×` faster by median. Its padded 32×33 static shared-memory tile turns the transpose into coalesced global reads and writes while avoiding shared-memory bank conflicts. It uses 4,224 bytes of static shared memory per block.

## Host Transfers and Pipelines

| Experiment | Median | P95 | Logical GB/s | Change from matched baseline |
| --- | ---: | ---: | ---: | ---: |
| Pageable synchronous round trip | 17.620300 ms | 18.788755 ms | 7.617 | baseline |
| Pinned synchronous round trip | 10.684950 ms | 11.053660 ms | 12.561 | 1.649× faster |
| Pageable asynchronous round trip | 17.618950 ms | 18.894030 ms | 7.618 | 0.008% faster |
| Pinned asynchronous round trip | 10.657150 ms | 10.861205 ms | 12.594 | 0.260% faster |
| Pinned single-stream, eight chunks | 17.688900 ms | 18.116805 ms | 11.382 | baseline |
| Pinned two-stream, eight chunks | 17.707150 ms | 17.996040 ms | 11.370 | 0.103% slower |

Pinned host memory materially improves the round-trip transfer result. Merely switching from synchronous to asynchronous APIs does not materially improve a serial round trip that immediately synchronizes. The two-stream chunked pipeline is neutral within this run and is slightly slower by median.

## Nsight Systems Timeline

Nsight Systems 2025.6.3 captured a separate representative profile with one measured single-stream and one measured two-stream pipeline. The trace contains 64 H2D copies, 32 SAXPY launches, and 32 D2H copies across validation and measured executions.

| Operation | Count | Total GPU time | Median |
| --- | ---: | ---: | ---: |
| H2D | 64 | 69.379256 ms | 0.987726 ms |
| SAXPY | 32 | 4.305431 ms | 0.134992 ms |
| D2H | 32 | 34.898313 ms | 0.993742 ms |

The two-stream portion uses streams 15 and 16, but its GPU operations are serialized. Stream 16's first H2D begins 3.616 µs after the preceding stream 15 D2H completes. The trace therefore shows no H2D/kernel/D2H overlap for this audited pipeline, consistent with its lack of end-to-end speedup.

Profiler commands and exported excerpts are in [`benchmarks/profiles/stage3/`](../../benchmarks/profiles/stage3/README.md). Raw `.nsys-rep` and SQLite files remain ignored under `build/profiles/`.

## Run Conditions

Before the report run, `nvidia-smi` reported 64 C, P8, 0% utilization, 3.79 W, 210 MHz SM clock, and 405 MHz memory clock. Immediately afterward it reported 80 C, P0, 97% utilization, 29.20 W, 2017 MHz SM clock, and 6121 MHz memory clock. These are endpoint snapshots, not continuous telemetry or locked-clock conditions.

The complete results are in [`benchmarks/results/stage3/summary.csv`](../../benchmarks/results/stage3/summary.csv), with all 500 samples and metadata preserved in the adjacent JSON files.

## Conclusions

- One-warp blocks underutilize these large SAXPY and copy workloads; 128–256 threads are sound local defaults.
- Coalescing matters sharply: stride 8 uses roughly one quarter of stride 1's logical bandwidth.
- Shared-memory tiling makes both sides of transpose coalesced and produces a measured 3.007× improvement.
- Pinned memory improves large host/device transfers on this machine.
- Asynchronous APIs enable overlap but do not create it automatically.
- Two streams did not overlap or improve this specific pipeline, so WarpForge retains the experiment as a measured neutral result rather than an optimization claim.

## Limitations and Next Experiment

- Results cover one Windows laptop GPU, one driver/toolkit combination, and one report size per experiment family.
- GPU clocks were not locked, and the report run raised the endpoint temperature from 64 C to 80 C.
- Logical bandwidth is not equivalent to physical DRAM bandwidth.
- Nsight Systems instrumentation changes absolute timings; profile values explain sequencing only and are not compared with report-run latency.
- Nsight Compute metrics are deferred until Stage 4, where reduction variants provide a focused kernel-level optimization question.

Stage 4 will apply progressively stronger block and warp reduction strategies to FP32 sum and maximum, validate arbitrary lengths, and use Nsight Compute to connect code changes to divergence, shared memory, and warp-level behavior.
