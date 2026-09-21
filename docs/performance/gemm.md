# GEMM Optimization Ladder

## Problem and Semantics

Stage 5 computes row-major `C[M,N] = A[M,K] × B[K,N]` with no transpose, `alpha = 1`, and `beta = 0`. `GemmProblem`, `GemmVariant`, `GemmBackend`, and `GemmDispatch` make dimensions and backend selection explicit. All output accumulation is FP32; input storage is either FP32 or FP16.

The custom ladder contains:

1. one-thread-per-output naive FP32;
2. 16 × 16 shared-memory tiled FP32;
3. 32 × 32 coalesced tile loading with four output rows per thread;
4. 64 × 64 register-blocked FP32 with a 4 × 4 per-thread output tile;
5. shared-memory tiled FP16 input with FP32 accumulation;
6. 16 × 16 × 16 WMMA FP16 input with FP32 accumulation.

Matched row-major cuBLAS FP32 and FP16-input/FP32-accumulation paths are the library baselines. The row-major mapping swaps operands and output dimensions when calling the column-major cuBLAS interface.

## Correctness

The small-shape tests compare every applicable implementation with a CPU reference that accumulates products in `double` before conversion to FP32. Shapes include 1 × 1 × 1, irregular 3 × 5 × 7 and 65 × 47 × 33 problems, and 15/16/17/31/32/33 tile boundaries. FP16 references use the exact values after FP16 quantization. WMMA accepts only dimensions divisible by 16 and rejects incompatible shapes.

Report size 256 validates both cuBLAS baselines against CPU. Larger custom outputs are compared with the matched cuBLAS result. FP32 uses `atol = 1e-5 * max(1, ceil(log2(K)))`, `rtol = 1e-4`; FP16-input paths use `atol = 1e-3 * max(1, ceil(log2(K)))`, `rtol = 1e-2`.

All 15 Release CTests pass. Across the report and 4096 feasibility run, the largest FP32 custom absolute error is `8.20159912109375e-05`; the largest FP16 custom absolute error is `0.0002880096435546875`. Every persisted record passes its declared tolerance.

## Measurement Method

The main report was collected from clean implementation commit `f88127e` with seed `2027`, 50 warmups, and 100 measured samples for each of 32 implementation/size pairs. The sample count is lower than the normal 500-sample report target because the complete 2,048 ladder contains several long custom kernels and sustained execution materially heats this laptop GPU.

Sizes 256, 512, 1,024, and 2,048 are in [`benchmarks/results/stage5/`](../../benchmarks/results/stage5/summary.csv). After allocation and runtime checks passed, 4,096 was attempted separately with 5 warmups and 20 samples in [`benchmarks/results/stage5_4096/`](../../benchmarks/results/stage5_4096/summary.csv).

CUDA events measure only the selected GEMM call. Allocation, input generation, H2D copies, reference generation, D2H validation, and serialization are excluded. Throughput is `2*M*N*K / seconds`. Custom percentage of cuBLAS is matched by shape and input datatype.

## FP32 Results

Each implementation cell is `median milliseconds / GFLOP/s`.

| Size | Naive | Tiled | Coalesced | Register blocked | cuBLAS | Register / cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.099328 / 337.8 | 0.076800 / 436.9 | 0.068608 / 489.1 | 0.046080 / 728.2 | 0.026624 / 1,260.3 | 57.78% |
| 512 | 0.558080 / 481.0 | 0.417520 / 642.9 | 0.375808 / 714.3 | 0.142336 / 1,885.9 | 0.104448 / 2,570.0 | 73.38% |
| 1,024 | 5.115392 / 419.8 | 3.741184 / 574.0 | 2.965504 / 724.2 | 1.094704 / 1,961.7 | 0.527360 / 4,072.1 | 48.17% |
| 2,048 | 44.185488 / 388.8 | 37.277695 / 460.9 | 33.396641 / 514.4 | 12.292096 / 1,397.6 | 4.140032 / 4,149.7 | 33.68% |
| 4,096* | 412.803482 / 332.9 | 357.000702 / 385.0 | 319.887360 / 429.6 | 99.736576 / 1,378.0 | 34.175440 / 4,021.6 | 34.27% |

`*` The 4,096 row is the shorter feasibility run.

Every optimization step improves the measured median at every size. At 1,024, register blocking is `4.673×` faster than naive FP32. It reaches the largest observed library-relative fraction at size 512, `73.38%`, but cuBLAS remains faster for every measured size.

## FP16 and Tensor Core Results

| Size | Tiled FP16 median / GFLOP/s | WMMA median / GFLOP/s | cuBLAS FP16 median / GFLOP/s | WMMA / cuBLAS |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.080896 / 414.8 | 0.022496 / 1,491.6 | 0.016384 / 2,048.0 | 72.83% |
| 512 | 0.418160 / 641.9 | 0.077824 / 3,449.3 | 0.030432 / 8,820.8 | 39.10% |
| 1,024 | 3.701248 / 580.2 | 0.587264 / 3,656.8 | 0.216512 / 9,918.5 | 36.87% |
| 2,048 | 44.566528 / 385.5 | 7.652864 / 2,244.9 | 1.591184 / 10,796.9 | 20.79% |
| 4,096* | 357.781906 / 384.1 | 68.316158 / 2,011.8 | 14.724608 / 9,334.0 | 21.55% |

WMMA is substantially faster than the scalar/tiled FP16 implementation and proves that the Tensor Core path is active. It is not close to cuBLAS at large sizes because each warp independently reloads matrix tiles and the implementation does not yet stage or pipeline data for reuse across multiple output tiles.

## Nsight Compute Evidence

Nsight Compute 2024.3.2 profiled one 1,024 launch per selected kernel. Instrumented durations are explanatory only.

| Metric | Naive FP32 | Register-blocked FP32 | WMMA FP16/FP32 |
| --- | ---: | ---: | ---: |
| Instrumented duration | 7,415.328 µs | 1,501.504 µs | 914.816 µs |
| Grid / block threads | 4,096 / 256 | 256 / 256 | 4,096 / 32 |
| DRAM throughput | 21.56% | 8.58% | 22.55% |
| SM throughput | 96.88% | 57.35% | 35.00% |
| Tensor-pipe active | 0% | 0% | 11.58% |
| Achieved occupancy | 99.05% | 66.35% | 33.21% |
| Registers per thread | 34 | 56 | 42 |
| Static shared memory | 0 B | 8,512 B | 0 B |
| Warp cycles per issued instruction | 25.46 | 13.85 | 36.29 |
| Long-scoreboard stall | 12.93 | 4.20 | 30.53 |

### Baseline to optimized FP32

The naive kernel repeatedly loads one A value per output thread while only B's adjacent-column access is naturally coalesced. Nsight Compute reports 18 useful bytes per 32-byte load sector, long-scoreboard and LG-throttle stalls totaling 20.97 cycles per issued instruction, and only 6% of FP32 roofline peak.

Shared tiling, coarsened coalesced loading, and finally a 4 × 4 register tile increase data reuse and reduce the grid from 4,096 to 256 blocks. Register blocking reduces the 1,024 report median from `5.115392 ms` to `1.094704 ms` while preserving correctness.

The optimized kernel is not finished: 56 registers limit theoretical occupancy to 66.67%, and Nsight Compute reports uncoalesced stores, 393,216 excessive global sectors, shared-memory bank conflicts, and 3,670,016 excessive shared wavefronts. These measurements explain why cuBLAS remains `2.076×` faster at 1,024 and identify thread-to-output mapping and shared layout as the next custom FP32 work.

### Tensor Core experiment

The WMMA kernel executes HMMA/Tensor Core work, but tensor-pipe activity reaches only 11.58%. One warp computes one output tile, giving 33.33% theoretical occupancy; long-scoreboard stalls consume 30.53 of 36.29 cycles per issued instruction, and only 16 of 32 bytes per global-load sector are useful. The experiment is memory-latency/data-movement bound, not Tensor Core compute-bound.

Commands and compact metrics are in [`benchmarks/profiles/stage5/`](../../benchmarks/profiles/stage5/README.md). Raw `.ncu-rep` files remain ignored.

## Run Conditions

Before the main report, `nvidia-smi` reported 75 C, P5, 6.95 W, 29% utilization, 532 MHz SM clock, and 810 MHz memory clock. Immediately afterward it reported 83 C. The 4,096 attempt began at 79 C and ended at 84 C. These are endpoint snapshots, clocks were not locked, and unrelated background utilization was present. The declining large-size custom throughput is therefore not presented as architecture-only scaling.

## Conclusions and Limitations

- Correct tiling and register reuse produce a clear optimization ladder, but cuBLAS wins every matched comparison.
- Register blocking is the strongest custom FP32 path; its remaining store and shared-bank inefficiencies are profiler-visible.
- WMMA uses Tensor Cores correctly but lacks the cooperative data staging and occupancy needed to approach cuBLAS.
- Results cover square row-major matrices on one thermally constrained Windows laptop GPU.
- The public API currently implements `alpha = 1`, `beta = 0`, no transpose, contiguous storage, and FP32 output only.
- No split-K, asynchronous tile copy, double buffering, batched GEMM, or autotuning is implemented.

Stage 6 will build Transformer-oriented CUDA kernels on these validation and measurement contracts. It will not begin until explicitly requested.
