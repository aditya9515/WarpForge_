# WarpForge

WarpForge is a staged CUDA/C++ performance-engineering project that will grow from measured GPU fundamentals into a small, validated LLaMA-style inference runtime called MiniInfer.

The project is aimed at learning and demonstrating production-minded GPU engineering: numerical correctness, controlled benchmarking, profiler-guided optimization, NVIDIA library integration, and clear explanations of why performance changes.

## Current status

**Stage 6 — Transformer-Oriented CUDA Kernels: complete.**

The repository now has validated FP32 Softmax, RMSNorm, interleaved RoPE, SiLU, add, multiply, scale, unfused SwiGLU, and causal-mask paths. At the report shapes, warp Softmax measures `0.200672 ms` (`5.353×` naive) and block RMSNorm measures `0.197504 ms` (`4.599×` naive). Every report record passes its declared tolerance. Stage 7 will begin only when explicitly requested.

## Goals

- Build correct CUDA implementations before optimizing them.
- Develop optimization ladders for reduction, GEMM, and Transformer primitives.
- Compare custom kernels honestly with CPU references and NVIDIA libraries.
- Separate kernel-only latency from end-to-end application latency.
- Use Nsight Systems and Nsight Compute to form and test performance hypotheses.
- Integrate the validated primitives into a small inference runtime that fits a 4 GB GPU.
- Preserve the reasoning, measurements, and limitations behind every meaningful optimization.

## Non-goals

- Producing disconnected CUDA tutorial examples.
- Claiming performance without reproducible measurements.
- Attempting to beat cuBLAS or TensorRT as a headline objective.
- Building a general-purpose tensor framework.
- Starting with a large language model or multi-GPU system.
- Optimizing for datacenter GPUs at the expense of the available RTX 3050 Laptop GPU.

## Architecture direction

```text
                          MiniInfer application
                                   |
                         Transformer execution
                                   |
             +---------------------+---------------------+
             |                     |                     |
      Custom CUDA kernels    NVIDIA libraries      CPU/PyTorch
      reduction, GEMM,       cuBLAS and later      references
      softmax, RMSNorm,      selected baselines
      RoPE, activations
             |                     |                     |
             +---------------------+---------------------+
                                   |
                    Validation and benchmark layer
                                   |
                 +-----------------+-----------------+
                 |                 |                 |
            CUDA events      Nsight tools       JSON/CSV data
```

The detailed component boundaries and evolution plan are in [docs/architecture.md](docs/architecture.md).

## Engineering loop

Every optimization follows the same evidence-driven loop:

```text
Correctness -> Baseline -> Profile -> Identify bottleneck
            -> Form hypothesis -> Optimize -> Revalidate
            -> Rebenchmark -> Document the result
```

An optimization is not accepted because it should be faster. It is accepted only after correctness is preserved and the intended effect is measured under the same benchmark methodology.

## Roadmap

| Stage | Milestone |
| --- | --- |
| 0 | Requirements, environment, architecture, and benchmark design |
| 1 | CUDA/C++ project foundation and device discovery |
| 2 | Reusable validation and benchmark infrastructure |
| 3 | CUDA execution and memory-engineering experiments |
| 4 | Parallel reduction optimization ladder |
| 5 | GEMM optimization ladder and cuBLAS baseline |
| 6 | Transformer-oriented CUDA primitives |
| 7 | Kernel fusion and asynchronous execution |
| 8 | Minimal reusable WarpForge GPU runtime |
| 9 | Validated single-block MiniInfer path |
| 10 | NVIDIA library and framework baselines |
| 11 | Professional profiler-backed performance analysis |
| 12 | Testing, portability, CI, packaging, and portfolio hardening |

Stages are implemented one at a time. Completing one stage does not authorize work on the next.

## Build and verify on the audited Windows machine

CUDA 12.6 must use the installed MSVC 14.44 toolset rather than the newer default MSVC 19.51. From PowerShell, open a child command shell with that toolset selected:

```powershell
cmd.exe /k 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44'
```

Then run these commands in the resulting developer command shell:

```bat
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
ctest --preset windows-msvc-release
build\warpforge-device-info.exe
build\warpforge-benchmark-vector-add.exe --size 1048576 --warmups 10 --iterations 100 --output benchmarks\results\vector_add.json
build\warpforge-benchmark-memory.exe --size 16777216 --rows 2048 --columns 1536 --warmups 10 --iterations 100 --output-dir benchmarks\results\memory
build\warpforge-benchmark-reduction.exe --size 16777216 --block-size 256 --warmups 10 --iterations 100 --output-dir benchmarks\results\reduction
build\warpforge-benchmark-gemm.exe --sizes 256,512,1024,2048 --warmups 10 --iterations 100 --output-dir benchmarks\results\gemm
build\warpforge-benchmark-transformer.exe --rows 4096 --columns 1024 --elements 16777216 --sequence 2048 --heads 8 --head-dim 64 --mask-length 512 --warmups 10 --iterations 100 --output-dir benchmarks\results\stage6
```

The preset targets `sm_86` by default and passes `--use-local-env` to `nvcc` so it reuses the deliberately selected MSVC environment. Other NVIDIA architectures remain configurable with `-DCMAKE_CUDA_ARCHITECTURES=<architectures>`.

## Build targets

| Target | Responsibility |
| --- | --- |
| `WarpForge::warpforge` | Static library containing shared CUDA runtime functionality |
| `warpforge_device_info` | Reports CUDA versions and useful properties for every detected device |
| `warpforge_benchmark_vector_add` | Runs CPU reference, CUDA validation, event timing, statistics, and JSON export |
| `warpforge_benchmark_memory` | Runs Stage 3 block, stride, transpose, transfer, and stream experiments |
| `warpforge_benchmark_reduction` | Validates and measures Stage 4 sum/maximum reduction variants |
| `warpforge_benchmark_gemm` | Validates and measures custom FP32/FP16 GEMM and matched cuBLAS baselines |
| `warpforge_benchmark_transformer` | Validates and measures Stage 6 Transformer-oriented FP32 kernels |
| `warpforge_cuda_smoke` | Validates runtime initialization and basic device discovery through CTest |

`CUDA_CHECK(...)` evaluates a CUDA runtime call once and throws an exception containing the expression, source location, error name, numeric code, and description. A successful kernel launch will only show that work was accepted for execution; later stages must check launch errors immediately and use synchronization at validation boundaries to surface asynchronous execution failures. The helper deliberately does not synchronize every call because unconditional device-wide synchronization would distort performance-sensitive paths.

## Measured development platform

The Stage 0 audit was performed on 2026-09-20.

| Component | Observed value |
| --- | --- |
| Operating system | Windows 11, build 26200.9457 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Compute capability | 8.6 (`sm_86`) |
| Dedicated VRAM | 4096 MiB |
| NVIDIA driver | 616.92 |
| Driver-reported CUDA capability | 13.4 |
| CUDA Toolkit / `nvcc` | 12.6 / 12.6.85 |
| Nsight Compute | 2024.3.2 |
| Nsight Systems | 2024.5.1 and 2025.6.3, installed but not on `PATH` |
| Compute Sanitizer | 2024.3.0 |
| MSVC | 19.44.35228 and 19.51.36256 installed; `cl` not on the normal PowerShell `PATH` |
| CMake | 4.3.1-msvc1 bundled with Visual Studio; not available on `PATH` |
| Git | 2.53.0.windows.2 |

The CUDA Toolkit version and the maximum CUDA version reported by the driver are different concepts. Builds in this project are currently governed by the installed CUDA 12.6 Toolkit.

See [docs/requirements.md](docs/requirements.md) for compatibility findings and Stage 1 prerequisites, and [docs/benchmark_methodology.md](docs/benchmark_methodology.md) for the measurement contract.

## Documentation

- [Requirements and environment](docs/requirements.md)
- [System architecture](docs/architecture.md)
- [Benchmark and validation methodology](docs/benchmark_methodology.md)
- [VectorAdd validation baseline](docs/performance/vector_add.md)
- [CUDA execution and memory engineering](docs/performance/memory.md)
- [Parallel reduction engine](docs/performance/reduction.md)
- [GEMM optimization ladder](docs/performance/gemm.md)
- [Transformer-oriented CUDA kernels](docs/performance/transformer_kernels.md)
- [Benchmark JSON schema v1](benchmarks/schema/v1.json)

## Stage 0 completion report

### Stage 0 summary

The local hardware, CUDA stack, compiler installations, profiling tools, and repository state were audited. Project scope, architecture, numerical-validation policy, benchmark methodology, hardware limits, non-goals, and the staged roadmap are now documented.

### Files created or modified

- `README.md`
- `docs/requirements.md`
- `docs/architecture.md`
- `docs/benchmark_methodology.md`

### Commands executed

- `nvidia-smi` and targeted `nvidia-smi --query-gpu` queries
- `nvcc --version`
- `cmake --version` availability check and bundled CMake version check
- `cl` availability and installed compiler version checks
- `git --version`, repository status, remote, branch, and history checks
- `ncu --version`
- `nsys --version` using the installed executable paths
- `compute-sanitizer --version`
- Visual Studio installation and CUDA host-compiler guard inspection

### Observed toolchain

The primary GPU, CUDA Toolkit, Nsight Compute, Compute Sanitizer, and Git installations are available. CMake 4.3.1-msvc1 is bundled with Visual Studio but not discoverable through the current `PATH`. MSVC is installed but requires an initialized developer environment, and the installed 19.51 toolset exceeds CUDA 12.6's local host-compiler version guard. Nsight Systems is installed but not discoverable through the current `PATH`.

### Decisions

- Target C++17 and configurable CUDA architectures, with `sm_86` as the primary local target.
- Select the MSVC 14.44/19.44 toolset for CUDA 12.6 rather than bypassing the compiler compatibility check.
- Treat CPU implementations, PyTorch, cuBLAS, or another justified NVIDIA library as correctness references according to the operation.
- Keep allocation and transfers out of kernel-only timing unless they are the explicit subject of the experiment.
- Defer all source code, build files, dependency installation, and benchmark outputs to their designated stages.

### Problems

- The bundled CMake executable must be invoked explicitly or made discoverable before Stage 1 configures the project.
- The Stage 1 shell or build configuration must explicitly activate/select MSVC 14.44.
- Nsight Systems needs an explicit executable path or a deliberate `PATH` configuration before command-line profiling.
- Linux and WSL2 portability have not yet been validated.

### Definition-of-Done status

**PASS.** The environment is understood, the architecture and benchmark philosophy are documented, the repository is initialized, and the project scope is clear. Missing build prerequisites are recorded as Stage 1 entry conditions rather than hidden or worked around.

## Stage 1 completion report

### Implemented

- CMake C++17/CUDA project with a configurable architecture and local `sm_86` default
- `warpforge` static library and `WarpForge::warpforge` alias
- Checked CUDA runtime-call utility
- CUDA driver/runtime version and device-property discovery
- `warpforge-device-info` command-line application
- CTest-integrated CUDA runtime smoke test
- Reproducible Windows release configure, build, and test presets

### Files

- Root CMake configuration, Windows preset, and generated-file ignore rules
- Public runtime headers under `include/warpforge/`
- Device discovery implementation under `src/runtime/`
- Device-information application under `apps/device_info/`
- CUDA smoke test under `tests/cuda/`
- Stage 1 status and toolchain documentation updates

### Tests

- Release configuration identifies CUDA 12.6.85 and MSVC 19.44.35228.
- All library, application, and test targets compile and link.
- `cuda.runtime_smoke` passes through CTest on the RTX 3050 Laptop GPU.
- `warpforge-device-info` executes successfully and reports one CUDA device.

### Benchmarks

None. Benchmark infrastructure belongs to Stage 2, and Stage 1 reports no timing or throughput claims.

### Measured results

No performance results were measured. Runtime discovery observed one compute-capability 8.6 device with 16 streaming multiprocessors, a warp size of 32, 1024 maximum threads per block, and 4096 MiB of global memory.

### Concepts learned

- `nvcc` compiles CUDA translation units while using MSVC as its host compiler on Windows.
- CUDA runtime calls can fail immediately, while asynchronous execution errors may surface only at a later synchronization boundary.
- Compute capability describes device features; the build architecture selects generated device code.
- Device limits must be queried rather than inferred from the GPU product name.

### Known limitations

- Only the audited Windows/MSVC/CUDA configuration has been built and run.
- The bundled Ninja executable stalled before launching compiler commands in this OneDrive workspace, so the verified Windows preset uses NMake.
- The smoke test validates the runtime and device path but does not launch a computational kernel; the first validation workload belongs to Stage 2.
- No benchmark, numerical-comparison, RAII memory, stream, or event abstraction exists yet.

### Git commit recommendation

`feat(runtime): add CUDA project foundation and device discovery`

### Definition of Done

**PASS.** A fresh Release configuration builds, the device-information executable runs, the CUDA runtime smoke test passes through CTest, and no kernel optimization work has begun.

### Next stage

Stage 2 will add reusable validation, CUDA-event timing, statistical summaries, structured result export, and a CPU-versus-GPU VectorAdd validation workload. It will not begin until explicitly requested.

## Stage 2 completion report

### Implemented

- Reusable FP32 tolerance comparison and detailed error summaries
- CUDA-event kernel-only timing with warmup and per-sample collection
- Min, mean, median, interpolated p95, and sample standard deviation
- Versioned JSON schema and metadata-rich result export
- Deterministic CPU and CUDA VectorAdd pipeline with seed `2027`
- Configurable benchmark CLI and CTest JSON validation

### Files

- Public benchmark, validation, and VectorAdd headers under `include/warpforge/`
- CPU statistics/validation and CUDA timing/kernel implementations under `src/`
- VectorAdd benchmark application and focused CPU/CUDA/Python tests
- JSON schema v1, measured result, methodology updates, and performance journal

### Tests

Six CTest cases pass in Release mode: validation, statistics, CUDA runtime smoke, VectorAdd correctness, benchmark execution, and JSON/schema parsing. VectorAdd coverage includes empty, tiny, warp-boundary, block-boundary, non-divisible, large, and multiple-block-size cases.

### Benchmarks

The report run used 1,048,576 FP32 elements, 256 threads per block, 4,096 blocks, 50 warmups, 500 measured iterations, and seed `2027`. Allocation, input generation, H2D copies, D2H validation, and JSON output were excluded from the CUDA-event interval.

### Measured results

On the RTX 3050 Laptop GPU, code commit `04ff8c9` measured:

- minimum: `0.071680 ms`
- mean: `0.073380 ms`
- median: `0.072704 ms`
- p95: `0.074405 ms`
- sample standard deviation: `0.003197 ms`
- median-derived effective bandwidth: `173.070 GB/s`
- maximum absolute error: `0`

The bandwidth calculation counts two FP32 reads and one FP32 write per element. These values describe this exact run, configuration, and timing boundary; they are not universal GPU specifications.

### Concepts learned

- CUDA events measure elapsed work in a stream without including host allocation or transfer setup.
- A launch-error check validates enqueue/configuration; synchronizing the stop event surfaces asynchronous execution failures.
- Warmup separates one-time initialization from measured steady-state work.
- Median and p95 complement the minimum and mean when individual samples include interference or outliers.

### Known limitations

- Only kernel-only latency is reported; end-to-end timing is intentionally deferred.
- VectorAdd uses a single baseline implementation and has not been profiled or optimized.
- Effective bandwidth is based on logical algorithmic bytes, not measured physical DRAM transactions.
- The JSON schema is parsed and contract-checked without adding a third-party JSON Schema validator.

### Commits

- `04ff8c9 feat(bench): add CUDA event validation pipeline`
- `bench(vector): record Stage 2 validation baseline` (this measured-result and completion-report commit)

### Definition of Done

**PASS.** One command builds, executes the CPU reference and CUDA workload, validates correctness, measures kernel-only latency, reports statistics, and writes schema-versioned JSON.

### Next stage

Stage 3 will run controlled execution and memory experiments covering block sizes, coalescing and stride, transfers, transpose, pinned memory, asynchronous copies, and stream overlap. It will not begin until explicitly requested.

## Stage 3 completion report

### Implemented

- CPU/CUDA SAXPY, custom FP32 copy, strided gather, and naive/tiled transpose
- Controlled 32, 64, 128, 256, and 512-thread block sweeps
- Controlled stride 1, 2, 4, 8, 16, and 32 access experiment
- Pageable/pinned and synchronous/asynchronous transfer comparisons
- Pinned eight-chunk single/two-stream H2D → SAXPY → D2H pipelines
- Per-case schema-v1 JSON, CSV summary, and Nsight Systems trace evidence

### Files

- Public memory-kernel APIs in `include/warpforge/memory.cuh`
- CUDA implementations in `src/kernels/memory/memory.cu`
- Experiment runner in `apps/benchmarks/memory.cpp`
- CUDA correctness and result-contract tests
- 24 measured JSON records, a CSV summary, profiler excerpts, and this performance report

### Tests

All nine Release CTests pass. Stage 3 correctness covers empty and irregular 1D inputs, every requested block size and stride, rectangular/tail-safe transposes, invalid arguments, and explicit two-stream ordering. All 24 report records pass validation with zero failures.

### Benchmarks and measured results

The report run used code commit `3c30bc7`, seed `2027`, 50 warmups, and 500 samples per case on the RTX 3050 Laptop GPU.

- strongest SAXPY result: `1.074176 ms`, `187.424 GB/s` at 256 and 512 threads
- strongest custom-copy result: `0.731136 ms`, `183.574 GB/s` at 128 threads
- stride 1 → stride 8: `163.840 GB/s` → `40.570 GB/s`
- tiled transpose: `0.141312 ms`, `3.007×` faster than naive
- pinned synchronous round trip: `1.649×` faster than pageable synchronous
- two-stream pipeline: `17.707150 ms`, `0.103%` slower than the single-stream result

Nsight Systems used distinct streams but observed serialized H2D, SAXPY, and D2H intervals. No transfer/compute overlap occurred in the representative capture.

### Concepts learned

- Block size has a plateau; maximum threads per block is not automatically fastest.
- Adjacent warp accesses preserve useful bytes per memory transaction, while stride rapidly reduces them.
- Padded shared-memory tiling can make both transpose directions coalesced.
- Pinned memory can improve transfer throughput, but asynchronous calls require independent work and actual device scheduling overlap to help.
- Separate streams express concurrency; a timeline must confirm whether concurrency occurred.

### Known limitations

- Measurements cover one Windows laptop GPU and one shape per experiment family.
- GPU clocks were not locked; endpoint temperature rose from 64 C to 80 C during the report suite.
- Bandwidth values count logical bytes rather than physical DRAM transactions.
- The Nsight report is used for ordering evidence, not headline latency, because instrumentation adds overhead.

### Commits

- `3c30bc7 feat(memory): add CUDA execution experiment suite`
- `bench(memory): record Stage 3 memory experiments` (this evidence and completion-report commit)

### Definition of Done

**PASS.** Every workload is correct; controlled measurements explain block-size, stride, transfer-memory, and transpose effects; and Nsight Systems confirms that the audited two-stream pipeline did not overlap or improve latency.

### Next stage

Stage 4 will implement FP32 sum and maximum reduction variants, arbitrary-length multi-pass reduction, size-aware validation, and Nsight Compute analysis. It will not begin until explicitly requested.

## Stage 4 completion report

### Implemented

- Public sum/maximum operation and five-variant reduction interfaces
- Double-precision CPU sum and FP32 CPU maximum references
- Arbitrary-length non-atomic, multi-pass CUDA reduction with caller-owned ping-pong workspaces
- Naive interleaved, sequential shared-memory, two-elements-per-thread reduced-divergence, unrolled, and warp-shuffle kernels
- Size-aware sum tolerance, deterministic report runner, schema-v1 JSON, and CSV summary
- Nsight Compute comparison of the baseline and strongest implementation

### Files

- Public API in `include/warpforge/reduction.cuh`
- CUDA reference helpers, dispatch, and kernels in `src/kernels/reduction/reduction.cu`
- Benchmark runner in `apps/benchmarks/reduction.cpp`
- CUDA correctness and result-contract tests
- Ten measured JSON records, summary CSV, compact profiler metrics, and the reduction performance journal

### Tests

All 12 Release CTests pass. Reduction coverage includes both operations, every variant, power-of-two and irregular inputs, values below a warp or block, boundary/tail cases, 1,048,576-element arrays, negative-only maximum inputs, block sizes 32–512, invalid arguments, undersized workspace handling, and double-precision CPU accumulation. All ten report JSON records pass the Stage 4 result validator.

### Benchmarks and measured results

The report run used code commit `26ac9d9`, 16,777,216 FP32 elements, 256 threads per block, seed `2027`, 50 warmups, and 500 measured samples per case.

- strongest sum result: warp shuffle at `0.365536 ms`, `183.590 GB/s`, and `2.709×` baseline speedup
- strongest maximum result: warp shuffle at `0.364544 ms`, `184.090 GB/s`, and `2.716×` baseline speedup
- largest sum absolute error: `0.000143409` against a double-precision CPU reference, within the declared size-aware tolerance
- maximum error: exactly `0` for every variant
- neutral step retained: sequential shared memory measured `0.999×` for sum and `0.998×` for maximum

Nsight Compute measured the first-pass DRAM bandwidth rising from `43.264 GB/s` to `169.265 GB/s`, dynamic shared memory falling from 1,024 to 32 bytes per block, and excessive shared-memory wavefronts falling from 6,356,992 to zero. The optimized first pass is memory-bound by the captured evidence.

### Concepts learned

- A deterministic multi-pass tree handles arbitrary lengths without depending on a global floating-point atomic.
- Removing index arithmetic alone is insufficient when memory access and synchronization remain dominant.
- Loading two elements per thread can reduce both grid size and pass count.
- Warp shuffle keeps the final tree in registers and minimizes shared-memory coordination.
- Higher occupancy is not automatically faster: the optimized kernel improved while achieved occupancy decreased.

### Known limitations

- Measurements cover one RTX 3050 Laptop GPU, one large input, and one report block size.
- GPU clocks were not locked, and the report suite ended at an 86 C GPU temperature.
- FP32 reduction order differs from serial double accumulation; compensated summation is not implemented.
- Effective input bandwidth counts original logical bytes, while profiler DRAM bandwidth is a separate physical-traffic measurement.
- Nsight Compute required a UAC-authorized elevated process for performance counters; no driver setting was changed.

### Commits

- `26ac9d9 feat(reduction): add multi-pass optimization ladder`
- `bench(reduction): record Stage 4 optimization results` (this evidence and completion-report commit)

### Definition of Done

**PASS.** Every variant is correct for the tested edge cases, arbitrary lengths use a non-atomic multi-pass path, the strongest sum and maximum variants have reproducible measurements, and the optimization conclusion is supported by Nsight Compute evidence.

### Next stage

Stage 5 will define GEMM semantics and implement the CPU, custom CUDA, mixed-precision/Tensor Core experiment, and cuBLAS comparison ladder. It will not begin until explicitly requested.

## Stage 5 completion report

### Implemented

- Row-major `C = A × B` problem, variant, backend, and dispatch contracts
- Double-accumulation CPU reference for small-shape validation
- Naive, 16 × 16 shared-tiled, 32 × 32 coalesced, and 64 × 64 register-blocked FP32 CUDA kernels
- FP16-input/FP32-accumulation shared-tiled and WMMA Tensor Core kernels
- Matched row-major cuBLAS FP32 and FP16-input baselines
- Deterministic benchmark CLI, schema-v1 JSON, CSV summaries, and Nsight Compute evidence

### Files

- Public GEMM API in `include/warpforge/gemm.cuh`
- Custom and cuBLAS dispatch in `src/kernels/gemm/gemm.cu`
- Benchmark runner in `apps/benchmarks/gemm.cu`
- CUDA correctness and Python result-contract tests
- Forty measured JSON records across the report and 4,096 feasibility run
- Profiler summary and `docs/performance/gemm.md`

### Tests

All 15 Release CTests pass. GEMM tests cover empty, 1 × 1, irregular rectangular, tile-boundary, and WMMA-compatible shapes; all custom variants; both cuBLAS datatypes; invalid pointers; and incompatible WMMA dimensions. All 40 persisted results pass the Stage 5 validator.

### Benchmarks and measured results

The main report uses code commit `f88127e`, seed `2027`, 50 warmups, and 100 samples at sizes 256–2,048. The 4,096 feasibility run uses 5 warmups and 20 samples because of cumulative runtime and thermal load.

- strongest custom FP32 at 1,024: register blocked, `1.094704 ms`, `1,961.7 GFLOP/s`, `48.17%` of cuBLAS
- matched FP32 cuBLAS at 1,024: `0.527360 ms`, `4,072.1 GFLOP/s`
- custom WMMA at 1,024: `0.587264 ms`, `3,656.8 GFLOP/s`, `36.87%` of FP16 cuBLAS
- matched FP16 cuBLAS at 1,024: `0.216512 ms`, `9,918.5 GFLOP/s`
- 4,096 allocations and execution succeeded; register-blocked FP32 reached `1,378.0 GFLOP/s` and WMMA reached `2,011.8 GFLOP/s`
- largest observed custom errors: `8.2016e-05` FP32 and `2.8801e-04` FP16-input, both within declared tolerances

Nsight Compute confirms that register blocking cuts the 1,024 instrumented duration by nearly fivefold, but 56 registers per thread, uncoalesced stores, and shared-memory bank conflicts remain. WMMA activates the tensor pipeline, while low occupancy and long-scoreboard stalls show that its data movement is not yet competitive with cuBLAS.

### Concepts learned

- Shared tiles reduce redundant global loads; coalesced cooperative loading improves them further.
- Register tiling increases arithmetic intensity but trades against register pressure and occupancy.
- Tensor Core instructions alone do not guarantee library-level performance; tile reuse and pipeline design dominate.
- CPU validation is practical for small shapes, while matched cuBLAS is the trusted reference for large matrices.
- Library-relative claims require identical shapes, datatypes, accumulation, and timing boundaries.

### Known limitations

- Only square, contiguous, row-major `alpha=1`, `beta=0`, no-transpose GEMM is implemented.
- Measurements cover one thermally constrained RTX 3050 Laptop GPU; clocks were not locked.
- The main run ended at 83 C and the 4,096 attempt at 84 C, with unrelated background GPU utilization present.
- WMMA requires all dimensions to be multiples of 16.
- No asynchronous tile copy, double buffering, split-K, batching, autotuning, or compensated accumulation is implemented.

### Commits

- `f88127e feat(gemm): add CUDA optimization ladder and cuBLAS dispatch`
- `bench(gemm): record Stage 5 cuBLAS comparisons` (this evidence and completion-report commit)

### Definition of Done

**PASS.** Every implementation is correct under its declared policy; sizes 256–2,048 are measured; 4,096 was safely attempted; custom performance is reported as a percentage of matched cuBLAS; and the naive, strongest FP32, and Tensor Core kernels have profiler-backed explanations without claiming to beat cuBLAS.

### Next stage

Stage 6 will implement and validate Softmax, RMSNorm, RoPE, elementwise operations, SwiGLU support, and causal masking one coherent operation at a time. It will not begin until explicitly requested.

## Stage 6 completion report

### Implemented

- Stable row-wise Softmax with naive, shared-memory block, and warp-shuffle variants
- RMSNorm with naive and cooperative block variants, FP32 device accumulation, and configurable epsilon
- Interleaved-pair RoPE for contiguous `[batch, sequence, heads, head_dimension]` tensors, position offsets, and configurable base
- SiLU, add, multiply, scale, and an explicit two-kernel unfused SwiGLU baseline
- Standalone causal-mask support for contiguous `[batch, heads, query, key]` scores
- Unified deterministic benchmark CLI, schema-v1 JSON records, CSV summary, and result validator

### Files

- Public APIs in `include/warpforge/softmax.cuh`, `rmsnorm.cuh`, `rope.cuh`, `elementwise.cuh`, and `causal_mask.cuh`
- CUDA/CPU implementations under `src/kernels/transformer/`
- Five focused correctness tests under `tests/cuda/`
- Benchmark runner in `apps/benchmarks/transformer.cpp` and Python result-contract test
- Twelve measured JSON records, summary CSV, and `docs/performance/transformer_kernels.md`

### Tests

All 22 Release CTests pass. Stage 6 coverage includes empty and tiny inputs, irregular and boundary widths, extreme Softmax logits, zero/near-zero RMSNorm values, long-position and invalid-dimension RoPE, elementwise and in-place aliasing rules, SwiGLU intermediate restrictions, mask boundaries and offsets, and the benchmark JSON contract.

The initial full-size benchmark correctly failed at RoPE index 1,022,595 under the generic `1e-5` tolerance. Investigation traced the `1.96084e-4` difference to FP32 inverse-frequency rounding amplified at long positions. A 2,048-token regression case and explicit `atol=2.5e-4`, `rtol=1e-5` policy were added before results were accepted; the double-precision CPU reference was not weakened.

### Benchmarks and measured results

The report run used clean code commit `16c59a5`, seed `2027`, 50 warmups, and 500 CUDA-event samples for each of 12 cases on the RTX 3050 Laptop GPU.

- warp Softmax: `0.200672 ms`, `167.210` logical GB/s, `5.353×` faster than naive
- block Softmax: `0.222208 ms`, `151.005` logical GB/s, `4.834×` faster than naive
- block RMSNorm: `0.197504 ms`, `254.839` logical GB/s, `4.599×` faster than naive
- RoPE: `0.053248 ms`, `157.538` logical GB/s, maximum absolute error `1.96084e-4`
- SiLU/add/multiply/scale: `0.736256–1.079072 ms` and `181.839–187.246` logical GB/s
- unfused SwiGLU: `1.828864 ms`, including both launches and intermediate traffic
- causal mask: `0.090112 ms`, `186.182` logical GB/s, exact agreement

Allocation, transfers, CPU reference work, validation, and serialization are excluded from the event interval. Logical bandwidth is not measured DRAM traffic. All 12 JSON records contain 500 samples, zero failures, and code SHA `16c59a5`.

### Concepts learned

- Stable Softmax needs a row maximum before exponentiation even when the performance question is the reduction strategy.
- Cooperative row reductions recover parallelism that one-thread-per-row baselines leave idle.
- Warp shuffles reduce shared-memory coordination while retaining a small cross-warp summary.
- Long-position RoPE needs a precision policy that accounts for FP32 frequency and angle rounding without changing the trusted reference.
- An explicit unfused SwiGLU path establishes the traffic and launch baseline needed to judge Stage 7 fusion honestly.

### Known limitations

- Transformer primitives are FP32 only and were measured at one shape per operation family on one Windows laptop GPU.
- Endpoint temperature rose from 80 C to 83 C; clocks were not locked.
- Softmax/RMSNorm results are measured but not yet supported by focused Nsight Compute captures; that analysis belongs to Stage 11.
- RoPE supports interleaved pairs only, causal masking is standalone, and SwiGLU remains deliberately unfused.
- No attention orchestration, residual fusion, asynchronous pipeline, tensor abstraction, or MiniInfer block has been introduced.

### Commits

- `aa741b5 feat(transformer): add stable Softmax variants`
- `c3a5b67 feat(transformer): add FP32 RMSNorm kernels`
- `eac3e68 feat(transformer): add interleaved RoPE`
- `9ad7108 feat(transformer): add elementwise SwiGLU and causal mask`
- `83be964 feat(transformer): add Stage 6 benchmark suite`
- `16c59a5 test(transformer): cover long-position RoPE precision`
- `bench(transformer): record Stage 6 kernel results` (this evidence and completion-report commit)

### Definition of Done

**PASS.** Every Stage 6 operation has a trusted reference, CUDA implementation, focused tests, a correctness-gated report benchmark, and documented numerical/performance limitations. Twelve report records pass validation with no fabricated or profiler-derived timing claims.

### Next stage

Stage 7 will compare separate and fused residual + RMSNorm and SiLU + multiply paths, account for logical memory traffic, inspect resource pressure, and revisit the pinned two-stream pipeline with explicit events. It will not begin until explicitly requested.
