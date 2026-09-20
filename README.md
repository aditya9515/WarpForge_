# WarpForge

WarpForge is a staged CUDA/C++ performance-engineering project that will grow from measured GPU fundamentals into a small, validated LLaMA-style inference runtime called MiniInfer.

The project is aimed at learning and demonstrating production-minded GPU engineering: numerical correctness, controlled benchmarking, profiler-guided optimization, NVIDIA library integration, and clear explanations of why performance changes.

## Current status

**Stage 2 — Validation and Benchmark Infrastructure: complete.**

The repository now has a verified C++17/CUDA build, reusable CUDA error checking, device discovery, tolerance-based FP32 validation, CUDA-event kernel timing, statistical summaries, versioned JSON export, and a CPU-versus-CUDA VectorAdd pipeline. VectorAdd exists to validate this infrastructure; it is not presented as the project's optimization feature. Stage 3 will begin only when explicitly requested.

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
```

The preset targets `sm_86` by default and passes `--use-local-env` to `nvcc` so it reuses the deliberately selected MSVC environment. Other NVIDIA architectures remain configurable with `-DCMAKE_CUDA_ARCHITECTURES=<architectures>`.

## Foundation targets

| Target | Responsibility |
| --- | --- |
| `WarpForge::warpforge` | Static library containing shared CUDA runtime functionality |
| `warpforge_device_info` | Reports CUDA versions and useful properties for every detected device |
| `warpforge_benchmark_vector_add` | Runs CPU reference, CUDA validation, event timing, statistics, and JSON export |
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
