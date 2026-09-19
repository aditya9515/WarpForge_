# WarpForge

WarpForge is a staged CUDA/C++ performance-engineering project that will grow from measured GPU fundamentals into a small, validated LLaMA-style inference runtime called MiniInfer.

The project is aimed at learning and demonstrating production-minded GPU engineering: numerical correctness, controlled benchmarking, profiler-guided optimization, NVIDIA library integration, and clear explanations of why performance changes.

## Current status

**Stage 0 — Requirements, Environment and Design: complete.**

This repository currently contains design and methodology documentation only. It intentionally has no CUDA kernels, CMake project, generated benchmark data, or performance claims. Stage 1 will begin only after its prerequisites are satisfied and the stage is explicitly started.

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
