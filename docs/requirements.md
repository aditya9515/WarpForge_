# WarpForge Requirements and Environment

## Purpose

This document defines the initial operating environment, project scope, constraints, and entry conditions for implementation. It reflects measurements made on the primary development machine on 2026-09-20; it does not assume that every future machine has the same tool versions.

## Project requirements

WarpForge must:

- use modern C++ with C++17 as the minimum language level;
- compile CUDA translation units as `.cu` files through CMake's CUDA language support;
- keep the CUDA architecture configurable while using `sm_86` as the primary local target;
- check CUDA runtime calls and kernel-launch errors;
- use trusted references and tolerance-based floating-point comparisons;
- benchmark with a documented and consistent protocol;
- preserve measurements and profiler evidence for meaningful optimizations;
- fit representative tests and MiniInfer configurations within 4 GB of VRAM;
- remain modular enough to support custom kernels and selected NVIDIA-library backends;
- add Linux/WSL2 portability deliberately after the Windows path is established.

## Verified primary environment

| Area | Observation | Status |
| --- | --- | --- |
| OS | Windows 11, build 26200.9457 | Primary platform |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU | Available |
| Compute capability | 8.6 | Primary architecture is `sm_86` |
| Dedicated VRAM | 4096 MiB | Hard design constraint |
| NVIDIA driver | 616.92 | Available |
| Driver-reported CUDA capability | 13.4 | Driver capability, not Toolkit version |
| CUDA Toolkit | 12.6 | Available |
| `nvcc` | 12.6.85 | Available on `PATH` |
| Visual Studio Build Tools | Visual Studio Build Tools 2026, 18.9.1 | Installed |
| MSVC toolsets | 19.44.35228 and 19.51.36256 | Installed; not active in normal PowerShell |
| CMake | 4.3.1-msvc1 bundled with Visual Studio | Installed; absent from `PATH` |
| Nsight Compute | 2024.3.2 | Available on `PATH` as `ncu` |
| Nsight Systems | 2024.5.1 and 2025.6.3 | Installed; `nsys` absent from `PATH` |
| Compute Sanitizer | 2024.3.0 | Available on `PATH` |
| Git | 2.53.0.windows.2 | Available |
| Python | 3.13.14 | Available |
| Docker CLI | 29.6.2 | Available; daemon not validated in Stage 0 |

Dynamic observations such as temperature, utilization, running processes, and current memory usage are intentionally excluded because they are not stable environment requirements.

## Toolchain compatibility findings

### CUDA driver and Toolkit versions

`nvidia-smi` reports the maximum CUDA level supported by the installed driver, while `nvcc --version` reports the installed compiler Toolkit. The current development baseline is CUDA Toolkit 12.6.85 even though the driver reports CUDA 13.4 capability. Code must not use APIs introduced after CUDA 12.6 unless the baseline is explicitly changed.

### MSVC selection

The local CUDA 12.6 `crt/host_config.h` accepts `_MSC_VER` values from 1910 through 1949 and rejects 1950 or newer. The installed MSVC versions therefore have different status:

- MSVC 19.44 (`_MSC_VER` 1944) is within the local CUDA 12.6 guard.
- MSVC 19.51 (`_MSC_VER` 1951) is outside that guard.

Stage 1 activates the Visual Studio developer environment and selects the 14.44 toolset explicitly. The Windows CMake preset passes `--use-local-env` so `nvcc` retains that selection. The baseline does not use `-allow-unsupported-compiler` merely to force MSVC 19.51 past the guard. A CUDA configure, compile, link, and runtime smoke test now provide the compatibility proof.

### Missing command discovery

- `cmake` is not available in ordinary PowerShell. Visual Studio Build Tools includes CMake 4.3.1-msvc1 and adds it to `PATH` in its developer environment.
- `cl` is not available in ordinary PowerShell, although both compiler installations exist. The verified build initializes the developer environment with `-vcvars_ver=14.44`.
- `nsys` is not available on `PATH`. Both installed versions can be invoked by absolute path; one version should be selected deliberately before profiling begins.

These are diagnosed prerequisites, not evidence that the CUDA Toolkit itself is broken.

## Supported and planned environments

### Supported during the initial implementation

- Windows 11
- x64 host
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.6
- MSVC 19.44 selected from Visual Studio Build Tools
- Primary device architecture `sm_86`

### Planned, not yet validated

- Linux
- WSL2 with NVIDIA GPU access
- Other NVIDIA compute capabilities through configurable CMake architecture settings
- Docker for reproducible tools where it adds value

Documentation must distinguish a planned platform from a tested platform.

## Scope

The staged project includes:

- CUDA execution and memory experiments;
- reduction and GEMM optimization ladders;
- Softmax, RMSNorm, RoPE, activation, and elementwise primitives;
- selected kernel fusion and asynchronous-execution experiments;
- a minimal RAII-oriented GPU runtime;
- a small, validated Transformer-block inference path;
- cuBLAS, PyTorch, and practical TensorRT comparisons;
- Nsight Compute and Nsight Systems analysis;
- reusable tests, benchmark exports, reports, and plots.

## Non-goals

- A general-purpose tensor or deep-learning framework
- A production LLM serving platform
- Large-model training
- Multi-GPU or NCCL work without suitable hardware and a justified experiment
- Unsupported claims about hosted CI having an NVIDIA GPU
- Performance comparisons made with different timing boundaries
- Unmeasured claims that an optimization improved performance

## Resource constraints

- Treat 4 GB VRAM as a hard upper bound, leaving headroom for the display driver and other applications.
- Begin with small deterministic workloads and scale only after measuring allocation requirements.
- Reuse buffers and workspaces where that is part of the experiment, but do not hide setup costs in end-to-end measurements.
- Avoid large Transformer configurations until one small block is numerically validated.
- Record out-of-memory failures as constraints; do not silently reduce workload sizes during a comparison.

## Numerical requirements

- Floating-point outputs use `abs(actual - expected) <= atol + rtol * abs(expected)` rather than exact equality.
- Each operation and datatype declares its tolerance and reference implementation.
- Validation reports at least pass/fail and maximum absolute error; mean absolute error is reported when useful.
- Numerically sensitive algorithms such as Softmax, RMSNorm, FP16 GEMM, and long reductions require operation-specific analysis.
- An optimized implementation is not accepted until it passes the same correctness contract as its baseline.

The complete policy is defined in [benchmark_methodology.md](benchmark_methodology.md).

## Stage 1 entry criteria and result

The Stage 1 entry checks were completed as follows:

1. Bundled CMake 4.3.1-msvc1 is exposed through the Visual Studio developer environment.
2. The documented shell command selects MSVC toolset 14.44 reproducibly.
3. CMake detects CUDA 12.6.85 with host compiler MSVC 19.44.35228.
4. `sm_86` is the configurable default architecture.
5. The static library, device-information application, and runtime smoke test compile and link.
6. CTest and the device-information application run successfully on the primary GPU.

The first Ninja configure path was diagnosed separately: the bundled Ninja process stalled before invoking MSVC in the OneDrive workspace. The verified Windows preset therefore uses NMake. This did not require reinstalling CUDA or bypassing its compiler guard.

## Authoritative references

Technical decisions should prefer the documentation matching the installed Toolkit:

- [CUDA C++ Programming Guide 12.6](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide 12.6](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-best-practices-guide/index.html)
- [CUDA Runtime API 12.6](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/index.html)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/)
