# Stage 12 release and portability notes

## Build modes

The C++17 CPU core (`WarpForge::cpu`) contains FP32 validation and sample statistics. It has no CUDA header, runtime, toolkit, or GPU dependency. The CUDA library (`WarpForge::warpforge`) remains the default and links the CPU core plus CUDA runtime/cuBLAS. `WARPFORGE_ENABLE_CUDA=OFF` excludes all CUDA kernels, GPU applications, and GPU CTests; it does not pretend to test those paths on a hosted runner.

On the audited Windows host, use the installed MSVC 14.44 developer environment for both modes because CUDA 12.6 rejects its newer MSVC 19.51 toolset:

```powershell
cmd.exe /k 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44'
```

Then, in that developer shell:

```bat
cmake --preset windows-msvc-cpu
cmake --build --preset windows-msvc-cpu
ctest --preset windows-msvc-cpu

cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
ctest --preset windows-msvc-release
```

Both presets enable warnings-as-errors for project sources. The CPU preset uses `out/cpu-windows`; the CUDA preset retains the established NMake `build` directory and `sm_86` default. On a Linux host **with a C++ compiler and CMake already installed**, the equivalent CPU-only commands are:

```sh
cmake -S . -B out/cpu-linux -DWARPFORGE_ENABLE_CUDA=OFF -DWARPFORGE_WARNINGS_AS_ERRORS=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/cpu-linux --parallel 2
ctest --test-dir out/cpu-linux --output-on-failure
```

The ordinary CUDA configuration remains `WARPFORGE_ENABLE_CUDA=ON`. The CUDA toolkit is not silently downloaded by CMake or CI.

## Verification boundary

- Local Windows: the CPU-only mode builds with MSVC 19.44, and all three CPU/fixture tests pass. The CUDA mode builds warning-clean with CUDA 12.6.85, and all 31 CTests pass.
- Focused [Compute Sanitizer script](../scripts/run_compute_sanitizer.ps1): runtime memcheck, racecheck, and synccheck, plus MiniInfer memcheck, each report zero errors/hazards. The [Windows GPU driver script](../scripts/test_gpu_windows.ps1) runs the build, all CTests, and optionally this sanitizer suite.
- [Hosted CPU workflow](../.github/workflows/cpu.yml): [run 35818970719](https://github.com/aditya9515/WarpForge_/actions/runs/35818970719) passed on Ubuntu 24.04 and Windows Server 2022 for code commit `2ac42af`. Both jobs compiled and tested the CPU-only mode, checked Python syntax and repository links/JSON, and verified the committed release plot. These jobs make no GPU claim.
- [Formatting policy](../.clang-format) and [check script](../scripts/check_format.py): the policy is checked in, but a pinned `clang-format` binary is not installed locally. Formatting verification and a hosted format gate await separate installation approval; no formatting pass is claimed yet.
- [Opt-in GPU workflow](../.github/workflows/gpu-self-hosted.yml): manual `workflow_dispatch` on `main` only, requiring an explicitly registered Windows runner labeled `warpforge-gpu`. No such runner is assumed to exist; a queued or unrun workflow is not evidence of GPU CI success. Public-repository self-hosted runners should not be exposed to untrusted pull-request code.

The hosted Ubuntu CPU build is validated by the linked CI run. Separately, the local WSL2 host check found Ubuntu 24.04.1 with Python 3 and NVIDIA driver visibility, but no `cmake`, C++ compiler, or Linux CUDA Toolkit. Local WSL2 CPU validation therefore awaits separate approval for Ubuntu `cmake` and `build-essential`; Linux CUDA remains unvalidated. Docker CLI exists, but the Docker Desktop Linux daemon is unavailable, so no Dockerfile or image is claimed as reproducible here.

## Reproducible evidence and release assets

The default benchmark seed is `2027`; fixed MiniInfer fixtures and result records retain seed, sample counts, correctness, hardware, timing scope, and source SHA. [Benchmark methodology](benchmark_methodology.md) separates CUDA-event kernel-only data from host-clock or profiler timing. The plotting script reads validated Stage 11 reduction, GEMM, and Transformer summary CSVs from clean code commit `f637452` and regenerates the [release summary](../benchmarks/results/stage12/release_summary.csv) and [selected latency plot](performance/stage12_latency.svg). Those are **derived Stage 11 measurements**, not new Stage 12 timings.

```powershell
python scripts\check_repository.py
python scripts\plot_benchmarks.py --check
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_gpu_windows.ps1 -RunSanitizer
```

Raw `.ncu-rep`, `.nsys-rep`, SQLite exports, binaries, build directories, virtual environments, downloaded SDKs/models, ONNX graphs, and TensorRT engines remain out of Git. Committed evidence is limited to compact JSON/CSV, selected SVGs, profiling recipes, and written analysis. [The MIT license](../LICENSE) covers the source and documentation.

## Scope limitations

This portfolio release demonstrates one fixed FP32 pre-norm Transformer block, 4 GB GPU-aware engineering, explicit custom/cuBLAS dispatch, PyTorch/TensorRT comparisons, and measured optimization ladders. It does not provide a general tensor/autograd framework, model loader, KV cache, tokenizer, generation loop, multi-block runtime, dynamic TensorRT engine, or cross-device performance guarantee. Hosted CI is a CPU compilation/correctness gate; Windows GPU correctness and performance require a real configured NVIDIA host.
