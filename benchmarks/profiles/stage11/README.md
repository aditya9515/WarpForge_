# Stage 11 profiler reproduction

These commands profile clean implementation commit `f637452` on the RTX 3050 Laptop GPU (CC 8.6) under Windows. The report-date tool versions are Nsight Compute 2024.3.2 and Nsight Systems 2025.6.3. The installed full fixture is `benchmarks/fixtures/miniinfer_full`. Profiled runs are **not** the 50-warmup/500-sample latency measurements under `benchmarks/results/stage11/`.

Run the 31 Release CTests and the existing result validators before interpreting a profile. The exact eight Nsight Compute configurations are in [`capture_ncu.ps1`](capture_ncu.ps1); the two MiniInfer timeline configurations are in [`capture_nsys.ps1`](capture_nsys.ps1). Both scripts resolve the repository from their own location and keep raw output under ignored `build/profiles/stage11/`. They accept an executable-path override on another machine. `capture_ncu.ps1` uses `--set full`, selecting launch, occupancy, memory-workload, speed-of-light, warp-state, register/shared-memory, and roofline sections, and captures one matching launch per case.

```powershell
ctest --test-dir build -C Release --output-on-failure
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks\profiles\stage11\capture_nsys.ps1
```

The normal Windows account received `ERR_NVGPUCTRPERM` for hardware performance counters. A UAC-authorized elevated PowerShell process successfully ran Nsight Compute without changing a driver setting:

```powershell
$script = (Resolve-Path benchmarks\profiles\stage11\capture_ncu.ps1).Path
$process = Start-Process -FilePath powershell.exe -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $script + '"'))
if ($process.ExitCode -ne 0) { throw "Nsight Compute failed: $($process.ExitCode)" }
```

To export the ignored profiler tables for auditing, use:

```powershell
ncu --import build\profiles\stage11\softmax-warp.ncu-rep --page raw --csv --log-file build\profiles\stage11\softmax-warp-raw.csv
ncu --import build\profiles\stage11\softmax-warp.ncu-rep --page details --section SpeedOfLight --section SpeedOfLight_RooflineChart --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --print-details all --log-file build\profiles\stage11\softmax-warp-details.txt
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe' stats --report cuda_gpu_trace --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_api_sum --report cuda_api_trace --report cuda_kern_exec_trace --format csv --output build\profiles\stage11\miniinfer-cublas-stats build\profiles\stage11\miniinfer-cublas.nsys-rep
```

Repeat the two `ncu --import` commands with each capture name and `nsys stats` with `miniinfer-custom` for all rows. [`ncu_summary.csv`](ncu_summary.csv) is a normalized, selected-metric export: raw NCU durations mix `ms` and `us`, while shared-memory values mix bytes and Kbytes; this CSV converts both to microseconds and bytes. Its dominant stall values are NCU *warp cycles per issued instruction*, not an additive time decomposition. `fp32_ffma_thread_instructions_per_cycle` is one roofline metric and is **not** total FP32 FLOP/s. [`nsys_steady_state.csv`](nsys_steady_state.csv) is calculated from the last five complete H2D (0.262 MB) → 18/23 kernel launches → D2H (0.262 MB) passes in `cuda_gpu_trace`, then correlated to `cuda_api_trace` by correlation ID. Each backend uses only stream 13. GPU idle gap is the span less the sum of its non-overlapping GPU operation durations; it does not identify the CPU-side cause. The CPU API span is from the H2D API start to the D2H API end, not a direct measure of GPU work or independent memcpy overhead.

The uninstrumented report commands were:

```powershell
build\warpforge-benchmark-reduction.exe --size 16777216 --block-size 256 --operation all --variant all --warmups 50 --iterations 500 --output-dir benchmarks\results\stage11\reduction
build\warpforge-benchmark-gemm.exe --size 1024 --warmups 50 --iterations 500 --output-dir benchmarks\results\stage11\gemm
build\warpforge-benchmark-transformer.exe --rows 4096 --columns 1024 --elements 16777216 --sequence 2048 --heads 8 --head-dim 64 --mask-length 512 --warmups 50 --iterations 500 --output-dir benchmarks\results\stage11\transformer
python tests\python\validate_reduction_results.py benchmarks\results\stage11\reduction
python tests\python\validate_gemm_results.py benchmarks\results\stage11\gemm
python tests\python\validate_transformer_results.py benchmarks\results\stage11\transformer
```

Raw `.ncu-rep`, `.nsys-rep`, generated SQLite, verbose CSV exports, and binaries remain ignored under `build/`. The committed CSVs and per-sample benchmark JSON preserve the auditable numerical evidence. See [the Stage 11 analysis](../../../docs/performance/profiler_analysis.md) for interpretation and limitations. Profiler usage follows the official [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/) and [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/).
