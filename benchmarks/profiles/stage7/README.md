# Stage 7 Profiler Evidence

These captures explain the Stage 7 fusion and two-stream results from clean code commit `b21cdf0`. Profiler runs are separate from the uninstrumented report benchmark and are not used as headline timings.

## Nsight Compute

Nsight Compute 2024.3.2 captured the two separate kernels and fused kernel for each operation at the report shapes. The command template was:

```bat
ncu --set full --target-processes all --kernel-name-base function --kernel-name "regex:.*KERNEL_NAME.*" --launch-count 1 --export build\profiles\REPORT_NAME --force-overwrite build\warpforge-benchmark-fusion.exe --rows 4096 --columns 1024 --elements 16777216 --pipeline-elements 16777216 --chunks 8 --warmups 0 --iterations 1 --fusion-only --output-dir build\profiles\stage7-fusion-output
```

`KERNEL_NAME` was replaced with `add_kernel`, `rmsnorm_block_kernel`, `residual_rmsnorm_fused_kernel`, `silu_kernel`, `multiply_kernel`, and `swiglu_fused_kernel`. Windows denied performance-counter access to a normal process with `ERR_NVGPUCTRPERM`, so the six commands ran in a UAC-authorized elevated PowerShell process. No driver setting was changed.

Compact exported metrics are in [`ncu_summary.csv`](ncu_summary.csv). Raw `.ncu-rep` files remain ignored under `build/profiles/`.

| Kernel | Instrumented duration | DRAM throughput | Registers/thread | Dynamic shared memory | Theoretical / achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual add | 272.160 us | 94.60% | 16 | 0 B | 100% / 84.18% |
| RMSNorm block | 217.696 us | 78.61% | 18 | 1,024 B | 100% / 93.97% |
| fused residual + RMSNorm | 325.600 us | 83.61% | 18 | 1,024 B | 100% / 86.74% |
| SiLU | 756.704 us | 90.65% | 16 | 0 B | 100% / 81.14% |
| multiply | 1,079.232 us | 95.40% | 16 | 0 B | 100% / 83.62% |
| fused SwiGLU | 1,121.248 us | 91.91% | 16 | 0 B | 100% / 85.99% |

The separate residual path totals 489.856 us in the instrumented captures versus 325.600 us fused. Fusion removes one launch and the logical intermediate write/read while matching the RMSNorm kernel's 18-register and 1,024-byte shared-memory footprint. The fused kernel reaches 83.61% of peak DRAM throughput and only 2% of FP32 roofline peak, supporting a memory-bound classification.

The separate SwiGLU captures total 1,835.936 us versus 1,121.248 us fused. The fused path keeps the same 16 registers per thread and no shared memory while removing the intermediate tensor and one launch. Its 91.91% DRAM throughput supports the same memory-bound conclusion.

## Nsight Systems

Nsight Systems 2025.6.3 captured one correctness execution and one measured execution for the single- and two-stream pipelines:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe' profile `
  --trace=cuda --sample=none --cpuctxsw=none --wait=primary `
  --force-overwrite=true --output=build\profiles\stage7-pipeline `
  build\warpforge-benchmark-fusion.exe `
  --rows 4096 --columns 1024 --elements 16777216 `
  --pipeline-elements 16777216 --chunks 8 --warmups 0 --iterations 1 `
  --pipeline-only --output-dir build\profiles\stage7-pipeline-output
```

The raw `.nsys-rep` and generated SQLite file remain ignored. [`nsys_cuda_summary.csv`](nsys_cuda_summary.csv) retains aggregate kernel/copy evidence, and [`nsys_timeline_excerpt.csv`](nsys_timeline_excerpt.csv) retains representative operations from streams 15 and 16.

Across all 48 H2D, SiLU, and D2H operations in the two-stream sections, no GPU intervals overlap. The minimum gap between consecutive operations is 1,728 ns. The capture alternates streams 15 and 16, proving that distinct streams and event completion were used, but each H2D → kernel → D2H sequence remains serialized. The profiled device reports one asynchronous copy engine, which is consistent with the absence of bidirectional-copy overlap.

The timeline evidence and the report benchmark agree: the audited two-stream pipeline is correct but does not overlap and should not replace the simpler single-stream path on this machine.
