# Stage 5 Nsight Compute Evidence

Nsight Compute 2024.3.2 profiled one 1,024 × 1,024 × 1,024 launch for the naive FP32, register-blocked FP32, and WMMA FP16-input/FP32-accumulation kernels. These captures explain kernel behavior and remain separate from the uninstrumented report timing.

## Commands

Run from the repository root:

```bat
ncu --set full --target-processes all --kernel-name-base function --kernel-name "regex:.*naive_fp32_kernel.*" --launch-count 1 --export build\profiles\stage5-naive-fp32 --force-overwrite build\warpforge-benchmark-gemm.exe --size 1024 --warmups 0 --iterations 1 --implementation custom_naive_fp32 --profile-only --output-dir build\profiles\stage5-profile-output

ncu --set full --target-processes all --kernel-name-base function --kernel-name "regex:.*register_blocked_fp32_kernel.*" --launch-count 1 --export build\profiles\stage5-register-blocked-fp32 --force-overwrite build\warpforge-benchmark-gemm.exe --size 1024 --warmups 0 --iterations 1 --implementation custom_register_blocked_fp32 --profile-only --output-dir build\profiles\stage5-profile-output

ncu --set full --target-processes all --kernel-name-base function --kernel-name "regex:.*wmma_fp16_fp32_kernel.*" --launch-count 1 --export build\profiles\stage5-wmma-fp16-fp32 --force-overwrite build\warpforge-benchmark-gemm.exe --size 1024 --warmups 0 --iterations 1 --implementation custom_wmma_fp16_fp32 --profile-only --output-dir build\profiles\stage5-profile-output
```

Hardware-counter collection required a UAC-elevated profiler process on this Windows machine. No NVIDIA driver setting was changed.

## Retention Policy

The raw reports remain ignored under `build/profiles/`:

- `stage5-naive-fp32.ncu-rep`
- `stage5-register-blocked-fp32.ncu-rep`
- `stage5-wmma-fp16-fp32.ncu-rep`

[`ncu_summary.csv`](ncu_summary.csv) is the compact export retained in Git.

## Interpretation

- Naive FP32 spends 12.934 cycles per issued instruction on long-scoreboard stalls and 8.031 on LG-throttle stalls. Nsight Compute reports only 18 useful bytes per 32-byte global-load sector and just 6% of FP32 roofline peak, despite the aggregate throughput indicator reaching 96.88%. It is load-pipe/instruction constrained rather than DRAM-bandwidth saturated.
- Register blocking reduces the instrumented duration from 7,415.328 to 1,501.504 µs and the grid from 4,096 to 256 blocks. The tradeoff is 56 registers per thread, 66.67% theoretical occupancy, 8,512 bytes of static shared memory, 393,216 excessive global sectors, and 3,670,016 excessive shared wavefronts. The strongest FP32 kernel still has clear store-coalescing and bank-conflict work remaining.
- WMMA executes half-precision tensor operations, but the tensor pipe is active for only 11.576% of elapsed cycles. One warp per block yields 33.33% theoretical occupancy; 30.525 of 36.290 cycles per issued instruction are long-scoreboard stalls, and global loads use only 16 of 32 bytes per sector. The experiment proves Tensor Core execution, not an optimized Tensor Core data-movement strategy.

Profiler durations include replay/instrumentation effects and are not headline benchmark results.
