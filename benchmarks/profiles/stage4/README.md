# Stage 4 Nsight Compute Evidence

Nsight Compute 2024.3.2 profiled one first-pass FP32 sum launch for the naive-interleaved and warp-shuffle variants. Both use 16,777,216 elements and 256 threads per block. These captures explain the optimization; they are separate from the 50-warmup, 500-sample timing run.

## Commands

Run from the repository root in the verified MSVC 14.44 environment:

```bat
ncu --set full --target-processes all --kernel-name-base function --launch-count 1 --export build\profiles\stage4-interleaved --force-overwrite build\warpforge-benchmark-reduction.exe --size 16777216 --block-size 256 --operation sum --variant naive_interleaved --profile-only --output-dir build\profiles\stage4-profile-output

ncu --set full --target-processes all --kernel-name-base function --launch-count 1 --export build\profiles\stage4-warp-shuffle --force-overwrite build\warpforge-benchmark-reduction.exe --size 16777216 --block-size 256 --operation sum --variant warp_shuffle --profile-only --output-dir build\profiles\stage4-profile-output
```

The first non-elevated attempts returned `ERR_NVGPUCTRPERM`. The same read-only profiling commands succeeded from an elevated process authorized through Windows UAC. WarpForge did not change global NVIDIA driver settings.

## Retention Policy

The raw reports are `build/profiles/stage4-interleaved.ncu-rep` and `build/profiles/stage4-warp-shuffle.ncu-rep`. They are intentionally excluded from Git with the entire build tree. [`ncu_summary.csv`](ncu_summary.csv) is the compact, reviewable export retained with the code.

## Interpretation

- Naive interleaving issues 65,536 blocks, uses 1,024 bytes of dynamic shared memory per block, and accumulates 6,356,992 excessive shared-memory wavefronts.
- Warp shuffle processes two elements per thread, halves the first-pass grid, uses 32 bytes of dynamic shared memory per block, and has zero excessive shared wavefronts.
- Baseline barrier and MIO-throttle stalls fall from 8.65 and 5.75 to 2.56 and 0.56 cycles per issued instruction.
- Profiled DRAM bandwidth rises from 43.264 to 169.265 GB/s. The optimized first pass reaches 86.60% DRAM throughput, and long-scoreboard stalls become dominant.
- Achieved occupancy falls from 89.92% to 74.80% while performance improves. This is direct evidence that maximizing occupancy was not the relevant goal.

The profiler's branch counter does not call the interleaved tree divergent because its control flow is predicated. The shrinking active-lane set, synchronization, and shared-memory access pattern remain real costs and are described with the counters that expose them.
