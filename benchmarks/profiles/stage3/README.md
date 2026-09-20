# Stage 3 Nsight Systems Evidence

The representative trace was collected from clean code commit `3c30bc7` with Nsight Systems 2025.6.3. The profiled run is separate from the report benchmark and is not used for headline latency.

## Capture

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe' profile `
  --trace=cuda --sample=none --cpuctxsw=none --wait=primary `
  --force-overwrite=true --output=build\profiles\stage3-memory `
  build\warpforge-benchmark-memory.exe `
  --size 16777216 --rows 2048 --columns 1536 --profile-only `
  --output-dir build\profiles\stage3-output
```

The CLI needed to run outside the filesystem/process sandbox so its injection profiler could attach. Attempts inside the sandbox with both installed versions, 2024.5.1 and 2025.6.3, timed out after 75 seconds without launching even the device-information executable. The approved out-of-sandbox 2025.6.3 capture completed normally.

## Export

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe' stats `
  --force-export=true `
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace `
  --format csv build\profiles\stage3-memory.nsys-rep
```

- [`nsys_cuda_summary.csv`](nsys_cuda_summary.csv) preserves the exported aggregate kernel and memory-operation rows.
- [`nsys_timeline_excerpt.csv`](nsys_timeline_excerpt.csv) preserves representative chronological H2D → SAXPY → D2H rows from the single- and two-stream sections.

The raw `.nsys-rep` and generated SQLite database remain under ignored `build/profiles/` because they are generated profiler artifacts. The report was 48,839 bytes; it can be regenerated with the command above.

## Observation

The two-stream capture uses CUDA streams 15 and 16. Despite the distinct streams, the selected GPU operations do not overlap. The first operation on stream 16 starts at `1,403,770,038 ns`, after stream 15's preceding D2H ends at `1,403,766,422 ns`, leaving a `3,616 ns` gap. Later chunks show the same serialized H2D → kernel → D2H pattern.

This trace supports the measured conclusion that the audited two-stream pipeline did not produce transfer/compute overlap and did not improve end-to-end latency.
