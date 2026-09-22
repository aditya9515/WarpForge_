# WarpForge GPU Runtime

## Scope

Stage 8 replaces benchmark-local device, stream, and event owners with one small runtime layer. It deliberately does not become a general tensor framework. The design question is whether WarpForge can provide reliable ownership and checked dispatch without hiding allocation, synchronization, or backend costs.

## Public Surface

| Type | Ownership and purpose | Hidden GPU work |
| --- | --- | --- |
| `DeviceBuffer<T>` | Move-only `cudaMalloc` allocation with element/byte counts and async copy helpers | None beyond explicitly called allocate/copy/reset operations |
| `CudaStream` | Move-only nonblocking CUDA stream | None; synchronization is explicit |
| `CudaEvent` | Move-only CUDA event with record/synchronize/elapsed-time operations | None beyond the requested event operation |
| `TensorShape` | Checked dimensions, element count, and dtype-sized byte count | None |
| `DType` | FP32 or FP16 only | None |
| `Tensor` | Contiguous owning device bytes plus shape/dtype metadata | Allocation only on construction/resize |
| `TensorView` | Non-owning pointer, shape, and dtype | None |
| `DeviceWorkspace` | Reusable capacity plus aligned bump-allocation cursor | Allocation only on construction/capacity growth |

Every resource owner is non-copyable and movable. Destructors are `noexcept` and perform best-effort cleanup. `reset()` reports cleanup errors when the caller needs them, `release()` transfers raw ownership where appropriate, and `native_handle()` exposes the CUDA pointer/handle directly.

The default `TensorShape{}` represents zero elements. Shape products and dtype byte counts reject `size_t` overflow. `TensorView` accepts a null pointer only for an empty shape and checks typed access against its recorded dtype.

## Workspace Contract

`DeviceWorkspace` owns one CUDA allocation and returns capacity-bounded slices. Alignment must be a non-zero power of two; CUDA allocation alignment supplies the base guarantee and the runtime aligns every cursor offset. `clear()` resets only the cursor, allowing the same pointers/capacity to be reused. `reserve()` is a no-op when capacity is sufficient and replaces the allocation only when growth is required.

This is intentionally a single-threaded bump allocator. It does not track individual lifetimes or free slices, and it never grows implicitly from `allocate_bytes()`.

## GEMM Dispatch

The TensorView overload checks that:

- A contains exactly `M*K` elements;
- B contains exactly `K*N` elements;
- C contains exactly `M*N` elements;
- A and B have matching FP32 or FP16 dtype;
- C is FP32.

It then calls the existing FP32 or FP16 GEMM path with the caller's explicit `GemmDispatch`, cuBLAS handle, and CUDA stream. TensorView construction happens before every measured launch loop, so metadata validation and allocation do not enter CUDA-event timings.

## Validation

The Release suite contains 26 passing CTests. The focused runtime test covers:

- compile-time move-only/noexcept contracts and coherent moved-from objects;
- zero-size buffers and tensors;
- shape/byte overflow and invalid view type access;
- asynchronous buffer and tensor H2D/D2H round trips;
- stream and event record/move/synchronize behavior;
- explicit release and checked reset;
- 256/128-byte workspace alignment, cursor reuse, growth, and exhaustion;
- a real CUDA allocation failure;
- custom tiled and cuBLAS FP32 GEMM through the same TensorView dispatcher;
- rejection of a mismatched GEMM output view.

Compute Sanitizer 2024.3.0 was run as:

```bat
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 build\warpforge_runtime_test.exe --skip-allocation-failure
```

It reported `0 errors` and `0 bytes leaked in 0 allocations`. The ordinary CTest run includes the real out-of-memory case. The sanitizer command skips only that case because Compute Sanitizer correctly counts the deliberately failed `cudaMalloc` call as an API error.

## Compatibility Measurement Method

The report binaries were configured and built from clean code commit `e8803f2`. Both runs used deterministic seed `2027`, 50 warmups, and 500 measured samples per case on the RTX 3050 Laptop GPU.

- GEMM used a 1,024 × 1,024 × 1,024 problem and all eight Stage 5 implementations/backends.
- Fusion used the exact Stage 7 report shapes: 4,096 × 1,024 residual RMSNorm, 16,777,216-element SwiGLU, and a 16,777,216-element eight-chunk pinned pipeline.
- Kernel-only records use CUDA events; the pipeline retains its explicit end-to-end host interval.
- Allocation, deterministic initialization, reference work, validation, and result export remain outside measured intervals.

The authoritative records are in the [GEMM summary](../../benchmarks/results/stage8/gemm/summary.csv) and [fusion/pipeline summary](../../benchmarks/results/stage8/fusion/summary.csv). These reruns test compatibility after ownership/dispatch migration; they do not establish a new kernel optimization.

## Measured GEMM Results

| Input/path | Median ms | p95 ms | GFLOP/s | cuBLAS-relative | Validation |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 custom naive | 4.822992 | 4.913000 | 445.260 | 11.996% | PASS |
| FP32 custom tiled | 3.600896 | 3.638309 | 596.375 | 16.067% | PASS |
| FP32 custom coalesced | 3.185536 | 3.202850 | 674.136 | 18.162% | PASS |
| FP32 custom register blocked | 1.188864 | 1.198176 | 1,806.332 | 48.665% | PASS |
| FP32 cuBLAS | 0.578560 | 0.619571 | 3,711.773 | 100% | PASS |
| FP16 custom tiled | 3.821568 | 3.960923 | 561.938 | 5.868% | PASS |
| FP16 custom WMMA | 0.744448 | 0.750592 | 2,884.666 | 30.124% | PASS |
| FP16 cuBLAS | 0.224256 | 0.245760 | 9,576.037 | 100% | PASS |

The FP32 optimization order is unchanged. Register-blocked FP32 reaches 48.66% of cuBLAS versus 48.17% in the earlier Stage 5 report. Absolute latency changed with normal thermal/background variability, but the matched backend ratio and implementation ordering support the conclusion that the checked view bridge did not alter the dispatched work.

## Measured Fusion/Pipeline Results

| Operation | Path | Median ms | p95 ms | Within-run speedup | Validation |
| --- | --- | ---: | ---: | ---: | --- |
| residual + RMSNorm | separate | 0.475136 | 0.489194 | 1.000× | PASS |
| residual + RMSNorm | fused | 0.319264 | 0.322560 | 1.488× | PASS |
| SwiGLU | separate | 1.811456 | 1.829606 | 1.000× | PASS |
| SwiGLU | fused | 1.110976 | 1.138506 | 1.631× | PASS |
| pinned SiLU pipeline | single stream | 12.671850 | 12.919585 | 1.000× | PASS |
| pinned SiLU pipeline | two streams | 12.700850 | 13.004745 | 0.998× | PASS |

The four kernel medians are within 1.31% of their Stage 7 values, and the two retained fusion decisions are unchanged. The pipeline's absolute host-timed latency improved in this run, but two streams remain slightly slower than one; the earlier Nsight Systems conclusion of no observed overlap therefore remains the operative hardware result.

## Decisions and Limitations

- Retain the small runtime: it centralizes ownership and failure handling while exposing every native handle and synchronization boundary.
- Retain pointer-based APIs for non-GEMM kernels until Stage 9 has a real view-based consumer; adding unused overloads would expand surface without improving execution.
- Keep pinned host buffer ownership local for now because Stage 8 did not require a pinned-memory type.
- `Tensor` is contiguous-only; there are no strides, broadcasting, implicit conversion, layout transformation, autograd, or global mutable allocator.
- `DeviceWorkspace` is not thread-safe and does not manage slice lifetimes.
- Cleanup errors cannot escape destructors. Callers that need diagnostics use explicit `reset()`.
- Performance data covers one unlocked-clock Windows laptop GPU and is reported as compatibility evidence, not as a universal runtime-overhead claim.

Stage 9 will use these owners/views/workspace to construct one deterministic Transformer block. It will begin only after explicit user authorization.
