# WarpForge Architecture

## Architectural goals

WarpForge should make GPU behavior visible rather than hiding it behind a large abstraction layer. The architecture separates reference implementations, CUDA kernels, runtime resources, validation, measurement, and application orchestration so each layer can be tested and reasoned about independently.

The design prioritizes:

- correctness before optimization;
- explicit ownership of GPU resources;
- comparable measurements;
- small, inspectable interfaces;
- reusable primitives without building a general-purpose framework;
- incremental evolution, one stage at a time.

## Logical system

```text
                          +-----------------------+
                          | MiniInfer application |
                          +-----------+-----------+
                                      |
                          +-----------v-----------+
                          | Transformer execution |
                          +-----------+-----------+
                                      |
                +---------------------+---------------------+
                |                     |                     |
        +-------v--------+    +-------v--------+    +-------v--------+
        | Custom kernels |    | NVIDIA library |    | Reference path |
        | CUDA C++       |    | cuBLAS, etc.   |    | CPU/PyTorch   |
        +-------+--------+    +-------+--------+    +-------+--------+
                |                     |                     |
                +---------------------+---------------------+
                                      |
                +---------------------v---------------------+
                | Runtime, validation, and benchmark layer  |
                +----------+----------------------+---------+
                           |                      |
              +------------v----------+  +--------v---------+
              | CUDA events and data  |  | Profiling tools  |
              | export (JSON/CSV)     |  | NCU / Nsight Sys |
              +-----------------------+  +------------------+
```

Stages 1 and 2 implement the foundation slice: CUDA error checking, device discovery, tolerance-based FP32 validation, CUDA-event benchmark timing, JSON result export, a diagnostic application, and the VectorAdd validation workload. Stage 3 adds explicit memory/execution experiments. Stage 4 adds a selectable, arbitrary-length sum/maximum reduction ladder. Stage 5 adds row-major custom FP32/FP16 GEMM, WMMA, and cuBLAS backend dispatch. Stage 6 adds validated FP32 Softmax, RMSNorm, RoPE, elementwise, unfused SwiGLU, and causal-mask primitives. Stage 7 adds measured fused residual + RMSNorm and SwiGLU paths plus an explicit-event pinned pipeline. Stage 8 adds transparent CUDA resource ownership, tensor metadata/views, reusable device workspace, and checked view-based GEMM dispatch. Stage 9 composes those layers into one fixture-validated FP32 MiniInfer block with custom/cuBLAS projection dispatch. Stage 10 adds equivalent eager PyTorch CUDA and fixed-shape TensorRT baselines without expanding the C++ runtime surface. Stage 11 links optimization conclusions to NCU counters and MiniInfer timelines. Stage 12 separates a CUDA-free C++ core for hosted compile/tests while preserving CUDA as the default local build.

### Build dependency boundary

`WarpForge::cpu` contains `validate_fp32` and benchmark sample statistics with only standard C++17 headers. Its public headers are `validation.hpp` and `benchmark_statistics.hpp`. `WarpForge::warpforge` links that core and supplies CUDA-event timing, metadata, kernels, runtime, and MiniInfer; its CUDA-dependent `benchmark.hpp` retains the same public statistics type through the new CPU header. `WARPFORGE_ENABLE_CUDA=OFF` creates only the CPU library and CPU/fixture tests, avoiding CUDA compiler detection and toolkit lookup altogether. GPU executables and their tests are compiled only in the default CUDA-enabled mode. This is a narrow portability seam, not a CPU implementation of GPU inference.

## Component responsibilities

### References

References define trusted behavior before a custom GPU implementation is optimized. Depending on the operation, the reference may be a simple CPU C++ routine, NumPy, PyTorch, cuBLAS, or another justified NVIDIA library. A library used as a performance baseline may also serve as a numerical reference, but those roles must be named separately.

### Custom kernel library

The custom-kernel layer contains focused implementations for:

- vector and memory-access experiments;
- sum and maximum reductions;
- GEMM;
- Softmax and RMSNorm;
- RoPE;
- SiLU and SwiGLU support;
- elementwise and selected fused operations.

Optimization variants remain identifiable so measurements can be connected to the code that produced them. Shared building blocks should be reused when that does not obscure the hardware behavior under study.

### NVIDIA library backends

NVIDIA libraries provide optimized baselines and practical backend choices. cuBLAS is the primary GEMM comparison. TensorRT 10.7 consumes a fixed-shape, standard-operator ONNX version of the Stage 9 block; its generated engines remain outside the source tree. cuDNN is included only for an experiment with a genuinely matching primitive or graph, which Stage 10 did not identify.

### Runtime layer

The runtime provides narrow move-only `DeviceBuffer<T>`, `CudaStream`, `CudaEvent`, `Tensor`, and `DeviceWorkspace` owners plus `TensorShape`, FP32/FP16 `DType`, and non-owning `TensorView`. Destruction is non-throwing; explicit reset paths report cleanup failures. Native pointers and CUDA handles remain directly available, and no implicit synchronization, allocation, conversion, broadcasting, or layout transformation occurs. The TensorView GEMM bridge validates element counts and dtypes before selecting the existing custom CUDA or cuBLAS path.

### Validation layer

Validation owns tolerance-based comparisons, error summaries, deterministic test inputs, and operation-specific edge cases. It does not silently change tolerances to make a failing kernel pass.

### Benchmark layer

The benchmark layer owns warmup, repetition, synchronization boundaries, sample statistics, environment metadata, and structured export. It keeps kernel-only and end-to-end timing as different benchmark types.

### MiniInfer

MiniInfer composes the validated primitives into one pre-norm LLaMA-style Transformer block. `MiniInferWeights` owns and caches views for nine parameter tensors; `MiniInferWorkspace` allocates all 17 internal activations from one aligned reusable device allocation; and `MiniInferBlock` orchestrates the explicit-stream forward path into a caller-owned output view. Custom and cuBLAS projection GEMMs are selectable without changing the remaining kernel sequence.

The CPU-PyTorch reference generator emits deterministic little-endian FP32 fixtures with a versioned manifest, shapes, byte counts, and SHA-256 hashes. An optional intermediate observer validates 18 named semantic boundaries without adding copies or synchronization to normal inference calls. Tokenization, KV cache, text generation, multiple layers, and model downloads remain outside this bounded runtime.

The Stage 10 Python mirror loads those same fixture bytes and exposes normal forward and intermediate-observation paths. It supplies an eager PyTorch CUDA baseline and a fixed-shape opset-17 ONNX export. TensorRT is an external serialized-engine baseline rather than a backend hidden inside `MiniInferBlock`; this keeps native allocation, stream, and backend ownership unchanged and makes timing boundaries explicit.

## Primary data flows

### Kernel development flow

```text
Define operation
    -> implement trusted reference
    -> implement baseline CUDA kernel
    -> compare numerically
    -> benchmark consistently
    -> profile the important variant
    -> identify a bottleneck
    -> state an optimization hypothesis
    -> change one meaningful factor
    -> revalidate and rebenchmark
    -> document the evidence
```

### MiniInfer execution flow

```text
Input
  -> RMSNorm
  -> QKV projection
  -> RoPE
  -> scaled causal self-attention
  -> output projection
  -> residual
  -> RMSNorm
  -> SwiGLU/MLP
  -> projection
  -> residual
```

Each intermediate operation must be validated before the complete block is treated as correct.

## Intended repository evolution

Directories are added only when their stage needs them. The long-term shape is:

```text
WarpForge/
|-- CMakeLists.txt
|-- README.md
|-- cmake/
|-- include/warpforge/
|-- src/runtime/
|-- src/kernels/
|-- apps/device_info/
|-- apps/benchmarks/
|-- apps/miniinfer/
|-- tests/cpu/
|-- tests/cuda/
|-- python/
|-- benchmarks/configs/
|-- benchmarks/results/
|-- profiles/
|-- docs/performance/
`-- docker/
```

Stage 0 creates only the root README and design documents. Empty future directories are not created as placeholders.

## Interface conventions

When implementation begins:

- public C++ symbols use the `warpforge` namespace;
- public headers use `.hpp` or `.cuh` according to CUDA exposure;
- CUDA implementation files use `.cu`;
- resource owners are non-copyable and movable where appropriate;
- CUDA runtime errors use a common checked-call utility;
- kernel launches receive an immediate launch-error check;
- synchronization is explicit and justified;
- no unconditional device-wide synchronization is placed in a performance-sensitive path merely for convenience;
- matrix multiplication exposes a small backend choice between custom CUDA and cuBLAS without leaking backend-specific state into MiniInfer orchestration.

Stage 1 exposes `CudaVersions`, `DeviceInfo`, `query_cuda_versions()`, `device_count()`, `query_device()`, and `CUDA_CHECK(...)`. Stage 2 adds validation/result types, CUDA-event measurement, JSON export, and CPU/CUDA VectorAdd entry points. Stage 3 adds focused memory launch APIs. Stage 4 adds selectable multi-pass reduction. Stage 5 adds `GemmProblem`, `GemmVariant`, `GemmBackend`, `GemmDispatch`, sizing/tolerance helpers, CPU reference, and FP32/FP16 launch functions. GEMM calls accept explicit CUDA streams and an externally owned cuBLAS handle so allocation and library state remain visible. Stage 6 adds `SoftmaxVariant`, `RmsNormVariant`, `RopeProblem`, `CausalMaskProblem`, CPU references, elementwise/SwiGLU entry points, and explicit-stream CUDA launch functions. Stage 7 adds fused residual-RMSNorm and SwiGLU entry points. Stage 8 adds `DeviceBuffer<T>`, `CudaStream`, `CudaEvent`, `TensorShape`, `DType`, `Tensor`, `TensorView`, and `DeviceWorkspace`; all benchmark owners use these runtime types, while GEMM adds a checked TensorView overload. Stage 9 adds `AttentionProblem`, `MiniInferConfig`, owning weights/workspace, block execution, deterministic fixture loading, backend selection, and an optional intermediate-validation hook. Stage 10 adds Python-side fixture loading, an equivalent PyTorch module, standard ONNX export, TensorRT engine validation, and normalized framework result export. Non-GEMM kernels keep their explicit contiguous pointers, dimensions, and streams rather than gaining unused framework-style wrappers.

## Stage boundaries

| Stages | Architectural result |
| --- | --- |
| 0 | Requirements and design contract |
| 1-2 | Buildable CUDA foundation plus shared validation/benchmark infrastructure |
| 3-7 | Measured kernel library and execution experiments |
| 8 | Minimal reusable runtime abstractions |
| 9 | One validated Transformer block |
| 10-11 | External baselines and profiler-backed analysis |
| 12 | Portability, automation, packaging, and portfolio hardening |

Work stops at each boundary for review. Later-stage interfaces must not be introduced early without an immediate consumer.

## Design decisions

- **CMake over IDE-only projects:** enables repeatable Windows builds and later Linux/WSL2 portability.
- **Configurable architecture with local `sm_86` default:** serves the available GPU without making it a library-wide assumption.
- **Custom and NVIDIA-library paths:** teaches kernel engineering while preserving a realistic optimized backend.
- **Small runtime surface:** keeps allocations, synchronization, and dispatch costs understandable.
- **Structured benchmark output:** connects code revisions to plots and reports without treating raw numbers as timeless facts.
- **One Transformer block first:** bounds memory use and makes intermediate numerical validation tractable.

## Current limitations

- The foundation has been compiled and run only on the audited Windows configuration.
- The current public API covers device discovery, checked CUDA calls, validation, benchmarking, move-only CUDA ownership, contiguous FP32/FP16 tensor views, reusable workspace, VectorAdd, memory kernels, reduction, row-major GEMM with custom/cuBLAS dispatch, FP32 Transformer primitives/fusions, attention products, and one FP32 MiniInfer block.
- GEMM currently supports contiguous no-transpose matrices with `alpha=1`, `beta=0`, and FP32 output; it is deliberately not a general BLAS wrapper.
- MiniInfer orchestration is contiguous FP32 only; non-GEMM Transformer kernels still use explicit pointers/dimensions, and FP16 block execution does not yet exist.
- Fusion is limited to residual + RMSNorm and SiLU + multiply; attention/GEMM/mask fusion is not implemented.
- Hosted Ubuntu 24.04 validates only the CUDA-free CPU library and fixture tests. Local WSL2 and Linux CUDA remain unvalidated. CPU PyTorch, PyTorch CUDA, native custom/cuBLAS MiniInfer, and TensorRT 10.7 FP32/mixed-FP16 paths are validated on Windows for the single fixed Stage 10 shape.
- TensorRT remains an offline fixed-shape baseline: engines are platform-specific generated artifacts, the C++ runtime does not load them, and no custom plugin or dynamic-shape path exists.
- The 4 GB device requires workload sizes to be selected from measured allocation needs.
