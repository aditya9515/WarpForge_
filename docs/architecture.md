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

Stages 1 and 2 implement the foundation slice: CUDA error checking, device discovery, tolerance-based FP32 validation, CUDA-event benchmark timing, JSON result export, a diagnostic application, and the VectorAdd validation workload. Stage 3 adds explicit memory/execution experiments. Stage 4 adds a selectable, arbitrary-length sum/maximum reduction ladder with double-precision CPU sum validation and caller-owned multi-pass workspaces. GEMM, Transformer kernels, and MiniInfer remain future work.

## Component responsibilities

### References

References define trusted behavior before a custom GPU implementation is optimized. Depending on the operation, the reference may be a simple CPU C++ routine, NumPy, PyTorch, cuBLAS, or another justified NVIDIA library. A library used as a performance baseline may also serve as a numerical reference, but those roles must be named separately.

### Custom kernel library

The custom-kernel layer will contain focused implementations for:

- vector and memory-access experiments;
- sum and maximum reductions;
- GEMM;
- Softmax and RMSNorm;
- RoPE;
- SiLU and SwiGLU support;
- elementwise and selected fused operations.

Optimization variants remain identifiable so measurements can be connected to the code that produced them. Shared building blocks should be reused when that does not obscure the hardware behavior under study.

### NVIDIA library backends

NVIDIA libraries provide optimized baselines and practical backend choices. cuBLAS will be the primary GEMM comparison. TensorRT will be considered only after an equivalent small model path exists. cuDNN is included only for an experiment with a genuinely matching primitive or graph.

### Runtime layer

The runtime will eventually provide narrow RAII abstractions such as `DeviceBuffer`, `CudaStream`, `CudaEvent`, tensor shape/type metadata, workspace ownership, and kernel-dispatch helpers. The abstractions must expose costs clearly and support move semantics without unnecessary object-oriented hierarchy.

### Validation layer

Validation owns tolerance-based comparisons, error summaries, deterministic test inputs, and operation-specific edge cases. It does not silently change tolerances to make a failing kernel pass.

### Benchmark layer

The benchmark layer owns warmup, repetition, synchronization boundaries, sample statistics, environment metadata, and structured export. It keeps kernel-only and end-to-end timing as different benchmark types.

### MiniInfer

MiniInfer will compose validated primitives into one small LLaMA-style Transformer block before expanding further. It will use deterministic synthetic inputs or a deliberately small configuration, selectable custom/cuBLAS GEMM backends, reusable memory, and a PyTorch reference path. Tokenization and text generation are out of scope until the numerical forward path is correct.

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

Stage 1 exposes `CudaVersions`, `DeviceInfo`, `query_cuda_versions()`, `device_count()`, `query_device()`, and `CUDA_CHECK(...)`. Stage 2 adds validation/result types, CUDA-event measurement, JSON export, and CPU/CUDA VectorAdd entry points. Stage 3 adds focused SAXPY, copy, strided-access, and transpose launch APIs with explicit dimensions, launch choices, and streams. Stage 4 adds `ReductionOperation`, `ReductionVariant`, CPU references, workspace/pass sizing helpers, a size-aware sum validator, and `reduce_cuda(...)`. The reduction call keeps allocation outside the dispatcher, accepts an explicit stream, exposes each optimization variant, and uses ping-pong workspaces for deterministic multi-pass execution. Later APIs remain deferred until the stage that implements and tests them.

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
- The current public API covers device discovery, checked CUDA calls, validation, benchmarking, VectorAdd, Stage 3 memory kernels, and Stage 4 sum/maximum reductions with selectable launch parameters; GEMM and Transformer kernels remain deferred.
- Linux, WSL2, PyTorch, cuBLAS, and TensorRT paths remain planned but unvalidated.
- The 4 GB device requires workload sizes to be selected from measured allocation needs.
