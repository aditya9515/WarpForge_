#pragma once

#include <warpforge/runtime.cuh>
#include <warpforge/validation.hpp>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

struct GemmProblem final {
    std::size_t m{};
    std::size_t n{};
    std::size_t k{};
};

enum class GemmVariant {
    naive_fp32,
    tiled_fp32,
    coalesced_fp32,
    register_blocked_fp32,
    tiled_fp16_fp32,
    wmma_fp16_fp32,
};

enum class GemmBackend {
    custom,
    cublas,
};

struct GemmDispatch final {
    GemmBackend backend{GemmBackend::custom};
    GemmVariant variant{GemmVariant::naive_fp32};
};

[[nodiscard]] const char* gemm_variant_name(GemmVariant variant) noexcept;
[[nodiscard]] const char* gemm_backend_name(GemmBackend backend) noexcept;
[[nodiscard]] bool gemm_problem_is_empty(const GemmProblem& problem) noexcept;
[[nodiscard]] bool gemm_variant_supports_problem(GemmVariant variant,
                                                 const GemmProblem& problem) noexcept;
[[nodiscard]] std::size_t gemm_a_elements(const GemmProblem& problem);
[[nodiscard]] std::size_t gemm_b_elements(const GemmProblem& problem);
[[nodiscard]] std::size_t gemm_c_elements(const GemmProblem& problem);
[[nodiscard]] double gemm_flop_count(const GemmProblem& problem) noexcept;
[[nodiscard]] Tolerance gemm_fp32_tolerance(std::size_t inner_dimension) noexcept;
[[nodiscard]] Tolerance gemm_fp16_tolerance(std::size_t inner_dimension) noexcept;

void gemm_cpu_fp32(const float* a, const float* b, float* c, const GemmProblem& problem);

void gemm_fp32_cuda(const float* a, const float* b, float* c, const GemmProblem& problem,
                    GemmDispatch dispatch, cublasHandle_t cublas_handle = nullptr,
                    cudaStream_t stream = nullptr);

void gemm_fp16_cuda(const __half* a, const __half* b, float* c, const GemmProblem& problem,
                    GemmDispatch dispatch, cublasHandle_t cublas_handle = nullptr,
                    cudaStream_t stream = nullptr);

// Cost-transparent tensor-view dispatch. Inputs may be FP32 or FP16; output is
// always FP32. The selected backend and native handles remain caller-visible.
void gemm_cuda(const TensorView& a, const TensorView& b, TensorView& c, const GemmProblem& problem,
               GemmDispatch dispatch, cublasHandle_t cublas_handle = nullptr,
               cudaStream_t stream = nullptr);

} // namespace warpforge
