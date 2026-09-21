#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

inline constexpr unsigned int fusion_default_block_size = 256U;

[[nodiscard]] std::size_t residual_rmsnorm_dynamic_shared_memory_bytes(
    unsigned int block_size = fusion_default_block_size);

void residual_rmsnorm_cpu(
    const float* input,
    const float* residual,
    const float* weight,
    float* output,
    std::size_t rows,
    std::size_t columns,
    float epsilon);

void residual_rmsnorm_fused_cuda(
    const float* input,
    const float* residual,
    const float* weight,
    float* output,
    std::size_t rows,
    std::size_t columns,
    float epsilon,
    unsigned int block_size = fusion_default_block_size,
    cudaStream_t stream = nullptr);

void swiglu_fused_cpu(
    const float* gate,
    const float* up,
    float* output,
    std::size_t element_count);

void swiglu_fused_cuda(
    const float* gate,
    const float* up,
    float* output,
    std::size_t element_count,
    unsigned int block_size = fusion_default_block_size,
    cudaStream_t stream = nullptr);

}  // namespace warpforge
