#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

enum class SoftmaxVariant {
    naive,
    block,
    warp,
};

inline constexpr unsigned int softmax_default_block_size = 256U;

[[nodiscard]] const char* softmax_variant_name(SoftmaxVariant variant) noexcept;
[[nodiscard]] std::size_t softmax_dynamic_shared_memory_bytes(
    SoftmaxVariant variant,
    unsigned int block_size = softmax_default_block_size);

void softmax_cpu(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t columns);

void softmax_cuda(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    SoftmaxVariant variant,
    unsigned int block_size = softmax_default_block_size,
    cudaStream_t stream = nullptr);

}  // namespace warpforge
