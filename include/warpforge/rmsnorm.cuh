#pragma once

#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

enum class RmsNormVariant {
    naive,
    block,
};

inline constexpr unsigned int rmsnorm_default_block_size = 256U;

[[nodiscard]] const char* rmsnorm_variant_name(RmsNormVariant variant) noexcept;
[[nodiscard]] Tolerance rmsnorm_tolerance(std::size_t width) noexcept;
[[nodiscard]] std::size_t rmsnorm_dynamic_shared_memory_bytes(
    RmsNormVariant variant,
    unsigned int block_size = rmsnorm_default_block_size);

void rmsnorm_cpu(
    const float* input,
    const float* weight,
    float* output,
    std::size_t rows,
    std::size_t columns,
    float epsilon);

void rmsnorm_cuda(
    const float* input,
    const float* weight,
    float* output,
    std::size_t rows,
    std::size_t columns,
    float epsilon,
    RmsNormVariant variant,
    unsigned int block_size = rmsnorm_default_block_size,
    cudaStream_t stream = nullptr);

}  // namespace warpforge
