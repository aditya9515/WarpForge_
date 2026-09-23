#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

inline constexpr unsigned int vector_add_default_block_size = 256U;

void vector_add_cpu(const float* left, const float* right, float* output,
                    std::size_t element_count);

[[nodiscard]] unsigned int
vector_add_grid_size(std::size_t element_count,
                     unsigned int block_size = vector_add_default_block_size);

void vector_add_cuda(const float* left, const float* right, float* output,
                     std::size_t element_count,
                     unsigned int block_size = vector_add_default_block_size,
                     cudaStream_t stream = nullptr);

} // namespace warpforge
