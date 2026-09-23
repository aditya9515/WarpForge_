#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

inline constexpr unsigned int memory_default_block_size = 256U;
inline constexpr unsigned int transpose_tile_dimension = 32U;
inline constexpr unsigned int transpose_block_rows = 8U;

[[nodiscard]] unsigned int memory_grid_size(std::size_t element_count,
                                            unsigned int block_size = memory_default_block_size);

void saxpy_cpu(float alpha, const float* input, float* inout, std::size_t element_count);

void saxpy_cuda(float alpha, const float* input, float* inout, std::size_t element_count,
                unsigned int block_size = memory_default_block_size, cudaStream_t stream = nullptr);

void memory_copy_cuda(const float* input, float* output, std::size_t element_count,
                      unsigned int block_size = memory_default_block_size,
                      cudaStream_t stream = nullptr);

void strided_copy_cpu(const float* input, float* output, std::size_t element_count,
                      std::size_t stride);

void strided_copy_cuda(const float* input, float* output, std::size_t element_count,
                       std::size_t stride, unsigned int block_size = memory_default_block_size,
                       cudaStream_t stream = nullptr);

void transpose_cpu(const float* input, float* output, std::size_t rows, std::size_t columns);

[[nodiscard]] dim3 transpose_grid_size(std::size_t rows, std::size_t columns, dim3 block);

void transpose_naive_cuda(const float* input, float* output, std::size_t rows, std::size_t columns,
                          dim3 block = dim3{32U, 8U, 1U}, cudaStream_t stream = nullptr);

void transpose_tiled_cuda(const float* input, float* output, std::size_t rows, std::size_t columns,
                          cudaStream_t stream = nullptr);

} // namespace warpforge
