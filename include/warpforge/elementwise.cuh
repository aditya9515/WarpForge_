#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

inline constexpr unsigned int elementwise_default_block_size = 256U;

void silu_cpu(const float* input, float* output, std::size_t element_count);
void add_cpu(const float* left, const float* right, float* output, std::size_t element_count);
void multiply_cpu(const float* left, const float* right, float* output, std::size_t element_count);
void scale_cpu(const float* input, float scale, float* output, std::size_t element_count);
void swiglu_unfused_cpu(const float* gate, const float* up, float* intermediate, float* output,
                        std::size_t element_count);

void silu_cuda(const float* input, float* output, std::size_t element_count,
               unsigned int block_size = elementwise_default_block_size,
               cudaStream_t stream = nullptr);
void add_cuda(const float* left, const float* right, float* output, std::size_t element_count,
              unsigned int block_size = elementwise_default_block_size,
              cudaStream_t stream = nullptr);
void multiply_cuda(const float* left, const float* right, float* output, std::size_t element_count,
                   unsigned int block_size = elementwise_default_block_size,
                   cudaStream_t stream = nullptr);
void scale_cuda(const float* input, float scale, float* output, std::size_t element_count,
                unsigned int block_size = elementwise_default_block_size,
                cudaStream_t stream = nullptr);
void swiglu_unfused_cuda(const float* gate, const float* up, float* intermediate, float* output,
                         std::size_t element_count,
                         unsigned int block_size = elementwise_default_block_size,
                         cudaStream_t stream = nullptr);

} // namespace warpforge
