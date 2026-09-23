#include <warpforge/cuda_check.cuh>
#include <warpforge/elementwise.cuh>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace warpforge {
namespace {

void validate_block_size(const unsigned int block_size) {
    if (block_size == 0U || block_size > 1024U) {
        throw std::invalid_argument("elementwise block size must be from 1 through 1024");
    }
}

void validate_unary(const float* input, const float* output, const std::size_t element_count) {
    if (element_count > 0U && (input == nullptr || output == nullptr)) {
        throw std::invalid_argument(
            "non-empty elementwise operation requires non-null input and output");
    }
}

void validate_binary(const float* left, const float* right, const float* output,
                     const std::size_t element_count) {
    if (element_count > 0U && (left == nullptr || right == nullptr || output == nullptr)) {
        throw std::invalid_argument(
            "non-empty binary operation requires non-null inputs and output");
    }
}

[[nodiscard]] unsigned int grid_size(const std::size_t element_count,
                                     const unsigned int block_size) {
    const std::size_t grid = (element_count + block_size - 1U) / block_size;
    if (grid > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("elementwise grid exceeds the CUDA x-dimension range");
    }
    return static_cast<unsigned int>(grid);
}

__device__ __forceinline__ float silu_value(const float value) {
    return value / (1.0F + expf(-value));
}

__global__ void silu_kernel(const float* input, float* output, const std::size_t element_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = silu_value(input[index]);
    }
}

__global__ void add_kernel(const float* left, const float* right, float* output,
                           const std::size_t element_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = left[index] + right[index];
    }
}

__global__ void multiply_kernel(const float* left, const float* right, float* output,
                                const std::size_t element_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = left[index] * right[index];
    }
}

__global__ void scale_kernel(const float* input, const float scale, float* output,
                             const std::size_t element_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = input[index] * scale;
    }
}

} // namespace

void silu_cpu(const float* input, float* output, const std::size_t element_count) {
    validate_unary(input, output, element_count);
    for (std::size_t index = 0U; index < element_count; ++index) {
        const double value = input[index];
        output[index] = static_cast<float>(value / (1.0 + std::exp(-value)));
    }
}

void add_cpu(const float* left, const float* right, float* output,
             const std::size_t element_count) {
    validate_binary(left, right, output, element_count);
    for (std::size_t index = 0U; index < element_count; ++index) {
        output[index] = left[index] + right[index];
    }
}

void multiply_cpu(const float* left, const float* right, float* output,
                  const std::size_t element_count) {
    validate_binary(left, right, output, element_count);
    for (std::size_t index = 0U; index < element_count; ++index) {
        output[index] = left[index] * right[index];
    }
}

void scale_cpu(const float* input, const float scale, float* output,
               const std::size_t element_count) {
    validate_unary(input, output, element_count);
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("elementwise scale must be finite");
    }
    for (std::size_t index = 0U; index < element_count; ++index) {
        output[index] = input[index] * scale;
    }
}

void swiglu_unfused_cpu(const float* gate, const float* up, float* intermediate, float* output,
                        const std::size_t element_count) {
    validate_binary(gate, up, output, element_count);
    if (element_count > 0U && intermediate == nullptr) {
        throw std::invalid_argument("non-empty SwiGLU requires an intermediate buffer");
    }
    if (element_count > 0U &&
        (intermediate == gate || intermediate == up || intermediate == output)) {
        throw std::invalid_argument("unfused SwiGLU intermediate must not alias inputs or output");
    }
    silu_cpu(gate, intermediate, element_count);
    multiply_cpu(intermediate, up, output, element_count);
}

void silu_cuda(const float* input, float* output, const std::size_t element_count,
               const unsigned int block_size, const cudaStream_t stream) {
    validate_block_size(block_size);
    validate_unary(input, output, element_count);
    if (element_count == 0U) {
        return;
    }
    silu_kernel<<<grid_size(element_count, block_size), block_size, 0U, stream>>>(input, output,
                                                                                  element_count);
    CUDA_CHECK(cudaGetLastError());
}

void add_cuda(const float* left, const float* right, float* output, const std::size_t element_count,
              const unsigned int block_size, const cudaStream_t stream) {
    validate_block_size(block_size);
    validate_binary(left, right, output, element_count);
    if (element_count == 0U) {
        return;
    }
    add_kernel<<<grid_size(element_count, block_size), block_size, 0U, stream>>>(
        left, right, output, element_count);
    CUDA_CHECK(cudaGetLastError());
}

void multiply_cuda(const float* left, const float* right, float* output,
                   const std::size_t element_count, const unsigned int block_size,
                   const cudaStream_t stream) {
    validate_block_size(block_size);
    validate_binary(left, right, output, element_count);
    if (element_count == 0U) {
        return;
    }
    multiply_kernel<<<grid_size(element_count, block_size), block_size, 0U, stream>>>(
        left, right, output, element_count);
    CUDA_CHECK(cudaGetLastError());
}

void scale_cuda(const float* input, const float scale, float* output,
                const std::size_t element_count, const unsigned int block_size,
                const cudaStream_t stream) {
    validate_block_size(block_size);
    validate_unary(input, output, element_count);
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("elementwise scale must be finite");
    }
    if (element_count == 0U) {
        return;
    }
    scale_kernel<<<grid_size(element_count, block_size), block_size, 0U, stream>>>(
        input, scale, output, element_count);
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_unfused_cuda(const float* gate, const float* up, float* intermediate, float* output,
                         const std::size_t element_count, const unsigned int block_size,
                         const cudaStream_t stream) {
    validate_binary(gate, up, output, element_count);
    if (element_count > 0U && intermediate == nullptr) {
        throw std::invalid_argument("non-empty SwiGLU requires an intermediate buffer");
    }
    if (element_count > 0U &&
        (intermediate == gate || intermediate == up || intermediate == output)) {
        throw std::invalid_argument("unfused SwiGLU intermediate must not alias inputs or output");
    }
    silu_cuda(gate, intermediate, element_count, block_size, stream);
    multiply_cuda(intermediate, up, output, element_count, block_size, stream);
}

} // namespace warpforge
