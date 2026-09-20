#include <warpforge/cuda_check.cuh>
#include <warpforge/vector_add.cuh>

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>

namespace warpforge {
namespace {

__global__ void vector_add_kernel(
    const float* left,
    const float* right,
    float* output,
    const std::size_t element_count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = left[index] + right[index];
    }
}

void validate_pointers(
    const float* left,
    const float* right,
    const float* output,
    const std::size_t element_count) {
    if (element_count > 0 && (left == nullptr || right == nullptr || output == nullptr)) {
        throw std::invalid_argument("VectorAdd pointers must not be null for a non-empty range");
    }
}

}  // namespace

void vector_add_cpu(
    const float* left,
    const float* right,
    float* output,
    const std::size_t element_count) {
    validate_pointers(left, right, output, element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        output[index] = left[index] + right[index];
    }
}

unsigned int vector_add_grid_size(
    const std::size_t element_count,
    const unsigned int block_size) {
    if (block_size == 0U || block_size > 1024U) {
        throw std::invalid_argument("VectorAdd block size must be in [1, 1024]");
    }
    if (element_count == 0) {
        return 0U;
    }

    const std::size_t blocks = 1U + (element_count - 1U) / block_size;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("VectorAdd grid exceeds the supported one-dimensional launch range");
    }
    return static_cast<unsigned int>(blocks);
}

void vector_add_cuda(
    const float* left,
    const float* right,
    float* output,
    const std::size_t element_count,
    const unsigned int block_size,
    cudaStream_t stream) {
    validate_pointers(left, right, output, element_count);
    const unsigned int grid_size = vector_add_grid_size(element_count, block_size);
    if (grid_size == 0U) {
        return;
    }

    vector_add_kernel<<<grid_size, block_size, 0, stream>>>(left, right, output, element_count);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace warpforge
