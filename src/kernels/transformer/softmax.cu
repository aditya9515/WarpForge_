#include <warpforge/cuda_check.cuh>
#include <warpforge/softmax.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace warpforge {
namespace {

constexpr unsigned int full_warp_mask = 0xffffffffU;

__device__ __forceinline__ float negative_infinity() {
    return -__int_as_float(0x7f800000);
}

void validate_block_size(const unsigned int block_size) {
    if (block_size < 32U || block_size > 1024U ||
        (block_size & (block_size - 1U)) != 0U) {
        throw std::invalid_argument(
            "softmax block size must be a power of two from 32 through 1024");
    }
}

[[nodiscard]] std::size_t checked_elements(
    const std::size_t rows,
    const std::size_t columns) {
    if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::overflow_error("softmax element count overflows size_t");
    }
    return rows * columns;
}

__device__ __forceinline__ float warp_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(full_warp_mask, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(full_warp_mask, value, offset);
    }
    return value;
}

__global__ void softmax_naive_kernel(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const std::size_t offset = row * columns;
    float maximum = negative_infinity();
    for (std::size_t column = 0U; column < columns; ++column) {
        maximum = fmaxf(maximum, input[offset + column]);
    }
    float sum = 0.0F;
    for (std::size_t column = 0U; column < columns; ++column) {
        sum += expf(input[offset + column] - maximum);
    }
    const float inverse = 1.0F / sum;
    for (std::size_t column = 0U; column < columns; ++column) {
        output[offset + column] = expf(input[offset + column] - maximum) * inverse;
    }
}

__global__ void softmax_block_kernel(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns) {
    extern __shared__ float scratch[];
    const std::size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int thread = threadIdx.x;
    const std::size_t offset = row * columns;
    float local_maximum = negative_infinity();
    for (std::size_t column = thread; column < columns; column += blockDim.x) {
        local_maximum = fmaxf(local_maximum, input[offset + column]);
    }
    scratch[thread] = local_maximum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (thread < stride) {
            scratch[thread] = fmaxf(scratch[thread], scratch[thread + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];
    float local_sum = 0.0F;
    for (std::size_t column = thread; column < columns; column += blockDim.x) {
        const float value = expf(input[offset + column] - maximum);
        output[offset + column] = value;
        local_sum += value;
    }
    scratch[thread] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (thread < stride) {
            scratch[thread] += scratch[thread + stride];
        }
        __syncthreads();
    }
    const float inverse = 1.0F / scratch[0];
    for (std::size_t column = thread; column < columns; column += blockDim.x) {
        output[offset + column] *= inverse;
    }
}

__global__ void softmax_warp_kernel(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns) {
    __shared__ float warp_values[32];
    __shared__ float row_maximum;
    __shared__ float row_sum;
    const std::size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int thread = threadIdx.x;
    const unsigned int lane = thread & 31U;
    const unsigned int warp = thread >> 5U;
    const unsigned int warp_count = blockDim.x >> 5U;
    const std::size_t offset = row * columns;

    float local_maximum = negative_infinity();
    for (std::size_t column = thread; column < columns; column += blockDim.x) {
        local_maximum = fmaxf(local_maximum, input[offset + column]);
    }
    local_maximum = warp_max(local_maximum);
    if (lane == 0U) {
        warp_values[warp] = local_maximum;
    }
    __syncthreads();
    if (warp == 0U) {
        float value = lane < warp_count ? warp_values[lane] : negative_infinity();
        value = warp_max(value);
        if (lane == 0U) {
            row_maximum = value;
        }
    }
    __syncthreads();

    float local_sum = 0.0F;
    for (std::size_t column = thread; column < columns; column += blockDim.x) {
        const float value = expf(input[offset + column] - row_maximum);
        output[offset + column] = value;
        local_sum += value;
    }
    local_sum = warp_sum(local_sum);
    if (lane == 0U) {
        warp_values[warp] = local_sum;
    }
    __syncthreads();
    if (warp == 0U) {
        float value = lane < warp_count ? warp_values[lane] : 0.0F;
        value = warp_sum(value);
        if (lane == 0U) {
            row_sum = value;
        }
    }
    __syncthreads();
    const float inverse = 1.0F / row_sum;
    for (std::size_t column = thread; column < columns; column += blockDim.x) {
        output[offset + column] *= inverse;
    }
}

}  // namespace

const char* softmax_variant_name(const SoftmaxVariant variant) noexcept {
    switch (variant) {
        case SoftmaxVariant::naive:
            return "naive";
        case SoftmaxVariant::block:
            return "block";
        case SoftmaxVariant::warp:
            return "warp";
    }
    return "unknown";
}

std::size_t softmax_dynamic_shared_memory_bytes(
    const SoftmaxVariant variant,
    const unsigned int block_size) {
    validate_block_size(block_size);
    return variant == SoftmaxVariant::block
        ? static_cast<std::size_t>(block_size) * sizeof(float)
        : 0U;
}

void softmax_cpu(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns) {
    const std::size_t elements = checked_elements(rows, columns);
    if (elements == 0U) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("non-empty softmax requires non-null input and output");
    }
    for (std::size_t row = 0U; row < rows; ++row) {
        const std::size_t offset = row * columns;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t column = 0U; column < columns; ++column) {
            maximum = std::max(maximum, input[offset + column]);
        }
        double sum = 0.0;
        for (std::size_t column = 0U; column < columns; ++column) {
            sum += std::exp(static_cast<double>(input[offset + column] - maximum));
        }
        for (std::size_t column = 0U; column < columns; ++column) {
            output[offset + column] = static_cast<float>(
                std::exp(static_cast<double>(input[offset + column] - maximum)) / sum);
        }
    }
}

void softmax_cuda(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    const SoftmaxVariant variant,
    const unsigned int block_size,
    const cudaStream_t stream) {
    validate_block_size(block_size);
    const std::size_t elements = checked_elements(rows, columns);
    if (elements == 0U) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("non-empty softmax requires non-null input and output");
    }
    if (rows > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("softmax row count exceeds the CUDA grid range");
    }
    if (variant == SoftmaxVariant::naive) {
        const auto grid = static_cast<unsigned int>((rows + block_size - 1U) / block_size);
        softmax_naive_kernel<<<grid, block_size, 0U, stream>>>(
            input, output, rows, columns);
    } else if (variant == SoftmaxVariant::block) {
        softmax_block_kernel<<<
            static_cast<unsigned int>(rows),
            block_size,
            softmax_dynamic_shared_memory_bytes(variant, block_size),
            stream>>>(input, output, rows, columns);
    } else if (variant == SoftmaxVariant::warp) {
        softmax_warp_kernel<<<static_cast<unsigned int>(rows), block_size, 0U, stream>>>(
            input, output, rows, columns);
    } else {
        throw std::invalid_argument("unknown softmax variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace warpforge
