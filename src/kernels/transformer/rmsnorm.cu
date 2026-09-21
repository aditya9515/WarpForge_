#include <warpforge/cuda_check.cuh>
#include <warpforge/rmsnorm.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace warpforge {
namespace {

void validate_block_size(const unsigned int block_size) {
    if (block_size < 32U || block_size > 1024U ||
        (block_size & (block_size - 1U)) != 0U) {
        throw std::invalid_argument(
            "RMSNorm block size must be a power of two from 32 through 1024");
    }
}

[[nodiscard]] std::size_t checked_elements(
    const std::size_t rows,
    const std::size_t columns) {
    if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::overflow_error("RMSNorm element count overflows size_t");
    }
    return rows * columns;
}

void validate_arguments(
    const float* input,
    const float* weight,
    const float* output,
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon) {
    const std::size_t elements = checked_elements(rows, columns);
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    }
    if (elements > 0U && (input == nullptr || weight == nullptr || output == nullptr)) {
        throw std::invalid_argument(
            "non-empty RMSNorm requires non-null input, weight, and output");
    }
    if (elements > 0U && output == weight) {
        throw std::invalid_argument("RMSNorm output must not alias its weight vector");
    }
}

__global__ void rmsnorm_naive_kernel(
    const float* input,
    const float* weight,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const std::size_t offset = row * columns;
    float square_sum = 0.0F;
    for (std::size_t column = 0U; column < columns; ++column) {
        const float value = input[offset + column];
        square_sum = fmaf(value, value, square_sum);
    }
    const float inverse_rms = rsqrtf(square_sum / static_cast<float>(columns) + epsilon);
    for (std::size_t column = 0U; column < columns; ++column) {
        output[offset + column] = input[offset + column] * inverse_rms * weight[column];
    }
}

__global__ void rmsnorm_block_kernel(
    const float* input,
    const float* weight,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon) {
    extern __shared__ float scratch[];
    const std::size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int thread = threadIdx.x;
    const std::size_t offset = row * columns;
    float local_square_sum = 0.0F;
    for (std::size_t column = thread; column < columns; column += blockDim.x) {
        const float value = input[offset + column];
        local_square_sum = fmaf(value, value, local_square_sum);
    }
    scratch[thread] = local_square_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (thread < stride) {
            scratch[thread] += scratch[thread + stride];
        }
        __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(scratch[0] / static_cast<float>(columns) + epsilon);
    for (std::size_t column = thread; column < columns; column += blockDim.x) {
        output[offset + column] = input[offset + column] * inverse_rms * weight[column];
    }
}

}  // namespace

const char* rmsnorm_variant_name(const RmsNormVariant variant) noexcept {
    switch (variant) {
        case RmsNormVariant::naive:
            return "naive";
        case RmsNormVariant::block:
            return "block";
    }
    return "unknown";
}

Tolerance rmsnorm_tolerance(const std::size_t width) noexcept {
    const double scale = std::max(
        1.0,
        std::ceil(std::log2(static_cast<double>(std::max<std::size_t>(1U, width)))));
    return Tolerance{1.0e-5 * scale, 1.0e-5};
}

std::size_t rmsnorm_dynamic_shared_memory_bytes(
    const RmsNormVariant variant,
    const unsigned int block_size) {
    validate_block_size(block_size);
    return variant == RmsNormVariant::block
        ? static_cast<std::size_t>(block_size) * sizeof(float)
        : 0U;
}

void rmsnorm_cpu(
    const float* input,
    const float* weight,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon) {
    validate_arguments(input, weight, output, rows, columns, epsilon);
    if (rows == 0U || columns == 0U) {
        return;
    }
    for (std::size_t row = 0U; row < rows; ++row) {
        const std::size_t offset = row * columns;
        double square_sum = 0.0;
        for (std::size_t column = 0U; column < columns; ++column) {
            const double value = static_cast<double>(input[offset + column]);
            square_sum += value * value;
        }
        const double inverse_rms =
            1.0 / std::sqrt(square_sum / static_cast<double>(columns) + epsilon);
        for (std::size_t column = 0U; column < columns; ++column) {
            output[offset + column] = static_cast<float>(
                static_cast<double>(input[offset + column]) * inverse_rms *
                static_cast<double>(weight[column]));
        }
    }
}

void rmsnorm_cuda(
    const float* input,
    const float* weight,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon,
    const RmsNormVariant variant,
    const unsigned int block_size,
    const cudaStream_t stream) {
    validate_block_size(block_size);
    validate_arguments(input, weight, output, rows, columns, epsilon);
    if (rows == 0U || columns == 0U) {
        return;
    }
    if (rows > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("RMSNorm row count exceeds the CUDA grid range");
    }
    if (variant == RmsNormVariant::naive) {
        const auto grid = static_cast<unsigned int>((rows + block_size - 1U) / block_size);
        rmsnorm_naive_kernel<<<grid, block_size, 0U, stream>>>(
            input, weight, output, rows, columns, epsilon);
    } else if (variant == RmsNormVariant::block) {
        rmsnorm_block_kernel<<<
            static_cast<unsigned int>(rows),
            block_size,
            rmsnorm_dynamic_shared_memory_bytes(variant, block_size),
            stream>>>(input, weight, output, rows, columns, epsilon);
    } else {
        throw std::invalid_argument("unknown RMSNorm variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace warpforge
