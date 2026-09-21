#include <warpforge/cuda_check.cuh>
#include <warpforge/fusion.cuh>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace warpforge {
namespace {

void validate_block_size(const unsigned int block_size) {
    if (block_size < 32U || block_size > 1024U ||
        (block_size & (block_size - 1U)) != 0U) {
        throw std::invalid_argument(
            "fusion block size must be a power of two from 32 through 1024");
    }
}

[[nodiscard]] std::size_t checked_elements(
    const std::size_t rows,
    const std::size_t columns) {
    if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::overflow_error("residual RMSNorm element count overflows size_t");
    }
    return rows * columns;
}

void validate_residual_rmsnorm(
    const float* input,
    const float* residual,
    const float* weight,
    const float* output,
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon) {
    const std::size_t elements = checked_elements(rows, columns);
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument(
            "residual RMSNorm epsilon must be finite and positive");
    }
    if (elements > 0U &&
        (input == nullptr || residual == nullptr || weight == nullptr || output == nullptr)) {
        throw std::invalid_argument(
            "non-empty residual RMSNorm requires non-null pointers");
    }
    if (elements > 0U && output == weight) {
        throw std::invalid_argument(
            "residual RMSNorm output must not alias its weight vector");
    }
}

void validate_swiglu(
    const float* gate,
    const float* up,
    const float* output,
    const std::size_t element_count) {
    if (element_count > 0U &&
        (gate == nullptr || up == nullptr || output == nullptr)) {
        throw std::invalid_argument("non-empty fused SwiGLU requires non-null pointers");
    }
}

[[nodiscard]] unsigned int elementwise_grid_size(
    const std::size_t element_count,
    const unsigned int block_size) {
    const std::size_t grid = (element_count + block_size - 1U) / block_size;
    if (grid > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("fused SwiGLU grid exceeds the CUDA x-dimension range");
    }
    return static_cast<unsigned int>(grid);
}

__global__ void residual_rmsnorm_fused_kernel(
    const float* input,
    const float* residual,
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
        const float value = input[offset + column] + residual[offset + column];
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
        const float value = input[offset + column] + residual[offset + column];
        output[offset + column] = value * inverse_rms * weight[column];
    }
}

__global__ void swiglu_fused_kernel(
    const float* gate,
    const float* up,
    float* output,
    const std::size_t element_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        const float gate_value = gate[index];
        const float silu = gate_value / (1.0F + expf(-gate_value));
        output[index] = silu * up[index];
    }
}

}  // namespace

std::size_t residual_rmsnorm_dynamic_shared_memory_bytes(
    const unsigned int block_size) {
    validate_block_size(block_size);
    return static_cast<std::size_t>(block_size) * sizeof(float);
}

void residual_rmsnorm_cpu(
    const float* input,
    const float* residual,
    const float* weight,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon) {
    validate_residual_rmsnorm(
        input, residual, weight, output, rows, columns, epsilon);
    if (rows == 0U || columns == 0U) {
        return;
    }
    for (std::size_t row = 0U; row < rows; ++row) {
        const std::size_t offset = row * columns;
        double square_sum = 0.0;
        for (std::size_t column = 0U; column < columns; ++column) {
            const double value = static_cast<double>(input[offset + column]) +
                                 static_cast<double>(residual[offset + column]);
            square_sum += value * value;
        }
        const double inverse_rms =
            1.0 / std::sqrt(square_sum / static_cast<double>(columns) + epsilon);
        for (std::size_t column = 0U; column < columns; ++column) {
            const double value = static_cast<double>(input[offset + column]) +
                                 static_cast<double>(residual[offset + column]);
            output[offset + column] = static_cast<float>(
                value * inverse_rms * static_cast<double>(weight[column]));
        }
    }
}

void residual_rmsnorm_fused_cuda(
    const float* input,
    const float* residual,
    const float* weight,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon,
    const unsigned int block_size,
    const cudaStream_t stream) {
    validate_block_size(block_size);
    validate_residual_rmsnorm(
        input, residual, weight, output, rows, columns, epsilon);
    if (rows == 0U || columns == 0U) {
        return;
    }
    if (rows > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error(
            "residual RMSNorm row count exceeds the CUDA grid range");
    }
    residual_rmsnorm_fused_kernel<<<
        static_cast<unsigned int>(rows),
        block_size,
        residual_rmsnorm_dynamic_shared_memory_bytes(block_size),
        stream>>>(input, residual, weight, output, rows, columns, epsilon);
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_fused_cpu(
    const float* gate,
    const float* up,
    float* output,
    const std::size_t element_count) {
    validate_swiglu(gate, up, output, element_count);
    for (std::size_t index = 0U; index < element_count; ++index) {
        const double gate_value = gate[index];
        const double silu = gate_value / (1.0 + std::exp(-gate_value));
        output[index] = static_cast<float>(silu * static_cast<double>(up[index]));
    }
}

void swiglu_fused_cuda(
    const float* gate,
    const float* up,
    float* output,
    const std::size_t element_count,
    const unsigned int block_size,
    const cudaStream_t stream) {
    validate_block_size(block_size);
    validate_swiglu(gate, up, output, element_count);
    if (element_count == 0U) {
        return;
    }
    swiglu_fused_kernel<<<
        elementwise_grid_size(element_count, block_size), block_size, 0U, stream>>>(
        gate, up, output, element_count);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace warpforge
