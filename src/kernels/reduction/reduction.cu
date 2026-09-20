#include <warpforge/cuda_check.cuh>
#include <warpforge/reduction.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace warpforge {
namespace {

template <ReductionOperation Operation>
__device__ __forceinline__ float identity_value();

template <>
__device__ __forceinline__ float identity_value<ReductionOperation::sum>() {
    return 0.0F;
}

template <>
__device__ __forceinline__ float identity_value<ReductionOperation::maximum>() {
    return -__int_as_float(0x7f800000);
}

template <ReductionOperation Operation>
__device__ __forceinline__ float combine_values(const float left, const float right);

template <>
__device__ __forceinline__ float combine_values<ReductionOperation::sum>(
    const float left,
    const float right) {
    return left + right;
}

template <>
__device__ __forceinline__ float combine_values<ReductionOperation::maximum>(
    const float left,
    const float right) {
    return fmaxf(left, right);
}

template <ReductionOperation Operation, unsigned int BlockSize>
__global__ void interleaved_kernel(
    const float* input,
    float* output,
    const std::size_t element_count) {
    extern __shared__ float shared[];
    const unsigned int thread = threadIdx.x;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * BlockSize + thread;
    shared[thread] =
        index < element_count ? input[index] : identity_value<Operation>();
    __syncthreads();

    for (unsigned int stride = 1U; stride < BlockSize; stride <<= 1U) {
        const unsigned int shared_index = 2U * stride * thread;
        if (shared_index < BlockSize && shared_index + stride < BlockSize) {
            shared[shared_index] = combine_values<Operation>(
                shared[shared_index], shared[shared_index + stride]);
        }
        __syncthreads();
    }
    if (thread == 0U) {
        output[blockIdx.x] = shared[0];
    }
}

template <ReductionOperation Operation, unsigned int BlockSize>
__global__ void shared_memory_kernel(
    const float* input,
    float* output,
    const std::size_t element_count) {
    extern __shared__ float shared[];
    const unsigned int thread = threadIdx.x;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * BlockSize + thread;
    shared[thread] =
        index < element_count ? input[index] : identity_value<Operation>();
    __syncthreads();

    for (unsigned int stride = BlockSize / 2U; stride > 0U; stride >>= 1U) {
        if (thread < stride) {
            shared[thread] =
                combine_values<Operation>(shared[thread], shared[thread + stride]);
        }
        __syncthreads();
    }
    if (thread == 0U) {
        output[blockIdx.x] = shared[0];
    }
}

template <ReductionOperation Operation, unsigned int BlockSize>
__global__ void reduced_divergence_kernel(
    const float* input,
    float* output,
    const std::size_t element_count) {
    extern __shared__ float shared[];
    const unsigned int thread = threadIdx.x;
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * (2U * BlockSize) + thread;
    float value = first < element_count ? input[first] : identity_value<Operation>();
    const std::size_t second = first + BlockSize;
    if (second < element_count) {
        value = combine_values<Operation>(value, input[second]);
    }
    shared[thread] = value;
    __syncthreads();

    for (unsigned int stride = BlockSize / 2U; stride > 0U; stride >>= 1U) {
        if (thread < stride) {
            shared[thread] =
                combine_values<Operation>(shared[thread], shared[thread + stride]);
        }
        __syncthreads();
    }
    if (thread == 0U) {
        output[blockIdx.x] = shared[0];
    }
}

template <ReductionOperation Operation, unsigned int BlockSize>
__global__ void unrolled_kernel(
    const float* input,
    float* output,
    const std::size_t element_count) {
    extern __shared__ float shared[];
    const unsigned int thread = threadIdx.x;
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * (2U * BlockSize) + thread;
    float value = first < element_count ? input[first] : identity_value<Operation>();
    const std::size_t second = first + BlockSize;
    if (second < element_count) {
        value = combine_values<Operation>(value, input[second]);
    }
    shared[thread] = value;
    __syncthreads();

    if constexpr (BlockSize >= 1024U) {
        if (thread < 512U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 512U]);
        }
        __syncthreads();
    }
    if constexpr (BlockSize >= 512U) {
        if (thread < 256U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 256U]);
        }
        __syncthreads();
    }
    if constexpr (BlockSize >= 256U) {
        if (thread < 128U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 128U]);
        }
        __syncthreads();
    }
    if constexpr (BlockSize >= 128U) {
        if (thread < 64U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 64U]);
        }
        __syncthreads();
    }
    if constexpr (BlockSize >= 64U) {
        if (thread < 32U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 32U]);
        }
        __syncthreads();
    }

    if (thread < 32U) {
        if (thread < 16U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 16U]);
        }
        __syncwarp();
        if (thread < 8U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 8U]);
        }
        __syncwarp();
        if (thread < 4U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 4U]);
        }
        __syncwarp();
        if (thread < 2U) {
            shared[thread] = combine_values<Operation>(shared[thread], shared[thread + 2U]);
        }
        __syncwarp();
        if (thread == 0U) {
            output[blockIdx.x] = combine_values<Operation>(shared[0], shared[1]);
        }
    }
}

template <ReductionOperation Operation, unsigned int BlockSize>
__global__ void warp_shuffle_kernel(
    const float* input,
    float* output,
    const std::size_t element_count) {
    extern __shared__ float warp_values[];
    const unsigned int thread = threadIdx.x;
    const unsigned int lane = thread & 31U;
    const unsigned int warp = thread >> 5U;
    constexpr unsigned int warp_count = BlockSize / 32U;
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * (2U * BlockSize) + thread;
    float value = first < element_count ? input[first] : identity_value<Operation>();
    const std::size_t second = first + BlockSize;
    if (second < element_count) {
        value = combine_values<Operation>(value, input[second]);
    }

    constexpr unsigned int full_mask = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = combine_values<Operation>(
            value, __shfl_down_sync(full_mask, value, offset));
    }
    if (lane == 0U) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0U) {
        value = lane < warp_count ? warp_values[lane] : identity_value<Operation>();
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = combine_values<Operation>(
                value, __shfl_down_sync(full_mask, value, offset));
        }
        if (lane == 0U) {
            output[blockIdx.x] = value;
        }
    }
}

bool is_supported_block_size(const unsigned int block_size) {
    switch (block_size) {
        case 32U:
        case 64U:
        case 128U:
        case 256U:
        case 512U:
        case 1024U:
            return true;
        default:
            return false;
    }
}

void validate_block_size(const unsigned int block_size) {
    if (!is_supported_block_size(block_size)) {
        throw std::invalid_argument(
            "reduction block size must be a power of two from 32 through 1024");
    }
}

template <ReductionOperation Operation, unsigned int BlockSize>
void launch_block(
    const ReductionVariant variant,
    const float* input,
    float* output,
    const std::size_t element_count,
    const unsigned int grid_size,
    cudaStream_t stream) {
    const std::size_t shared_bytes = reduction_dynamic_shared_memory_bytes(variant, BlockSize);
    switch (variant) {
        case ReductionVariant::naive_interleaved:
            interleaved_kernel<Operation, BlockSize>
                <<<grid_size, BlockSize, shared_bytes, stream>>>(input, output, element_count);
            break;
        case ReductionVariant::shared_memory:
            shared_memory_kernel<Operation, BlockSize>
                <<<grid_size, BlockSize, shared_bytes, stream>>>(input, output, element_count);
            break;
        case ReductionVariant::reduced_divergence:
            reduced_divergence_kernel<Operation, BlockSize>
                <<<grid_size, BlockSize, shared_bytes, stream>>>(input, output, element_count);
            break;
        case ReductionVariant::unrolled:
            unrolled_kernel<Operation, BlockSize>
                <<<grid_size, BlockSize, shared_bytes, stream>>>(input, output, element_count);
            break;
        case ReductionVariant::warp_shuffle:
            warp_shuffle_kernel<Operation, BlockSize>
                <<<grid_size, BlockSize, shared_bytes, stream>>>(input, output, element_count);
            break;
        default:
            throw std::invalid_argument("unknown reduction variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <ReductionOperation Operation>
void launch_pass(
    const ReductionVariant variant,
    const unsigned int block_size,
    const float* input,
    float* output,
    const std::size_t element_count,
    const unsigned int grid_size,
    cudaStream_t stream) {
    switch (block_size) {
        case 32U:
            launch_block<Operation, 32U>(
                variant, input, output, element_count, grid_size, stream);
            break;
        case 64U:
            launch_block<Operation, 64U>(
                variant, input, output, element_count, grid_size, stream);
            break;
        case 128U:
            launch_block<Operation, 128U>(
                variant, input, output, element_count, grid_size, stream);
            break;
        case 256U:
            launch_block<Operation, 256U>(
                variant, input, output, element_count, grid_size, stream);
            break;
        case 512U:
            launch_block<Operation, 512U>(
                variant, input, output, element_count, grid_size, stream);
            break;
        case 1024U:
            launch_block<Operation, 1024U>(
                variant, input, output, element_count, grid_size, stream);
            break;
        default:
            throw std::invalid_argument("unsupported reduction block size");
    }
}

}  // namespace

const char* reduction_operation_name(const ReductionOperation operation) noexcept {
    switch (operation) {
        case ReductionOperation::sum:
            return "sum";
        case ReductionOperation::maximum:
            return "maximum";
        default:
            return "unknown";
    }
}

const char* reduction_variant_name(const ReductionVariant variant) noexcept {
    switch (variant) {
        case ReductionVariant::naive_interleaved:
            return "naive_interleaved";
        case ReductionVariant::shared_memory:
            return "shared_memory";
        case ReductionVariant::reduced_divergence:
            return "reduced_divergence";
        case ReductionVariant::unrolled:
            return "unrolled";
        case ReductionVariant::warp_shuffle:
            return "warp_shuffle";
        default:
            return "unknown";
    }
}

std::size_t reduction_elements_per_block(
    const ReductionVariant variant,
    const unsigned int block_size) {
    validate_block_size(block_size);
    switch (variant) {
        case ReductionVariant::naive_interleaved:
        case ReductionVariant::shared_memory:
            return block_size;
        case ReductionVariant::reduced_divergence:
        case ReductionVariant::unrolled:
        case ReductionVariant::warp_shuffle:
            return 2U * static_cast<std::size_t>(block_size);
        default:
            throw std::invalid_argument("unknown reduction variant");
    }
}

std::size_t reduction_output_count(
    const std::size_t input_count,
    const ReductionVariant variant,
    const unsigned int block_size) {
    if (input_count == 0) {
        return 0;
    }
    const std::size_t elements_per_block =
        reduction_elements_per_block(variant, block_size);
    return 1U + (input_count - 1U) / elements_per_block;
}

std::size_t reduction_workspace_elements(
    const std::size_t input_count,
    const ReductionVariant variant,
    const unsigned int block_size) {
    const std::size_t first_output =
        reduction_output_count(input_count, variant, block_size);
    return first_output > 1U ? first_output : 0U;
}

std::size_t reduction_pass_count(
    const std::size_t input_count,
    const ReductionVariant variant,
    const unsigned int block_size) {
    if (input_count == 0) {
        return 0;
    }
    std::size_t passes = 0;
    std::size_t current = input_count;
    while (current > 1U) {
        current = reduction_output_count(current, variant, block_size);
        ++passes;
    }
    return passes;
}

std::size_t reduction_dynamic_shared_memory_bytes(
    const ReductionVariant variant,
    const unsigned int block_size) {
    validate_block_size(block_size);
    if (variant == ReductionVariant::warp_shuffle) {
        return static_cast<std::size_t>(block_size / 32U) * sizeof(float);
    }
    return static_cast<std::size_t>(block_size) * sizeof(float);
}

double reduction_sum_cpu(const float* input, const std::size_t element_count) {
    if (element_count > 0 && input == nullptr) {
        throw std::invalid_argument("sum input must not be null for non-empty input");
    }
    double sum = 0.0;
    for (std::size_t index = 0; index < element_count; ++index) {
        sum += static_cast<double>(input[index]);
    }
    return sum;
}

float reduction_max_cpu(const float* input, const std::size_t element_count) {
    if (element_count == 0) {
        throw std::invalid_argument("maximum reduction requires at least one element");
    }
    if (input == nullptr) {
        throw std::invalid_argument("maximum input must not be null");
    }
    float maximum = input[0];
    for (std::size_t index = 1; index < element_count; ++index) {
        maximum = std::max(maximum, input[index]);
    }
    return maximum;
}

Tolerance reduction_sum_tolerance(const std::size_t element_count) {
    const double levels = element_count > 1U
        ? std::ceil(std::log2(static_cast<double>(element_count)))
        : 0.0;
    return {1.0e-5 * (1.0 + levels), 2.0e-5};
}

ValidationResult validate_reduction_sum(
    const double expected,
    const float actual,
    const std::size_t input_count) {
    if (!std::isfinite(expected) || !std::isfinite(static_cast<double>(actual))) {
        throw std::invalid_argument("sum validation requires finite values");
    }
    const Tolerance tolerance = reduction_sum_tolerance(input_count);
    const double absolute_error = std::abs(static_cast<double>(actual) - expected);
    const double relative_error = expected == 0.0
        ? (absolute_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
        : absolute_error / std::abs(expected);
    const double allowed_error =
        tolerance.absolute + tolerance.relative * std::abs(expected);

    ValidationResult result;
    result.passed = absolute_error <= allowed_error;
    result.element_count = 1U;
    result.failure_count = result.passed ? 0U : 1U;
    result.max_absolute_error = absolute_error;
    result.mean_absolute_error = absolute_error;
    result.max_relative_error = relative_error;
    result.expected_at_worst = static_cast<float>(expected);
    result.actual_at_worst = actual;
    return result;
}

void reduce_cuda(
    const float* input,
    const std::size_t element_count,
    const ReductionOperation operation,
    const ReductionVariant variant,
    float* workspace_a,
    float* workspace_b,
    const std::size_t workspace_capacity,
    float* output,
    const unsigned int block_size,
    cudaStream_t stream) {
    validate_block_size(block_size);
    if (element_count == 0) {
        throw std::invalid_argument("GPU reduction requires at least one element");
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("GPU reduction input and output must not be null");
    }

    const std::size_t required_workspace =
        reduction_workspace_elements(element_count, variant, block_size);
    if (workspace_capacity < required_workspace) {
        throw std::invalid_argument("reduction workspace is smaller than the required capacity");
    }
    if (required_workspace > 0U && (workspace_a == nullptr || workspace_b == nullptr)) {
        throw std::invalid_argument("reduction workspace pointers must not be null");
    }

    if (element_count == 1U) {
        CUDA_CHECK(cudaMemcpyAsync(
            output, input, sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const float* current_input = input;
    std::size_t current_count = element_count;
    bool use_workspace_a = true;
    while (current_count > 1U) {
        const std::size_t output_count =
            reduction_output_count(current_count, variant, block_size);
        if (output_count > std::numeric_limits<unsigned int>::max()) {
            throw std::overflow_error("reduction grid exceeds the CUDA x-dimension range");
        }
        float* current_output = output_count == 1U
            ? output
            : (use_workspace_a ? workspace_a : workspace_b);
        const unsigned int grid_size = static_cast<unsigned int>(output_count);
        switch (operation) {
            case ReductionOperation::sum:
                launch_pass<ReductionOperation::sum>(
                    variant,
                    block_size,
                    current_input,
                    current_output,
                    current_count,
                    grid_size,
                    stream);
                break;
            case ReductionOperation::maximum:
                launch_pass<ReductionOperation::maximum>(
                    variant,
                    block_size,
                    current_input,
                    current_output,
                    current_count,
                    grid_size,
                    stream);
                break;
            default:
                throw std::invalid_argument("unknown reduction operation");
        }
        current_input = current_output;
        current_count = output_count;
        use_workspace_a = !use_workspace_a;
    }
}

}  // namespace warpforge
