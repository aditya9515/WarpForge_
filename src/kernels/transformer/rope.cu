#include <warpforge/cuda_check.cuh>
#include <warpforge/rope.cuh>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace warpforge {
namespace {

[[nodiscard]] std::size_t checked_product(const std::size_t left, const std::size_t right,
                                          const char* label) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return left * right;
}

void validate_problem(const RopeProblem& problem) {
    if (!std::isfinite(problem.base) || problem.base <= 0.0F) {
        throw std::invalid_argument("RoPE base must be finite and positive");
    }
    if (problem.head_dimension % 2U != 0U) {
        throw std::invalid_argument("RoPE head dimension must be even");
    }
    if (problem.sequence > 0U && problem.position_offset > std::numeric_limits<std::size_t>::max() -
                                                               (problem.sequence - 1U)) {
        throw std::overflow_error("RoPE position range overflows size_t");
    }
    static_cast<void>(rope_element_count(problem));
}

__global__ void rope_kernel(const float* input, float* output, const std::size_t pair_count,
                            const std::size_t sequence, const std::size_t heads,
                            const std::size_t head_dimension, const std::size_t position_offset,
                            const float base) {
    const std::size_t pair_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_index >= pair_count) {
        return;
    }
    const std::size_t pairs_per_head = head_dimension / 2U;
    const std::size_t pair = pair_index % pairs_per_head;
    const std::size_t head_group = pair_index / pairs_per_head;
    const std::size_t token = head_group / heads;
    const std::size_t sequence_index = token % sequence;
    const std::size_t element = head_group * head_dimension + pair * 2U;
    const float exponent = -2.0F * static_cast<float>(pair) / static_cast<float>(head_dimension);
    const float angle = static_cast<float>(position_offset + sequence_index) * powf(base, exponent);
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float first = input[element];
    const float second = input[element + 1U];
    output[element] = first * cosine - second * sine;
    output[element + 1U] = first * sine + second * cosine;
}

} // namespace

std::size_t rope_element_count(const RopeProblem& problem) {
    const std::size_t tokens = checked_product(problem.batch, problem.sequence, "RoPE token count");
    const std::size_t head_groups = checked_product(tokens, problem.heads, "RoPE head count");
    return checked_product(head_groups, problem.head_dimension, "RoPE element count");
}

std::size_t rope_pair_count(const RopeProblem& problem) {
    if (problem.head_dimension % 2U != 0U) {
        throw std::invalid_argument("RoPE head dimension must be even");
    }
    return rope_element_count(problem) / 2U;
}

void rope_cpu(const float* input, float* output, const RopeProblem& problem) {
    validate_problem(problem);
    const std::size_t pair_count = rope_pair_count(problem);
    if (pair_count == 0U) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("non-empty RoPE requires non-null input and output");
    }
    const std::size_t pairs_per_head = problem.head_dimension / 2U;
    for (std::size_t pair_index = 0U; pair_index < pair_count; ++pair_index) {
        const std::size_t pair = pair_index % pairs_per_head;
        const std::size_t head_group = pair_index / pairs_per_head;
        const std::size_t token = head_group / problem.heads;
        const std::size_t sequence_index = token % problem.sequence;
        const std::size_t element = head_group * problem.head_dimension + pair * 2U;
        const double exponent =
            -2.0 * static_cast<double>(pair) / static_cast<double>(problem.head_dimension);
        const double angle = static_cast<double>(problem.position_offset + sequence_index) *
                             std::pow(static_cast<double>(problem.base), exponent);
        const double sine = std::sin(angle);
        const double cosine = std::cos(angle);
        const double first = input[element];
        const double second = input[element + 1U];
        output[element] = static_cast<float>(first * cosine - second * sine);
        output[element + 1U] = static_cast<float>(first * sine + second * cosine);
    }
}

void rope_cuda(const float* input, float* output, const RopeProblem& problem,
               const unsigned int block_size, const cudaStream_t stream) {
    validate_problem(problem);
    if (block_size == 0U || block_size > 1024U) {
        throw std::invalid_argument("RoPE block size must be from 1 through 1024");
    }
    const std::size_t pair_count = rope_pair_count(problem);
    if (pair_count == 0U) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("non-empty RoPE requires non-null input and output");
    }
    const std::size_t grid = (pair_count + block_size - 1U) / block_size;
    if (grid > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("RoPE grid exceeds the CUDA x-dimension range");
    }
    rope_kernel<<<static_cast<unsigned int>(grid), block_size, 0U, stream>>>(
        input, output, pair_count, problem.sequence, problem.heads, problem.head_dimension,
        problem.position_offset, problem.base);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace warpforge
