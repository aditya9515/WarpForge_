#include <warpforge/causal_mask.cuh>
#include <warpforge/cuda_check.cuh>

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

void validate_problem(const CausalMaskProblem& problem) {
    if (problem.query_length > 0U &&
        problem.query_position_offset >
            std::numeric_limits<std::size_t>::max() - (problem.query_length - 1U)) {
        throw std::overflow_error("causal-mask query position range overflows size_t");
    }
    static_cast<void>(causal_mask_element_count(problem));
}

__global__ void causal_mask_kernel(const float* input, float* output,
                                   const std::size_t element_count, const std::size_t query_length,
                                   const std::size_t key_length,
                                   const std::size_t query_position_offset) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const std::size_t key = index % key_length;
    const std::size_t query = (index / key_length) % query_length;
    const bool masked = key > query_position_offset + query;
    output[index] = masked ? -__int_as_float(0x7f800000) : input[index];
}

} // namespace

std::size_t causal_mask_element_count(const CausalMaskProblem& problem) {
    const std::size_t head_groups =
        checked_product(problem.batch, problem.heads, "causal-mask head count");
    const std::size_t queries =
        checked_product(head_groups, problem.query_length, "causal-mask query count");
    return checked_product(queries, problem.key_length, "causal-mask element count");
}

void causal_mask_cpu(const float* input, float* output, const CausalMaskProblem& problem) {
    validate_problem(problem);
    const std::size_t element_count = causal_mask_element_count(problem);
    if (element_count == 0U) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("non-empty causal mask requires non-null input and output");
    }
    const float negative_infinity = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0U; index < element_count; ++index) {
        const std::size_t key = index % problem.key_length;
        const std::size_t query = (index / problem.key_length) % problem.query_length;
        output[index] =
            key > problem.query_position_offset + query ? negative_infinity : input[index];
    }
}

void causal_mask_cuda(const float* input, float* output, const CausalMaskProblem& problem,
                      const unsigned int block_size, const cudaStream_t stream) {
    validate_problem(problem);
    if (block_size == 0U || block_size > 1024U) {
        throw std::invalid_argument("causal-mask block size must be from 1 through 1024");
    }
    const std::size_t element_count = causal_mask_element_count(problem);
    if (element_count == 0U) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("non-empty causal mask requires non-null input and output");
    }
    const std::size_t grid = (element_count + block_size - 1U) / block_size;
    if (grid > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("causal-mask grid exceeds the CUDA x-dimension range");
    }
    causal_mask_kernel<<<static_cast<unsigned int>(grid), block_size, 0U, stream>>>(
        input, output, element_count, problem.query_length, problem.key_length,
        problem.query_position_offset);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace warpforge
