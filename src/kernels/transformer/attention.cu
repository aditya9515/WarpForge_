#include <warpforge/attention.cuh>
#include <warpforge/cuda_check.cuh>

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

void validate_problem(const AttentionProblem& problem) {
    if (problem.batch == 0U || problem.sequence == 0U || problem.heads == 0U ||
        problem.head_dimension == 0U) {
        throw std::invalid_argument("attention dimensions must be positive");
    }
    static_cast<void>(attention_qkv_element_count(problem));
    static_cast<void>(attention_score_element_count(problem));
}

void validate_block_size(const unsigned int block_size) {
    if (block_size == 0U || block_size > 1024U) {
        throw std::invalid_argument("attention block size must be from 1 through 1024");
    }
}

[[nodiscard]] unsigned int grid_dimension(const std::size_t elements,
                                          const unsigned int block_size) {
    const std::size_t grid = 1U + (elements - 1U) / block_size;
    if (grid > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("attention grid exceeds the CUDA x-dimension range");
    }
    return static_cast<unsigned int>(grid);
}

__global__ void attention_scores_kernel(const float* query, const float* key, float* scores,
                                        const std::size_t score_count, const std::size_t sequence,
                                        const std::size_t heads, const std::size_t head_dimension,
                                        const float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= score_count) {
        return;
    }

    const std::size_t key_position = index % sequence;
    const std::size_t query_position = (index / sequence) % sequence;
    const std::size_t head = (index / (sequence * sequence)) % heads;
    const std::size_t batch = index / (heads * sequence * sequence);
    const std::size_t query_offset =
        ((batch * sequence + query_position) * heads + head) * head_dimension;
    const std::size_t key_offset =
        ((batch * sequence + key_position) * heads + head) * head_dimension;

    float accumulator = 0.0F;
    for (std::size_t dimension = 0U; dimension < head_dimension; ++dimension) {
        accumulator =
            fmaf(query[query_offset + dimension], key[key_offset + dimension], accumulator);
    }
    scores[index] = accumulator * scale;
}

__global__ void attention_value_kernel(const float* probabilities, const float* value,
                                       float* context, const std::size_t context_count,
                                       const std::size_t sequence, const std::size_t heads,
                                       const std::size_t head_dimension) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= context_count) {
        return;
    }

    const std::size_t dimension = index % head_dimension;
    const std::size_t head = (index / head_dimension) % heads;
    const std::size_t query_position = (index / (head_dimension * heads)) % sequence;
    const std::size_t batch = index / (head_dimension * heads * sequence);
    const std::size_t probability_offset =
        ((batch * heads + head) * sequence + query_position) * sequence;

    float accumulator = 0.0F;
    for (std::size_t key_position = 0U; key_position < sequence; ++key_position) {
        const std::size_t value_offset =
            ((batch * sequence + key_position) * heads + head) * head_dimension;
        accumulator = fmaf(probabilities[probability_offset + key_position],
                           value[value_offset + dimension], accumulator);
    }
    context[index] = accumulator;
}

} // namespace

std::size_t attention_qkv_element_count(const AttentionProblem& problem) {
    const std::size_t tokens =
        checked_product(problem.batch, problem.sequence, "attention token count");
    const std::size_t head_groups = checked_product(tokens, problem.heads, "attention head count");
    return checked_product(head_groups, problem.head_dimension, "attention QKV element count");
}

std::size_t attention_score_element_count(const AttentionProblem& problem) {
    const std::size_t head_groups =
        checked_product(problem.batch, problem.heads, "attention head count");
    const std::size_t queries =
        checked_product(head_groups, problem.sequence, "attention query count");
    return checked_product(queries, problem.sequence, "attention score element count");
}

void attention_scores_cpu(const float* query, const float* key, float* scores,
                          const AttentionProblem& problem) {
    validate_problem(problem);
    if (query == nullptr || key == nullptr || scores == nullptr) {
        throw std::invalid_argument("attention scores require non-null pointers");
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(problem.head_dimension));
    for (std::size_t batch = 0U; batch < problem.batch; ++batch) {
        for (std::size_t head = 0U; head < problem.heads; ++head) {
            for (std::size_t query_position = 0U; query_position < problem.sequence;
                 ++query_position) {
                for (std::size_t key_position = 0U; key_position < problem.sequence;
                     ++key_position) {
                    const std::size_t query_offset =
                        ((batch * problem.sequence + query_position) * problem.heads + head) *
                        problem.head_dimension;
                    const std::size_t key_offset =
                        ((batch * problem.sequence + key_position) * problem.heads + head) *
                        problem.head_dimension;
                    double accumulator = 0.0;
                    for (std::size_t dimension = 0U; dimension < problem.head_dimension;
                         ++dimension) {
                        accumulator += static_cast<double>(query[query_offset + dimension]) *
                                       static_cast<double>(key[key_offset + dimension]);
                    }
                    const std::size_t score_index =
                        ((batch * problem.heads + head) * problem.sequence + query_position) *
                            problem.sequence +
                        key_position;
                    scores[score_index] = static_cast<float>(accumulator * scale);
                }
            }
        }
    }
}

void attention_scores_cuda(const float* query, const float* key, float* scores,
                           const AttentionProblem& problem, const unsigned int block_size,
                           const cudaStream_t stream) {
    validate_problem(problem);
    validate_block_size(block_size);
    if (query == nullptr || key == nullptr || scores == nullptr) {
        throw std::invalid_argument("attention scores require non-null pointers");
    }
    if (scores == query || scores == key) {
        throw std::invalid_argument("attention scores output must not alias Q or K");
    }
    const std::size_t score_count = attention_score_element_count(problem);
    const unsigned int grid = grid_dimension(score_count, block_size);
    attention_scores_kernel<<<grid, block_size, 0U, stream>>>(
        query, key, scores, score_count, problem.sequence, problem.heads, problem.head_dimension,
        rsqrtf(static_cast<float>(problem.head_dimension)));
    CUDA_CHECK(cudaGetLastError());
}

void attention_value_cpu(const float* probabilities, const float* value, float* context,
                         const AttentionProblem& problem) {
    validate_problem(problem);
    if (probabilities == nullptr || value == nullptr || context == nullptr) {
        throw std::invalid_argument("attention value product requires non-null pointers");
    }
    for (std::size_t batch = 0U; batch < problem.batch; ++batch) {
        for (std::size_t query_position = 0U; query_position < problem.sequence; ++query_position) {
            for (std::size_t head = 0U; head < problem.heads; ++head) {
                const std::size_t probability_offset =
                    ((batch * problem.heads + head) * problem.sequence + query_position) *
                    problem.sequence;
                for (std::size_t dimension = 0U; dimension < problem.head_dimension; ++dimension) {
                    double accumulator = 0.0;
                    for (std::size_t key_position = 0U; key_position < problem.sequence;
                         ++key_position) {
                        const std::size_t value_offset =
                            ((batch * problem.sequence + key_position) * problem.heads + head) *
                            problem.head_dimension;
                        accumulator +=
                            static_cast<double>(probabilities[probability_offset + key_position]) *
                            static_cast<double>(value[value_offset + dimension]);
                    }
                    const std::size_t context_index =
                        ((batch * problem.sequence + query_position) * problem.heads + head) *
                            problem.head_dimension +
                        dimension;
                    context[context_index] = static_cast<float>(accumulator);
                }
            }
        }
    }
}

void attention_value_cuda(const float* probabilities, const float* value, float* context,
                          const AttentionProblem& problem, const unsigned int block_size,
                          const cudaStream_t stream) {
    validate_problem(problem);
    validate_block_size(block_size);
    if (probabilities == nullptr || value == nullptr || context == nullptr) {
        throw std::invalid_argument("attention value product requires non-null pointers");
    }
    if (context == probabilities || context == value) {
        throw std::invalid_argument("attention context output must not alias probabilities or V");
    }
    const std::size_t context_count = attention_qkv_element_count(problem);
    const unsigned int grid = grid_dimension(context_count, block_size);
    attention_value_kernel<<<grid, block_size, 0U, stream>>>(probabilities, value, context,
                                                             context_count, problem.sequence,
                                                             problem.heads, problem.head_dimension);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace warpforge
