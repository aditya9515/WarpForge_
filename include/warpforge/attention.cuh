#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

struct AttentionProblem final {
    std::size_t batch{};
    std::size_t sequence{};
    std::size_t heads{};
    std::size_t head_dimension{};
};

inline constexpr unsigned int attention_default_block_size = 256U;

[[nodiscard]] std::size_t attention_qkv_element_count(const AttentionProblem& problem);
[[nodiscard]] std::size_t attention_score_element_count(const AttentionProblem& problem);

// Q/K/V and context use contiguous [batch, sequence, heads, head_dimension].
// Scores/probabilities use contiguous [batch, heads, query, key].
void attention_scores_cpu(const float* query, const float* key, float* scores,
                          const AttentionProblem& problem);

void attention_scores_cuda(const float* query, const float* key, float* scores,
                           const AttentionProblem& problem,
                           unsigned int block_size = attention_default_block_size,
                           cudaStream_t stream = nullptr);

void attention_value_cpu(const float* probabilities, const float* value, float* context,
                         const AttentionProblem& problem);

void attention_value_cuda(const float* probabilities, const float* value, float* context,
                          const AttentionProblem& problem,
                          unsigned int block_size = attention_default_block_size,
                          cudaStream_t stream = nullptr);

} // namespace warpforge
