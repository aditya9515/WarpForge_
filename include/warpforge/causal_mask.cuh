#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

struct CausalMaskProblem final {
    std::size_t batch{};
    std::size_t heads{};
    std::size_t query_length{};
    std::size_t key_length{};
    std::size_t query_position_offset{};
};

inline constexpr unsigned int causal_mask_default_block_size = 256U;

[[nodiscard]] std::size_t causal_mask_element_count(const CausalMaskProblem& problem);

void causal_mask_cpu(const float* input, float* output, const CausalMaskProblem& problem);

void causal_mask_cuda(const float* input, float* output, const CausalMaskProblem& problem,
                      unsigned int block_size = causal_mask_default_block_size,
                      cudaStream_t stream = nullptr);

} // namespace warpforge
