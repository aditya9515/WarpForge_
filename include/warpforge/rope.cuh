#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

struct RopeProblem final {
    std::size_t batch{};
    std::size_t sequence{};
    std::size_t heads{};
    std::size_t head_dimension{};
    std::size_t position_offset{};
    float base{10000.0F};
};

inline constexpr unsigned int rope_default_block_size = 256U;

[[nodiscard]] std::size_t rope_element_count(const RopeProblem& problem);
[[nodiscard]] std::size_t rope_pair_count(const RopeProblem& problem);

void rope_cpu(const float* input, float* output, const RopeProblem& problem);

void rope_cuda(const float* input, float* output, const RopeProblem& problem,
               unsigned int block_size = rope_default_block_size, cudaStream_t stream = nullptr);

} // namespace warpforge
