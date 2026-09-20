#pragma once

#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace warpforge {

enum class ReductionOperation {
    sum,
    maximum,
};

enum class ReductionVariant {
    naive_interleaved,
    shared_memory,
    reduced_divergence,
    unrolled,
    warp_shuffle,
};

inline constexpr unsigned int reduction_default_block_size = 256U;

[[nodiscard]] const char* reduction_operation_name(ReductionOperation operation) noexcept;
[[nodiscard]] const char* reduction_variant_name(ReductionVariant variant) noexcept;

[[nodiscard]] std::size_t reduction_elements_per_block(
    ReductionVariant variant,
    unsigned int block_size = reduction_default_block_size);

[[nodiscard]] std::size_t reduction_output_count(
    std::size_t input_count,
    ReductionVariant variant,
    unsigned int block_size = reduction_default_block_size);

[[nodiscard]] std::size_t reduction_workspace_elements(
    std::size_t input_count,
    ReductionVariant variant,
    unsigned int block_size = reduction_default_block_size);

[[nodiscard]] std::size_t reduction_pass_count(
    std::size_t input_count,
    ReductionVariant variant,
    unsigned int block_size = reduction_default_block_size);

[[nodiscard]] std::size_t reduction_dynamic_shared_memory_bytes(
    ReductionVariant variant,
    unsigned int block_size = reduction_default_block_size);

[[nodiscard]] double reduction_sum_cpu(const float* input, std::size_t element_count);
[[nodiscard]] float reduction_max_cpu(const float* input, std::size_t element_count);
[[nodiscard]] Tolerance reduction_sum_tolerance(std::size_t element_count);
[[nodiscard]] ValidationResult validate_reduction_sum(
    double expected,
    float actual,
    std::size_t input_count);

void reduce_cuda(
    const float* input,
    std::size_t element_count,
    ReductionOperation operation,
    ReductionVariant variant,
    float* workspace_a,
    float* workspace_b,
    std::size_t workspace_capacity,
    float* output,
    unsigned int block_size = reduction_default_block_size,
    cudaStream_t stream = nullptr);

}  // namespace warpforge
