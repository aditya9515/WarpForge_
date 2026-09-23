#include <warpforge/attention.cuh>
#include <warpforge/cuda_check.cuh>
#include <warpforge/runtime.cuh>
#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void run_case(const warpforge::AttentionProblem& problem) {
    const std::size_t qkv_count = warpforge::attention_qkv_element_count(problem);
    const std::size_t score_count = warpforge::attention_score_element_count(problem);
    std::vector<float> query(qkv_count);
    std::vector<float> key(qkv_count);
    std::vector<float> value(qkv_count);
    std::vector<float> probabilities(score_count);
    for (std::size_t index = 0U; index < qkv_count; ++index) {
        query[index] = std::sin(static_cast<float>(index) * 0.071F);
        key[index] = std::cos(static_cast<float>(index) * 0.053F);
        value[index] = static_cast<float>(static_cast<int>(index % 29U) - 14) / 17.0F;
    }
    for (std::size_t row = 0U; row < problem.batch * problem.heads * problem.sequence; ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0U; column < problem.sequence; ++column) {
            const float probability = static_cast<float>(column + 1U);
            probabilities[row * problem.sequence + column] = probability;
            sum += probability;
        }
        for (std::size_t column = 0U; column < problem.sequence; ++column) {
            probabilities[row * problem.sequence + column] /= sum;
        }
    }

    std::vector<float> expected_scores(score_count);
    std::vector<float> expected_context(qkv_count);
    warpforge::attention_scores_cpu(query.data(), key.data(), expected_scores.data(), problem);
    warpforge::attention_value_cpu(probabilities.data(), value.data(), expected_context.data(),
                                   problem);

    warpforge::CudaStream stream;
    warpforge::DeviceBuffer<float> device_query(qkv_count);
    warpforge::DeviceBuffer<float> device_key(qkv_count);
    warpforge::DeviceBuffer<float> device_value(qkv_count);
    warpforge::DeviceBuffer<float> device_probabilities(score_count);
    warpforge::DeviceBuffer<float> device_scores(score_count);
    warpforge::DeviceBuffer<float> device_context(qkv_count);
    device_query.copy_from_host_async(query.data(), qkv_count, stream.native_handle());
    device_key.copy_from_host_async(key.data(), qkv_count, stream.native_handle());
    device_value.copy_from_host_async(value.data(), qkv_count, stream.native_handle());
    device_probabilities.copy_from_host_async(probabilities.data(), score_count,
                                              stream.native_handle());
    warpforge::attention_scores_cuda(device_query.data(), device_key.data(), device_scores.data(),
                                     problem, 256U, stream.native_handle());
    warpforge::attention_value_cuda(device_probabilities.data(), device_value.data(),
                                    device_context.data(), problem, 256U, stream.native_handle());

    std::vector<float> actual_scores(score_count);
    std::vector<float> actual_context(qkv_count);
    device_scores.copy_to_host_async(actual_scores.data(), score_count, stream.native_handle());
    device_context.copy_to_host_async(actual_context.data(), qkv_count, stream.native_handle());
    stream.synchronize();

    const double score_scale =
        std::max(1.0, std::ceil(std::log2(static_cast<double>(problem.head_dimension))));
    const double value_scale =
        std::max(1.0, std::ceil(std::log2(static_cast<double>(problem.sequence))));
    const auto score_validation = warpforge::validate_fp32(
        expected_scores.data(), actual_scores.data(), score_count, {1.0e-5 * score_scale, 1.0e-5});
    const auto value_validation = warpforge::validate_fp32(
        expected_context.data(), actual_context.data(), qkv_count, {1.0e-5 * value_scale, 1.0e-5});
    expect(score_validation.passed, "attention score validation failed");
    expect(value_validation.passed, "attention value validation failed");
}

void test_invalid_arguments() {
    bool threw = false;
    try {
        const warpforge::AttentionProblem invalid{1U, 0U, 1U, 8U};
        warpforge::attention_scores_cpu(nullptr, nullptr, nullptr, invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "zero attention dimension was accepted");

    threw = false;
    try {
        const warpforge::AttentionProblem problem{1U, 1U, 1U, 8U};
        float value = 0.0F;
        warpforge::attention_scores_cuda(&value, &value, &value, problem, 0U);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "zero attention block size was accepted");
}

} // namespace

int main() {
    try {
        run_case({1U, 1U, 1U, 2U});
        run_case({1U, 5U, 2U, 8U});
        run_case({2U, 17U, 3U, 10U});
        test_invalid_arguments();
        std::cout << "Attention tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Attention tests failed: " << error.what() << '\n';
        return 1;
    }
}
