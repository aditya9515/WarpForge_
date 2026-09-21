#include <warpforge/cuda_check.cuh>
#include <warpforge/rope.cuh>
#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class DeviceBuffer final {
public:
    explicit DeviceBuffer(const std::size_t count) {
        if (count > 0U) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(float)));
        }
    }
    ~DeviceBuffer() noexcept {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] float* get() const noexcept { return pointer_; }

private:
    float* pointer_{};
};

void run_case(
    const warpforge::RopeProblem& problem,
    const bool in_place = false,
    const warpforge::Tolerance tolerance = {1.0e-5, 1.0e-5}) {
    const std::size_t count = warpforge::rope_element_count(problem);
    std::vector<float> input(count);
    std::mt19937 generator(
        2027U + static_cast<unsigned int>(problem.sequence + problem.head_dimension));
    std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
    for (float& value : input) {
        value = distribution(generator);
    }
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    warpforge::rope_cpu(input.data(), expected.data(), problem);
    DeviceBuffer device_input(count);
    DeviceBuffer device_output(in_place ? 0U : count);
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            device_input.get(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    float* output = in_place ? device_input.get() : device_output.get();
    warpforge::rope_cuda(device_input.get(), output, problem);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            actual.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    const auto validation =
        warpforge::validate_fp32(expected.data(), actual.data(), count, tolerance);
    if (!validation.passed) {
        throw std::runtime_error(
            "RoPE failed at index " + std::to_string(validation.worst_index) +
            " with max error " + std::to_string(validation.max_absolute_error));
    }
}

void test_position_layout() {
    const warpforge::RopeProblem problem{1U, 2U, 1U, 2U, 0U, 10000.0F};
    const std::vector<float> input{1.0F, 0.0F, 1.0F, 0.0F};
    std::vector<float> output(input.size());
    warpforge::rope_cpu(input.data(), output.data(), problem);
    if (output[0] != 1.0F || output[1] != 0.0F ||
        std::abs(output[2] - std::cos(1.0F)) > 1.0e-6F ||
        std::abs(output[3] - std::sin(1.0F)) > 1.0e-6F) {
        throw std::runtime_error("RoPE sequence-position layout is incorrect");
    }
}

void test_invalid_arguments() {
    bool odd_dimension = false;
    try {
        warpforge::rope_cpu(nullptr, nullptr, {1U, 1U, 1U, 3U, 0U, 10000.0F});
    } catch (const std::invalid_argument&) {
        odd_dimension = true;
    }
    if (!odd_dimension) {
        throw std::runtime_error("odd RoPE head dimension was accepted");
    }
    bool invalid_base = false;
    try {
        warpforge::rope_cuda(nullptr, nullptr, {0U, 0U, 0U, 0U, 0U, 0.0F});
    } catch (const std::invalid_argument&) {
        invalid_base = true;
    }
    if (!invalid_base) {
        throw std::runtime_error("invalid RoPE base was accepted");
    }
    warpforge::rope_cpu(nullptr, nullptr, {0U, 7U, 8U, 64U, 0U, 10000.0F});
}

}  // namespace

int main() {
    try {
        constexpr std::array problems{
            warpforge::RopeProblem{1U, 1U, 1U, 2U, 0U, 10000.0F},
            warpforge::RopeProblem{2U, 3U, 2U, 8U, 0U, 10000.0F},
            warpforge::RopeProblem{1U, 5U, 3U, 10U, 7U, 10000.0F},
            warpforge::RopeProblem{1U, 17U, 8U, 64U, 128U, 10000.0F},
            warpforge::RopeProblem{2U, 9U, 4U, 32U, 3U, 500000.0F},
        };
        for (const auto& problem : problems) {
            run_case(problem);
        }
        // Large positions amplify FP32 frequency rounding before sin/cos range reduction.
        run_case({1U, 2048U, 8U, 64U, 17U, 10000.0F}, false, {2.5e-4, 1.0e-5});
        run_case({1U, 7U, 2U, 16U, 4U, 10000.0F}, true);
        test_position_layout();
        test_invalid_arguments();
        std::cout << "Stage 6 RoPE correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 6 RoPE correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
