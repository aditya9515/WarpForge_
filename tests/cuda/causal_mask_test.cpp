#include <warpforge/causal_mask.cuh>
#include <warpforge/cuda_check.cuh>
#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
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
    [[nodiscard]] float* get() const noexcept {
        return pointer_;
    }

  private:
    float* pointer_{};
};

void run_case(const warpforge::CausalMaskProblem& problem, const bool in_place = false) {
    const std::size_t count = warpforge::causal_mask_element_count(problem);
    std::vector<float> input(count);
    for (std::size_t index = 0U; index < count; ++index) {
        input[index] = static_cast<float>(index % 101U) / 17.0F;
    }
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    warpforge::causal_mask_cpu(input.data(), expected.data(), problem);
    DeviceBuffer device_input(count);
    DeviceBuffer device_output(in_place ? 0U : count);
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), count * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    float* output = in_place ? device_input.get() : device_output.get();
    warpforge::causal_mask_cuda(device_input.get(), output, problem);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (count > 0U) {
        CUDA_CHECK(
            cudaMemcpy(actual.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    const auto validation =
        warpforge::validate_fp32(expected.data(), actual.data(), count, {0.0, 0.0});
    if (!validation.passed) {
        throw std::runtime_error("causal mask failed at index " +
                                 std::to_string(validation.worst_index));
    }
}

void test_boundaries() {
    const warpforge::CausalMaskProblem problem{1U, 1U, 3U, 5U, 1U};
    std::vector<float> input(15U, 0.0F);
    std::vector<float> output(15U);
    warpforge::causal_mask_cpu(input.data(), output.data(), problem);
    for (std::size_t query = 0U; query < 3U; ++query) {
        for (std::size_t key = 0U; key < 5U; ++key) {
            const bool should_mask = key > query + 1U;
            const float value = output[query * 5U + key];
            if (should_mask != std::isinf(value)) {
                throw std::runtime_error("causal-mask boundary is incorrect");
            }
        }
    }
}

void test_empty() {
    warpforge::causal_mask_cpu(nullptr, nullptr, {0U, 8U, 17U, 17U, 0U});
    warpforge::causal_mask_cuda(nullptr, nullptr, {1U, 8U, 0U, 17U, 0U});
}

} // namespace

int main() {
    try {
        constexpr std::array problems{
            warpforge::CausalMaskProblem{1U, 1U, 1U, 1U, 0U},
            warpforge::CausalMaskProblem{1U, 2U, 3U, 5U, 0U},
            warpforge::CausalMaskProblem{2U, 3U, 17U, 19U, 4U},
            warpforge::CausalMaskProblem{1U, 8U, 33U, 33U, 0U},
        };
        for (const auto& problem : problems) {
            run_case(problem);
        }
        run_case({1U, 2U, 7U, 11U, 3U}, true);
        test_boundaries();
        test_empty();
        std::cout << "Stage 6 causal-mask correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 6 causal-mask correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
