#include <warpforge/cuda_check.cuh>
#include <warpforge/rmsnorm.cuh>
#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <array>
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
    const std::size_t rows,
    const std::size_t columns,
    const float epsilon,
    const float magnitude = 3.0F) {
    const std::size_t count = rows * columns;
    std::vector<float> input(count);
    std::vector<float> weight(columns);
    std::mt19937 generator(2027U + static_cast<unsigned int>(rows + 13U * columns));
    std::uniform_real_distribution<float> input_distribution(-magnitude, magnitude);
    std::uniform_real_distribution<float> weight_distribution(0.5F, 1.5F);
    for (float& value : input) {
        value = input_distribution(generator);
    }
    for (float& value : weight) {
        value = weight_distribution(generator);
    }
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    warpforge::rmsnorm_cpu(
        input.data(), weight.data(), expected.data(), rows, columns, epsilon);
    DeviceBuffer device_input(count);
    DeviceBuffer device_weight(columns);
    DeviceBuffer device_output(count);
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            device_input.get(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_weight.get(),
            weight.data(),
            columns * sizeof(float),
            cudaMemcpyHostToDevice));
    }
    constexpr std::array variants{
        warpforge::RmsNormVariant::naive,
        warpforge::RmsNormVariant::block,
    };
    for (const auto variant : variants) {
        warpforge::rmsnorm_cuda(
            device_input.get(),
            device_weight.get(),
            device_output.get(),
            rows,
            columns,
            epsilon,
            variant);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (count > 0U) {
            CUDA_CHECK(cudaMemcpy(
                actual.data(),
                device_output.get(),
                count * sizeof(float),
                cudaMemcpyDeviceToHost));
        }
        const auto validation = warpforge::validate_fp32(
            expected.data(),
            actual.data(),
            count,
            warpforge::rmsnorm_tolerance(columns));
        if (!validation.passed) {
            throw std::runtime_error(
                std::string(warpforge::rmsnorm_variant_name(variant)) +
                " RMSNorm failed at index " + std::to_string(validation.worst_index));
        }
    }
}

void test_zero_and_near_zero() {
    run_case(4U, 513U, 1.0e-5F, 0.0F);
    run_case(3U, 257U, 1.0e-5F, 1.0e-10F);
}

void test_in_place() {
    constexpr std::size_t rows = 3U;
    constexpr std::size_t columns = 65U;
    std::vector<float> input(rows * columns);
    std::vector<float> weight(columns, 1.0F);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        input[index] = static_cast<float>(index % 23U) / 11.0F;
    }
    std::vector<float> expected(input.size());
    std::vector<float> actual(input.size());
    warpforge::rmsnorm_cpu(
        input.data(), weight.data(), expected.data(), rows, columns, 1.0e-5F);
    DeviceBuffer values(input.size());
    DeviceBuffer device_weight(weight.size());
    CUDA_CHECK(cudaMemcpy(
        values.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_weight.get(),
        weight.data(),
        weight.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    warpforge::rmsnorm_cuda(
        values.get(),
        device_weight.get(),
        values.get(),
        rows,
        columns,
        1.0e-5F,
        warpforge::RmsNormVariant::block);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(), values.get(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (!warpforge::validate_fp32(
             expected.data(),
             actual.data(),
             actual.size(),
             warpforge::rmsnorm_tolerance(columns))
             .passed) {
        throw std::runtime_error("in-place RMSNorm failed");
    }
}

void test_invalid_arguments() {
    bool epsilon_threw = false;
    try {
        warpforge::rmsnorm_cpu(nullptr, nullptr, nullptr, 0U, 0U, 0.0F);
    } catch (const std::invalid_argument&) {
        epsilon_threw = true;
    }
    if (!epsilon_threw) {
        throw std::runtime_error("non-positive RMSNorm epsilon was accepted");
    }
    warpforge::rmsnorm_cpu(nullptr, nullptr, nullptr, 0U, 17U, 1.0e-5F);
}

}  // namespace

int main() {
    try {
        constexpr std::array shapes{
            std::array<std::size_t, 2>{1U, 1U},
            std::array<std::size_t, 2>{2U, 17U},
            std::array<std::size_t, 2>{3U, 31U},
            std::array<std::size_t, 2>{3U, 32U},
            std::array<std::size_t, 2>{3U, 33U},
            std::array<std::size_t, 2>{4U, 257U},
            std::array<std::size_t, 2>{7U, 1003U},
        };
        for (const auto& shape : shapes) {
            run_case(shape[0], shape[1], 1.0e-5F);
        }
        test_zero_and_near_zero();
        test_in_place();
        test_invalid_arguments();
        std::cout << "Stage 6 RMSNorm correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 6 RMSNorm correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
