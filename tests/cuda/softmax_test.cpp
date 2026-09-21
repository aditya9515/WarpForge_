#include <warpforge/cuda_check.cuh>
#include <warpforge/softmax.cuh>
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
    const std::size_t rows,
    const std::size_t columns,
    const bool extreme = false) {
    const std::size_t count = rows * columns;
    std::vector<float> input(count);
    std::mt19937 generator(2027U + static_cast<unsigned int>(rows * 17U + columns));
    std::uniform_real_distribution<float> distribution(-8.0F, 8.0F);
    for (std::size_t index = 0U; index < count; ++index) {
        input[index] = distribution(generator);
    }
    if (extreme && columns >= 4U) {
        for (std::size_t row = 0U; row < rows; ++row) {
            input[row * columns] = 1000.0F;
            input[row * columns + 1U] = 999.0F;
            input[row * columns + 2U] = -1000.0F;
            input[row * columns + 3U] = -999.0F;
        }
    }
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    warpforge::softmax_cpu(input.data(), expected.data(), rows, columns);
    DeviceBuffer device_input(count);
    DeviceBuffer device_output(count);
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            device_input.get(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    constexpr std::array variants{
        warpforge::SoftmaxVariant::naive,
        warpforge::SoftmaxVariant::block,
        warpforge::SoftmaxVariant::warp,
    };
    for (const auto variant : variants) {
        warpforge::softmax_cuda(
            device_input.get(), device_output.get(), rows, columns, variant);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (count > 0U) {
            CUDA_CHECK(cudaMemcpy(
                actual.data(),
                device_output.get(),
                count * sizeof(float),
                cudaMemcpyDeviceToHost));
        }
        const auto validation = warpforge::validate_fp32(
            expected.data(), actual.data(), count, {1.0e-5, 1.0e-5});
        if (!validation.passed) {
            throw std::runtime_error(
                std::string(warpforge::softmax_variant_name(variant)) +
                " softmax failed at index " + std::to_string(validation.worst_index));
        }
        for (std::size_t row = 0U; row < rows; ++row) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < columns; ++column) {
                const float value = actual[row * columns + column];
                if (!std::isfinite(value) || value < 0.0F) {
                    throw std::runtime_error("softmax produced a non-finite or negative value");
                }
                sum += value;
            }
            if (std::abs(sum - 1.0) > 2.0e-5) {
                throw std::runtime_error("softmax row does not sum to one");
            }
        }
    }
}

void test_in_place() {
    constexpr std::size_t rows = 3U;
    constexpr std::size_t columns = 33U;
    std::vector<float> input(rows * columns);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        input[index] = static_cast<float>(index % 19U) - 9.0F;
    }
    std::vector<float> expected(input.size());
    std::vector<float> actual(input.size());
    warpforge::softmax_cpu(input.data(), expected.data(), rows, columns);
    DeviceBuffer values(input.size());
    CUDA_CHECK(cudaMemcpy(
        values.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    warpforge::softmax_cuda(
        values.get(), values.get(), rows, columns, warpforge::SoftmaxVariant::warp);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(), values.get(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (!warpforge::validate_fp32(expected.data(), actual.data(), actual.size()).passed) {
        throw std::runtime_error("in-place softmax failed");
    }
}

void test_invalid_arguments() {
    warpforge::softmax_cpu(nullptr, nullptr, 0U, 17U);
    warpforge::softmax_cuda(
        nullptr, nullptr, 4U, 0U, warpforge::SoftmaxVariant::warp);
    bool invalid_block = false;
    try {
        static_cast<void>(warpforge::softmax_dynamic_shared_memory_bytes(
            warpforge::SoftmaxVariant::block, 48U));
    } catch (const std::invalid_argument&) {
        invalid_block = true;
    }
    if (!invalid_block) {
        throw std::runtime_error("invalid softmax block size was accepted");
    }
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
            run_case(shape[0], shape[1]);
        }
        run_case(5U, 65U, true);
        test_in_place();
        test_invalid_arguments();
        std::cout << "Stage 6 Softmax correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 6 Softmax correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
