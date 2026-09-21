#include <warpforge/cuda_check.cuh>
#include <warpforge/elementwise.cuh>
#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <functional>
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

void require_valid(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const std::string& label) {
    const auto validation = warpforge::validate_fp32(
        expected.data(), actual.data(), actual.size(), {1.0e-5, 1.0e-5});
    if (!validation.passed) {
        throw std::runtime_error(
            label + " failed at index " + std::to_string(validation.worst_index));
    }
}

void run_case(const std::size_t count) {
    std::vector<float> left(count);
    std::vector<float> right(count);
    std::mt19937 generator(2027U + static_cast<unsigned int>(count));
    std::uniform_real_distribution<float> distribution(-6.0F, 6.0F);
    for (std::size_t index = 0U; index < count; ++index) {
        left[index] = distribution(generator);
        right[index] = distribution(generator);
    }
    if (count >= 4U) {
        left[0] = -100.0F;
        left[1] = -20.0F;
        left[2] = 20.0F;
        left[3] = 100.0F;
    }
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    DeviceBuffer device_left(count);
    DeviceBuffer device_right(count);
    DeviceBuffer device_output(count);
    DeviceBuffer device_intermediate(count);
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            device_left.get(), left.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_right.get(), right.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    const auto read_output = [&]() {
        CUDA_CHECK(cudaDeviceSynchronize());
        if (count > 0U) {
            CUDA_CHECK(cudaMemcpy(
                actual.data(),
                device_output.get(),
                count * sizeof(float),
                cudaMemcpyDeviceToHost));
        }
    };

    warpforge::silu_cpu(left.data(), expected.data(), count);
    warpforge::silu_cuda(device_left.get(), device_output.get(), count);
    read_output();
    require_valid(expected, actual, "SiLU");

    warpforge::add_cpu(left.data(), right.data(), expected.data(), count);
    warpforge::add_cuda(device_left.get(), device_right.get(), device_output.get(), count);
    read_output();
    require_valid(expected, actual, "add");

    warpforge::multiply_cpu(left.data(), right.data(), expected.data(), count);
    warpforge::multiply_cuda(
        device_left.get(), device_right.get(), device_output.get(), count);
    read_output();
    require_valid(expected, actual, "multiply");

    warpforge::scale_cpu(left.data(), 0.375F, expected.data(), count);
    warpforge::scale_cuda(device_left.get(), 0.375F, device_output.get(), count);
    read_output();
    require_valid(expected, actual, "scale");

    warpforge::swiglu_unfused_cpu(
        left.data(), right.data(), actual.data(), expected.data(), count);
    warpforge::swiglu_unfused_cuda(
        device_left.get(),
        device_right.get(),
        device_intermediate.get(),
        device_output.get(),
        count);
    read_output();
    require_valid(expected, actual, "unfused SwiGLU");
}

void test_aliasing() {
    constexpr std::size_t count = 1003U;
    std::vector<float> left(count);
    std::vector<float> right(count);
    for (std::size_t index = 0U; index < count; ++index) {
        left[index] = static_cast<float>(index % 37U) / 11.0F - 1.0F;
        right[index] = static_cast<float>(index % 29U) / 13.0F + 0.25F;
    }
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    DeviceBuffer device_left(count);
    DeviceBuffer device_right(count);
    DeviceBuffer intermediate(count);
    CUDA_CHECK(cudaMemcpy(
        device_left.get(), left.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_right.get(), right.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    warpforge::silu_cpu(left.data(), expected.data(), count);
    warpforge::silu_cuda(device_left.get(), device_left.get(), count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(), device_left.get(), count * sizeof(float), cudaMemcpyDeviceToHost));
    require_valid(expected, actual, "in-place SiLU");

    CUDA_CHECK(cudaMemcpy(
        device_left.get(), left.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    warpforge::add_cpu(left.data(), right.data(), expected.data(), count);
    warpforge::add_cuda(device_left.get(), device_right.get(), device_left.get(), count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(), device_left.get(), count * sizeof(float), cudaMemcpyDeviceToHost));
    require_valid(expected, actual, "aliased add");

    CUDA_CHECK(cudaMemcpy(
        device_left.get(), left.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    warpforge::swiglu_unfused_cpu(
        left.data(), right.data(), actual.data(), expected.data(), count);
    warpforge::swiglu_unfused_cuda(
        device_left.get(),
        device_right.get(),
        intermediate.get(),
        device_left.get(),
        count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(), device_left.get(), count * sizeof(float), cudaMemcpyDeviceToHost));
    require_valid(expected, actual, "SwiGLU output alias");
}

void test_invalid_arguments() {
    warpforge::silu_cuda(nullptr, nullptr, 0U);
    bool alias_threw = false;
    DeviceBuffer value(1U);
    DeviceBuffer other(1U);
    try {
        warpforge::swiglu_unfused_cuda(
            value.get(), other.get(), value.get(), other.get(), 1U);
    } catch (const std::invalid_argument&) {
        alias_threw = true;
    }
    if (!alias_threw) {
        throw std::runtime_error("SwiGLU accepted an aliased intermediate");
    }
}

}  // namespace

int main() {
    try {
        constexpr std::array<std::size_t, 8> sizes{
            0U, 1U, 17U, 31U, 32U, 33U, 1003U, 1U << 20U};
        for (const std::size_t size : sizes) {
            run_case(size);
        }
        test_aliasing();
        test_invalid_arguments();
        std::cout << "Stage 6 elementwise and SwiGLU correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 6 elementwise correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
