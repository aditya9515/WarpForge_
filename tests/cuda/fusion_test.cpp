#include <warpforge/cuda_check.cuh>
#include <warpforge/elementwise.cuh>
#include <warpforge/fusion.cuh>
#include <warpforge/rmsnorm.cuh>
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

void require_valid(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const warpforge::Tolerance tolerance,
    const std::string& label) {
    const auto validation = warpforge::validate_fp32(
        expected.data(), actual.data(), expected.size(), tolerance);
    if (!validation.passed) {
        throw std::runtime_error(
            label + " failed at index " + std::to_string(validation.worst_index) +
            " with max error " + std::to_string(validation.max_absolute_error));
    }
}

void run_residual_rmsnorm_case(
    const std::size_t rows,
    const std::size_t columns,
    const bool output_aliases_input = false,
    const bool output_aliases_residual = false) {
    const std::size_t count = rows * columns;
    std::mt19937 generator(2027U + static_cast<unsigned int>(rows + columns));
    std::uniform_real_distribution<float> value_distribution(-3.0F, 3.0F);
    std::uniform_real_distribution<float> weight_distribution(0.5F, 1.5F);
    std::vector<float> input(count);
    std::vector<float> residual(count);
    std::vector<float> weight(columns);
    for (float& value : input) {
        value = value_distribution(generator);
    }
    for (float& value : residual) {
        value = value_distribution(generator);
    }
    for (float& value : weight) {
        value = weight_distribution(generator);
    }
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    constexpr float epsilon = 1.0e-5F;
    warpforge::residual_rmsnorm_cpu(
        input.data(), residual.data(), weight.data(), expected.data(), rows, columns, epsilon);

    DeviceBuffer device_input(count);
    DeviceBuffer device_residual(count);
    DeviceBuffer device_weight(columns);
    DeviceBuffer device_intermediate(count);
    DeviceBuffer device_output(
        output_aliases_input || output_aliases_residual ? 0U : count);
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            device_input.get(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_residual.get(), residual.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_weight.get(), weight.data(), columns * sizeof(float), cudaMemcpyHostToDevice));
    }
    float* output = output_aliases_input
        ? device_input.get()
        : (output_aliases_residual ? device_residual.get() : device_output.get());
    warpforge::residual_rmsnorm_fused_cuda(
        device_input.get(),
        device_residual.get(),
        device_weight.get(),
        output,
        rows,
        columns,
        epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            actual.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    require_valid(
        expected, actual, warpforge::rmsnorm_tolerance(columns),
        "fused residual RMSNorm");

    if (!output_aliases_input && !output_aliases_residual && count > 0U) {
        warpforge::add_cuda(
            device_input.get(), device_residual.get(), device_intermediate.get(), count);
        warpforge::rmsnorm_cuda(
            device_intermediate.get(),
            device_weight.get(),
            device_output.get(),
            rows,
            columns,
            epsilon,
            warpforge::RmsNormVariant::block);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            actual.data(), device_output.get(), count * sizeof(float), cudaMemcpyDeviceToHost));
        require_valid(
            expected, actual, warpforge::rmsnorm_tolerance(columns),
            "separate residual plus RMSNorm");
    }
}

void run_swiglu_case(
    const std::size_t count,
    const bool output_aliases_gate = false,
    const bool output_aliases_up = false) {
    std::mt19937 generator(2027U + static_cast<unsigned int>(count));
    std::uniform_real_distribution<float> distribution(-5.0F, 5.0F);
    std::vector<float> gate(count);
    std::vector<float> up(count);
    for (float& value : gate) {
        value = distribution(generator);
    }
    for (float& value : up) {
        value = distribution(generator);
    }
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    warpforge::swiglu_fused_cpu(gate.data(), up.data(), expected.data(), count);

    DeviceBuffer device_gate(count);
    DeviceBuffer device_up(count);
    DeviceBuffer device_intermediate(count);
    DeviceBuffer device_output(output_aliases_gate || output_aliases_up ? 0U : count);
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            device_gate.get(), gate.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_up.get(), up.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    float* output = output_aliases_gate
        ? device_gate.get()
        : (output_aliases_up ? device_up.get() : device_output.get());
    warpforge::swiglu_fused_cuda(device_gate.get(), device_up.get(), output, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (count > 0U) {
        CUDA_CHECK(cudaMemcpy(
            actual.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    require_valid(expected, actual, {1.0e-5, 1.0e-5}, "fused SwiGLU");

    if (!output_aliases_gate && !output_aliases_up && count > 0U) {
        warpforge::swiglu_unfused_cuda(
            device_gate.get(),
            device_up.get(),
            device_intermediate.get(),
            device_output.get(),
            count);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            actual.data(), device_output.get(), count * sizeof(float), cudaMemcpyDeviceToHost));
        require_valid(expected, actual, {1.0e-5, 1.0e-5}, "unfused SwiGLU");
    }
}

void test_extreme_swiglu() {
    const std::vector<float> gate{-100.0F, -20.0F, 0.0F, 20.0F, 100.0F};
    const std::vector<float> up{2.0F, -3.0F, 4.0F, -5.0F, 6.0F};
    std::vector<float> expected(gate.size());
    std::vector<float> actual(gate.size());
    warpforge::swiglu_fused_cpu(gate.data(), up.data(), expected.data(), gate.size());
    DeviceBuffer device_gate(gate.size());
    DeviceBuffer device_up(up.size());
    DeviceBuffer device_output(gate.size());
    CUDA_CHECK(cudaMemcpy(
        device_gate.get(), gate.data(), gate.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_up.get(), up.data(), up.size() * sizeof(float), cudaMemcpyHostToDevice));
    warpforge::swiglu_fused_cuda(
        device_gate.get(), device_up.get(), device_output.get(), gate.size());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_output.get(),
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    require_valid(expected, actual, {1.0e-5, 1.0e-5}, "extreme fused SwiGLU");
}

void test_invalid_arguments() {
    bool invalid_epsilon = false;
    try {
        warpforge::residual_rmsnorm_cpu(nullptr, nullptr, nullptr, nullptr, 0U, 0U, 0.0F);
    } catch (const std::invalid_argument&) {
        invalid_epsilon = true;
    }
    if (!invalid_epsilon) {
        throw std::runtime_error("invalid fused residual RMSNorm epsilon was accepted");
    }

    bool invalid_weight_alias = false;
    try {
        float value = 1.0F;
        warpforge::residual_rmsnorm_cpu(
            &value, &value, &value, &value, 1U, 1U, 1.0e-5F);
    } catch (const std::invalid_argument&) {
        invalid_weight_alias = true;
    }
    if (!invalid_weight_alias) {
        throw std::runtime_error("fused residual RMSNorm weight alias was accepted");
    }

    bool invalid_block = false;
    try {
        warpforge::swiglu_fused_cuda(nullptr, nullptr, nullptr, 0U, 17U);
    } catch (const std::invalid_argument&) {
        invalid_block = true;
    }
    if (!invalid_block) {
        throw std::runtime_error("invalid fusion block size was accepted");
    }
}

}  // namespace

int main() {
    try {
        constexpr std::array<std::array<std::size_t, 2>, 7> shapes{{
            {0U, 0U}, {1U, 1U}, {3U, 17U}, {2U, 31U},
            {4U, 32U}, {3U, 33U}, {37U, 53U}}};
        for (const auto& shape : shapes) {
            run_residual_rmsnorm_case(shape[0], shape[1]);
        }
        run_residual_rmsnorm_case(3U, 33U, true, false);
        run_residual_rmsnorm_case(3U, 33U, false, true);

        constexpr std::array<std::size_t, 8> sizes{
            0U, 1U, 17U, 31U, 32U, 33U, 1003U, 1U << 20U};
        for (const std::size_t size : sizes) {
            run_swiglu_case(size);
        }
        run_swiglu_case(1003U, true, false);
        run_swiglu_case(1003U, false, true);
        test_extreme_swiglu();
        test_invalid_arguments();
        std::cout << "Stage 7 fusion correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 7 fusion correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
