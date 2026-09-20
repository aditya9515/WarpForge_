#include <warpforge/cuda_check.cuh>
#include <warpforge/reduction.cuh>
#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<warpforge::ReductionVariant, 5> variants{
    warpforge::ReductionVariant::naive_interleaved,
    warpforge::ReductionVariant::shared_memory,
    warpforge::ReductionVariant::reduced_divergence,
    warpforge::ReductionVariant::unrolled,
    warpforge::ReductionVariant::warp_shuffle,
};

template <typename T>
class TestDeviceBuffer final {
public:
    explicit TestDeviceBuffer(const std::size_t count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        }
    }

    ~TestDeviceBuffer() noexcept {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    TestDeviceBuffer(const TestDeviceBuffer&) = delete;
    TestDeviceBuffer& operator=(const TestDeviceBuffer&) = delete;

    [[nodiscard]] T* get() const noexcept {
        return pointer_;
    }

private:
    T* pointer_{};
};

std::vector<float> make_input(
    const std::size_t element_count,
    const bool negative_only) {
    std::vector<float> input(element_count);
    if (negative_only) {
        for (std::size_t index = 0; index < element_count; ++index) {
            input[index] = -1.0F - static_cast<float>(index % 127U) / 17.0F;
        }
        return input;
    }

    std::mt19937 generator(2027U + static_cast<unsigned int>(element_count));
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (float& value : input) {
        value = distribution(generator);
    }
    if (!input.empty()) {
        input[element_count / 2U] = 7.25F;
    }
    return input;
}

void run_case(
    const std::size_t element_count,
    const warpforge::ReductionOperation operation,
    const warpforge::ReductionVariant variant,
    const unsigned int block_size,
    const bool negative_only = false) {
    const std::vector<float> input = make_input(element_count, negative_only);
    const std::size_t workspace_elements =
        warpforge::reduction_workspace_elements(element_count, variant, block_size);
    TestDeviceBuffer<float> device_input(element_count);
    TestDeviceBuffer<float> workspace_a(workspace_elements);
    TestDeviceBuffer<float> workspace_b(workspace_elements);
    TestDeviceBuffer<float> device_output(1);
    CUDA_CHECK(cudaMemcpy(
        device_input.get(),
        input.data(),
        element_count * sizeof(float),
        cudaMemcpyHostToDevice));

    warpforge::reduce_cuda(
        device_input.get(),
        element_count,
        operation,
        variant,
        workspace_a.get(),
        workspace_b.get(),
        workspace_elements,
        device_output.get(),
        block_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    float actual = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &actual, device_output.get(), sizeof(float), cudaMemcpyDeviceToHost));

    warpforge::ValidationResult validation;
    if (operation == warpforge::ReductionOperation::sum) {
        validation = warpforge::validate_reduction_sum(
            warpforge::reduction_sum_cpu(input.data(), input.size()),
            actual,
            input.size());
    } else {
        const float expected = warpforge::reduction_max_cpu(input.data(), input.size());
        validation = warpforge::validate_fp32(&expected, &actual, 1U, {0.0, 0.0});
    }
    if (!validation.passed) {
        throw std::runtime_error(
            std::string(warpforge::reduction_operation_name(operation)) + "/" +
            warpforge::reduction_variant_name(variant) + " failed for size " +
            std::to_string(element_count) + ", block " + std::to_string(block_size) +
            ", absolute error " + std::to_string(validation.max_absolute_error));
    }
}

void test_shapes_and_variants() {
    constexpr std::array<std::size_t, 16> sizes{
        1U,
        2U,
        17U,
        31U,
        32U,
        33U,
        127U,
        255U,
        256U,
        257U,
        511U,
        512U,
        513U,
        1003U,
        4097U,
        1U << 20U,
    };
    for (const std::size_t size : sizes) {
        for (const auto variant : variants) {
            run_case(
                size,
                warpforge::ReductionOperation::sum,
                variant,
                warpforge::reduction_default_block_size);
            run_case(
                size,
                warpforge::ReductionOperation::maximum,
                variant,
                warpforge::reduction_default_block_size);
        }
    }
}

void test_block_dispatch() {
    for (const unsigned int block_size :
         std::array<unsigned int, 5>{32U, 64U, 128U, 256U, 512U}) {
        run_case(
            1003U,
            warpforge::ReductionOperation::sum,
            warpforge::ReductionVariant::unrolled,
            block_size);
        run_case(
            1003U,
            warpforge::ReductionOperation::maximum,
            warpforge::ReductionVariant::warp_shuffle,
            block_size,
            true);
    }
}

void test_negative_only_maximum() {
    for (const auto variant : variants) {
        run_case(
            4097U,
            warpforge::ReductionOperation::maximum,
            variant,
            warpforge::reduction_default_block_size,
            true);
    }
}

void test_cpu_double_accumulation() {
    const std::array<float, 3> values{100000000.0F, 1.0F, -100000000.0F};
    const double result = warpforge::reduction_sum_cpu(values.data(), values.size());
    if (result != 1.0) {
        throw std::runtime_error("CPU sum reference did not retain double-precision accumulation");
    }
}

void test_invalid_arguments() {
    bool empty_threw = false;
    try {
        warpforge::reduce_cuda(
            nullptr,
            0,
            warpforge::ReductionOperation::sum,
            warpforge::ReductionVariant::warp_shuffle,
            nullptr,
            nullptr,
            0,
            nullptr);
    } catch (const std::invalid_argument&) {
        empty_threw = true;
    }
    if (!empty_threw) {
        throw std::runtime_error("empty GPU reduction should throw");
    }

    bool invalid_block_threw = false;
    try {
        static_cast<void>(warpforge::reduction_output_count(
            100U, warpforge::ReductionVariant::warp_shuffle, 96U));
    } catch (const std::invalid_argument&) {
        invalid_block_threw = true;
    }
    if (!invalid_block_threw) {
        throw std::runtime_error("non-power-of-two reduction block should throw");
    }

    TestDeviceBuffer<float> input(1024U);
    TestDeviceBuffer<float> output(1U);
    bool workspace_threw = false;
    try {
        warpforge::reduce_cuda(
            input.get(),
            1024U,
            warpforge::ReductionOperation::sum,
            warpforge::ReductionVariant::naive_interleaved,
            nullptr,
            nullptr,
            0,
            output.get());
    } catch (const std::invalid_argument&) {
        workspace_threw = true;
    }
    if (!workspace_threw) {
        throw std::runtime_error("undersized reduction workspace should throw");
    }
}

}  // namespace

int main() {
    try {
        test_shapes_and_variants();
        test_block_dispatch();
        test_negative_only_maximum();
        test_cpu_double_accumulation();
        test_invalid_arguments();
        std::cout << "Stage 4 reduction correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 4 reduction correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
