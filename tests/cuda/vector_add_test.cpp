#include <warpforge/cuda_check.cuh>
#include <warpforge/benchmark.hpp>
#include <warpforge/validation.hpp>
#include <warpforge/vector_add.cuh>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

template <typename T>
class TestDeviceAllocation final {
public:
    explicit TestDeviceAllocation(const std::size_t count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        }
    }

    ~TestDeviceAllocation() noexcept {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    TestDeviceAllocation(const TestDeviceAllocation&) = delete;
    TestDeviceAllocation& operator=(const TestDeviceAllocation&) = delete;

    [[nodiscard]] T* get() const noexcept {
        return pointer_;
    }

private:
    T* pointer_{};
};

void run_case(const std::size_t element_count, const unsigned int block_size) {
    std::vector<float> left(element_count);
    std::vector<float> right(element_count);
    std::vector<float> expected(element_count);
    std::vector<float> actual(element_count);

    std::mt19937 generator(2027U + static_cast<unsigned int>(element_count));
    std::uniform_real_distribution<float> distribution(-10.0F, 10.0F);
    for (std::size_t index = 0; index < element_count; ++index) {
        left[index] = distribution(generator);
        right[index] = distribution(generator);
    }
    warpforge::vector_add_cpu(left.data(), right.data(), expected.data(), element_count);

    TestDeviceAllocation<float> device_left(element_count);
    TestDeviceAllocation<float> device_right(element_count);
    TestDeviceAllocation<float> device_output(element_count);
    if (element_count > 0) {
        const std::size_t bytes = element_count * sizeof(float);
        CUDA_CHECK(cudaMemcpy(device_left.get(), left.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_right.get(), right.data(), bytes, cudaMemcpyHostToDevice));
        warpforge::vector_add_cuda(
            device_left.get(),
            device_right.get(),
            device_output.get(),
            element_count,
            block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost));
    } else {
        warpforge::vector_add_cuda(nullptr, nullptr, nullptr, 0, block_size);
    }

    const auto validation =
        warpforge::validate_fp32(expected.data(), actual.data(), element_count, {1.0e-5, 1.0e-5});
    if (!validation.passed) {
        throw std::runtime_error("VectorAdd mismatch for size " + std::to_string(element_count));
    }
}

}  // namespace

int main() {
    try {
        for (const std::size_t element_count :
             std::array<std::size_t, 11>{0, 1, 31, 32, 33, 255, 256, 257, 1003, 4097, 1U << 20U}) {
            run_case(element_count, warpforge::vector_add_default_block_size);
        }
        for (const unsigned int block_size : std::array<unsigned int, 5>{32, 64, 128, 256, 512}) {
            run_case(1003, block_size);
        }

        bool invalid_block_threw = false;
        try {
            static_cast<void>(warpforge::vector_add_grid_size(100, 0));
        } catch (const std::invalid_argument&) {
            invalid_block_threw = true;
        }
        if (!invalid_block_threw) {
            throw std::runtime_error("zero block size should throw");
        }

        bool oversized_grid_threw = false;
        try {
            static_cast<void>(warpforge::vector_add_grid_size(
                std::numeric_limits<std::size_t>::max(), 1));
        } catch (const std::overflow_error&) {
            oversized_grid_threw = true;
        }
        if (!oversized_grid_threw) {
            throw std::runtime_error("oversized grids should throw");
        }

        bool zero_iterations_threw = false;
        try {
            warpforge::BenchmarkConfig invalid_config;
            invalid_config.measurement_iterations = 0;
            static_cast<void>(warpforge::measure_cuda_kernel(
                invalid_config, nullptr, [](cudaStream_t) {}));
        } catch (const std::invalid_argument&) {
            zero_iterations_threw = true;
        }
        if (!zero_iterations_threw) {
            throw std::runtime_error("zero measurement iterations should throw");
        }

        std::cout << "VectorAdd CPU/CUDA validation tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "VectorAdd CPU/CUDA validation tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
