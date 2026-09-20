#include <warpforge/cuda_check.cuh>
#include <warpforge/memory.cuh>
#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

template <typename T>
class TestPinnedBuffer final {
public:
    explicit TestPinnedBuffer(const std::size_t count) : count_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        }
    }

    ~TestPinnedBuffer() noexcept {
        if (pointer_ != nullptr) {
            cudaFreeHost(pointer_);
        }
    }

    TestPinnedBuffer(const TestPinnedBuffer&) = delete;
    TestPinnedBuffer& operator=(const TestPinnedBuffer&) = delete;

    [[nodiscard]] T* get() const noexcept {
        return pointer_;
    }

    [[nodiscard]] T& operator[](const std::size_t index) noexcept {
        return pointer_[index];
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }

private:
    T* pointer_{};
    std::size_t count_{};
};

class TestStream final {
public:
    TestStream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~TestStream() noexcept {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    TestStream(const TestStream&) = delete;
    TestStream& operator=(const TestStream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept {
        return stream_;
    }

private:
    cudaStream_t stream_{};
};

void require_valid(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const std::string& label,
    const warpforge::Tolerance tolerance = {1.0e-5, 1.0e-5}) {
    const auto result = warpforge::validate_fp32(
        expected.data(), actual.data(), expected.size(), tolerance);
    if (!result.passed) {
        throw std::runtime_error(
            label + " mismatch at index " + std::to_string(result.worst_index));
    }
}

void test_saxpy(const std::size_t element_count, const unsigned int block_size) {
    std::vector<float> input(element_count);
    std::vector<float> initial(element_count);
    std::vector<float> expected(element_count);
    std::vector<float> actual(element_count);
    std::mt19937 generator(2027U + static_cast<unsigned int>(element_count));
    std::uniform_real_distribution<float> distribution(-4.0F, 4.0F);
    for (std::size_t index = 0; index < element_count; ++index) {
        input[index] = distribution(generator);
        initial[index] = distribution(generator);
    }
    expected = initial;
    warpforge::saxpy_cpu(1.75F, input.data(), expected.data(), element_count);

    TestDeviceBuffer<float> device_input(element_count);
    TestDeviceBuffer<float> device_output(element_count);
    if (element_count > 0) {
        const std::size_t bytes = element_count * sizeof(float);
        CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_output.get(), initial.data(), bytes, cudaMemcpyHostToDevice));
        warpforge::saxpy_cuda(
            1.75F, device_input.get(), device_output.get(), element_count, block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost));
    } else {
        warpforge::saxpy_cuda(1.75F, nullptr, nullptr, 0, block_size);
    }
    require_valid(expected, actual, "SAXPY");
}

void test_copy(const std::size_t element_count, const unsigned int block_size) {
    std::vector<float> input(element_count);
    std::vector<float> actual(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        input[index] = static_cast<float>(index % 97U) - 48.0F;
    }

    TestDeviceBuffer<float> device_input(element_count);
    TestDeviceBuffer<float> device_output(element_count);
    if (element_count > 0) {
        const std::size_t bytes = element_count * sizeof(float);
        CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice));
        warpforge::memory_copy_cuda(
            device_input.get(), device_output.get(), element_count, block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost));
    } else {
        warpforge::memory_copy_cuda(nullptr, nullptr, 0, block_size);
    }
    require_valid(input, actual, "custom copy");
}

void test_strided_copy(const std::size_t element_count, const std::size_t stride) {
    const std::size_t source_count = element_count == 0 ? 0 : (element_count - 1U) * stride + 1U;
    std::vector<float> input(source_count);
    std::vector<float> expected(element_count);
    std::vector<float> actual(element_count);
    for (std::size_t index = 0; index < source_count; ++index) {
        input[index] = static_cast<float>((index * 17U) % 251U);
    }
    warpforge::strided_copy_cpu(
        input.data(), expected.data(), element_count, stride);

    TestDeviceBuffer<float> device_input(source_count);
    TestDeviceBuffer<float> device_output(element_count);
    if (element_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            device_input.get(),
            input.data(),
            source_count * sizeof(float),
            cudaMemcpyHostToDevice));
        warpforge::strided_copy_cuda(
            device_input.get(), device_output.get(), element_count, stride);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            actual.data(),
            device_output.get(),
            element_count * sizeof(float),
            cudaMemcpyDeviceToHost));
    } else {
        warpforge::strided_copy_cuda(nullptr, nullptr, 0, stride);
    }
    require_valid(expected, actual, "strided copy");
}

void test_transpose(
    const std::size_t rows,
    const std::size_t columns,
    const bool tiled) {
    const std::size_t element_count = rows * columns;
    std::vector<float> input(element_count);
    std::vector<float> expected(element_count);
    std::vector<float> actual(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        input[index] = static_cast<float>(index % 1009U) / 7.0F;
    }
    warpforge::transpose_cpu(input.data(), expected.data(), rows, columns);

    TestDeviceBuffer<float> device_input(element_count);
    TestDeviceBuffer<float> device_output(element_count);
    if (element_count > 0) {
        const std::size_t bytes = element_count * sizeof(float);
        CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice));
        if (tiled) {
            warpforge::transpose_tiled_cuda(
                device_input.get(), device_output.get(), rows, columns);
        } else {
            warpforge::transpose_naive_cuda(
                device_input.get(), device_output.get(), rows, columns);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost));
    } else if (tiled) {
        warpforge::transpose_tiled_cuda(nullptr, nullptr, rows, columns);
    } else {
        warpforge::transpose_naive_cuda(nullptr, nullptr, rows, columns);
    }
    require_valid(expected, actual, tiled ? "tiled transpose" : "naive transpose");
}

void test_two_stream_ordering() {
    constexpr std::size_t element_count = 1003;
    constexpr std::size_t chunk_count = 5;
    constexpr std::size_t chunk_capacity =
        (element_count + chunk_count - 1U) / chunk_count;
    constexpr float alpha = 0.625F;

    TestPinnedBuffer<float> input(element_count);
    TestPinnedBuffer<float> initial(element_count);
    TestPinnedBuffer<float> output(element_count);
    std::vector<float> expected(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        input[index] = static_cast<float>(index % 113U) / 11.0F;
        initial[index] = static_cast<float>(index % 47U) / 13.0F;
        expected[index] = alpha * input[index] + initial[index];
    }

    std::array<TestStream, 2> streams{};
    std::array<TestDeviceBuffer<float>, 2> device_input{
        TestDeviceBuffer<float>{chunk_capacity}, TestDeviceBuffer<float>{chunk_capacity}};
    std::array<TestDeviceBuffer<float>, 2> device_output{
        TestDeviceBuffer<float>{chunk_capacity}, TestDeviceBuffer<float>{chunk_capacity}};

    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const std::size_t stream_index = chunk % streams.size();
        const std::size_t offset = chunk * chunk_capacity;
        const std::size_t count = std::min(chunk_capacity, element_count - offset);
        const std::size_t bytes = count * sizeof(float);
        const cudaStream_t stream = streams[stream_index].get();
        CUDA_CHECK(cudaMemcpyAsync(
            device_input[stream_index].get(),
            input.get() + offset,
            bytes,
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            device_output[stream_index].get(),
            initial.get() + offset,
            bytes,
            cudaMemcpyHostToDevice,
            stream));
        warpforge::saxpy_cuda(
            alpha,
            device_input[stream_index].get(),
            device_output[stream_index].get(),
            count,
            warpforge::memory_default_block_size,
            stream);
        CUDA_CHECK(cudaMemcpyAsync(
            output.get() + offset,
            device_output[stream_index].get(),
            bytes,
            cudaMemcpyDeviceToHost,
            stream));
    }
    for (const auto& stream : streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    }

    const std::vector<float> actual(output.get(), output.get() + output.size());
    require_valid(expected, actual, "two-stream pipeline ordering");
}

void test_invalid_arguments() {
    bool invalid_stride_threw = false;
    try {
        warpforge::strided_copy_cuda(nullptr, nullptr, 0, 0);
    } catch (const std::invalid_argument&) {
        invalid_stride_threw = true;
    }
    if (!invalid_stride_threw) {
        throw std::runtime_error("zero stride should throw");
    }

    bool invalid_block_threw = false;
    try {
        static_cast<void>(warpforge::memory_grid_size(10, 0));
    } catch (const std::invalid_argument&) {
        invalid_block_threw = true;
    }
    if (!invalid_block_threw) {
        throw std::runtime_error("zero block size should throw");
    }

    bool invalid_transpose_block_threw = false;
    try {
        static_cast<void>(warpforge::transpose_grid_size(10, 10, dim3{64U, 32U, 1U}));
    } catch (const std::invalid_argument&) {
        invalid_transpose_block_threw = true;
    }
    if (!invalid_transpose_block_threw) {
        throw std::runtime_error("oversized transpose block should throw");
    }
}

}  // namespace

int main() {
    try {
        constexpr std::array<std::size_t, 10> sizes{
            0, 1, 31, 32, 33, 255, 256, 257, 1003, 65537};
        constexpr std::array<unsigned int, 5> block_sizes{32U, 64U, 128U, 256U, 512U};
        for (const std::size_t size : sizes) {
            test_saxpy(size, warpforge::memory_default_block_size);
            test_copy(size, warpforge::memory_default_block_size);
        }
        for (const unsigned int block_size : block_sizes) {
            test_saxpy(1003, block_size);
            test_copy(1003, block_size);
        }
        for (const std::size_t stride :
             std::array<std::size_t, 6>{1U, 2U, 4U, 8U, 16U, 32U}) {
            test_strided_copy(1003, stride);
        }
        for (const auto [rows, columns] :
             std::array<std::array<std::size_t, 2>, 8>{
                 std::array<std::size_t, 2>{0, 0},
                 {0, 17},
                 {1, 1},
                 {3, 5},
                 {31, 33},
                 {32, 32},
                 {33, 65},
                 {129, 257}}) {
            test_transpose(rows, columns, false);
            test_transpose(rows, columns, true);
        }
        test_two_stream_ordering();
        test_invalid_arguments();
        std::cout << "Stage 3 CUDA memory correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 3 CUDA memory correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
