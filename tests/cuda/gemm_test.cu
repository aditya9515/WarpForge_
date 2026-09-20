#include <warpforge/cuda_check.cuh>
#include <warpforge/gemm.cuh>
#include <warpforge/validation.hpp>

#include <cublas_v2.h>
#include <cuda_fp16.h>
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

template <typename T>
class TestDeviceBuffer final {
public:
    explicit TestDeviceBuffer(const std::size_t count) {
        if (count > 0U) {
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

class TestCublasHandle final {
public:
    TestCublasHandle() {
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
        if (cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasSetMathMode failed");
        }
    }

    ~TestCublasHandle() noexcept {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    TestCublasHandle(const TestCublasHandle&) = delete;
    TestCublasHandle& operator=(const TestCublasHandle&) = delete;

    [[nodiscard]] cublasHandle_t get() const noexcept {
        return handle_;
    }

private:
    cublasHandle_t handle_{};
};

void require_valid(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const warpforge::Tolerance tolerance,
    const std::string& label) {
    const auto result = warpforge::validate_fp32(
        expected.data(), actual.data(), expected.size(), tolerance);
    if (!result.passed) {
        throw std::runtime_error(
            label + " failed at output " + std::to_string(result.worst_index) +
            " with max abs error " + std::to_string(result.max_absolute_error));
    }
}

void make_inputs(
    const warpforge::GemmProblem& problem,
    std::vector<float>& a,
    std::vector<float>& b) {
    a.resize(warpforge::gemm_a_elements(problem));
    b.resize(warpforge::gemm_b_elements(problem));
    std::mt19937 generator(
        2027U + static_cast<unsigned int>(problem.m + 3U * problem.n + 7U * problem.k));
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (float& value : a) {
        value = distribution(generator);
    }
    for (float& value : b) {
        value = distribution(generator);
    }
}

void test_fp32_problem(
    const warpforge::GemmProblem& problem,
    const cublasHandle_t handle) {
    std::vector<float> a;
    std::vector<float> b;
    make_inputs(problem, a, b);
    std::vector<float> expected(warpforge::gemm_c_elements(problem));
    std::vector<float> actual(expected.size());
    warpforge::gemm_cpu_fp32(a.data(), b.data(), expected.data(), problem);

    TestDeviceBuffer<float> device_a(a.size());
    TestDeviceBuffer<float> device_b(b.size());
    TestDeviceBuffer<float> device_c(actual.size());
    CUDA_CHECK(cudaMemcpy(
        device_a.get(), a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_b.get(), b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

    constexpr std::array variants{
        warpforge::GemmVariant::naive_fp32,
        warpforge::GemmVariant::tiled_fp32,
        warpforge::GemmVariant::coalesced_fp32,
        warpforge::GemmVariant::register_blocked_fp32,
    };
    for (const auto variant : variants) {
        warpforge::gemm_fp32_cuda(
            device_a.get(),
            device_b.get(),
            device_c.get(),
            problem,
            {warpforge::GemmBackend::custom, variant});
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            actual.data(),
            device_c.get(),
            actual.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        require_valid(
            expected,
            actual,
            warpforge::gemm_fp32_tolerance(problem.k),
            warpforge::gemm_variant_name(variant));
    }

    warpforge::gemm_fp32_cuda(
        device_a.get(),
        device_b.get(),
        device_c.get(),
        problem,
        {warpforge::GemmBackend::cublas, warpforge::GemmVariant::naive_fp32},
        handle);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_c.get(),
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    require_valid(
        expected,
        actual,
        warpforge::gemm_fp32_tolerance(problem.k),
        "cuBLAS FP32");
}

void test_fp16_problem(
    const warpforge::GemmProblem& problem,
    const cublasHandle_t handle,
    const bool include_wmma) {
    std::vector<float> source_a;
    std::vector<float> source_b;
    make_inputs(problem, source_a, source_b);
    std::vector<__half> a(source_a.size());
    std::vector<__half> b(source_b.size());
    std::vector<float> quantized_a(source_a.size());
    std::vector<float> quantized_b(source_b.size());
    for (std::size_t index = 0; index < source_a.size(); ++index) {
        a[index] = __float2half(source_a[index]);
        quantized_a[index] = __half2float(a[index]);
    }
    for (std::size_t index = 0; index < source_b.size(); ++index) {
        b[index] = __float2half(source_b[index]);
        quantized_b[index] = __half2float(b[index]);
    }
    std::vector<float> expected(warpforge::gemm_c_elements(problem));
    std::vector<float> actual(expected.size());
    warpforge::gemm_cpu_fp32(
        quantized_a.data(), quantized_b.data(), expected.data(), problem);

    TestDeviceBuffer<__half> device_a(a.size());
    TestDeviceBuffer<__half> device_b(b.size());
    TestDeviceBuffer<float> device_c(actual.size());
    CUDA_CHECK(cudaMemcpy(
        device_a.get(), a.data(), a.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_b.get(), b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice));

    std::vector<warpforge::GemmVariant> variants{
        warpforge::GemmVariant::tiled_fp16_fp32};
    if (include_wmma) {
        variants.push_back(warpforge::GemmVariant::wmma_fp16_fp32);
    }
    for (const auto variant : variants) {
        warpforge::gemm_fp16_cuda(
            device_a.get(),
            device_b.get(),
            device_c.get(),
            problem,
            {warpforge::GemmBackend::custom, variant});
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            actual.data(),
            device_c.get(),
            actual.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        require_valid(
            expected,
            actual,
            warpforge::gemm_fp16_tolerance(problem.k),
            warpforge::gemm_variant_name(variant));
    }

    warpforge::gemm_fp16_cuda(
        device_a.get(),
        device_b.get(),
        device_c.get(),
        problem,
        {warpforge::GemmBackend::cublas, warpforge::GemmVariant::tiled_fp16_fp32},
        handle);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_c.get(),
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    require_valid(
        expected,
        actual,
        warpforge::gemm_fp16_tolerance(problem.k),
        "cuBLAS FP16/FP32");
}

void test_empty_and_invalid() {
    warpforge::gemm_cpu_fp32(nullptr, nullptr, nullptr, {0U, 7U, 3U});
    warpforge::gemm_fp32_cuda(
        nullptr,
        nullptr,
        nullptr,
        {4U, 0U, 3U},
        {warpforge::GemmBackend::custom, warpforge::GemmVariant::naive_fp32});

    bool null_threw = false;
    try {
        warpforge::gemm_fp32_cuda(
            nullptr,
            nullptr,
            nullptr,
            {2U, 3U, 4U},
            {warpforge::GemmBackend::custom, warpforge::GemmVariant::naive_fp32});
    } catch (const std::invalid_argument&) {
        null_threw = true;
    }
    if (!null_threw) {
        throw std::runtime_error("non-empty GEMM accepted null pointers");
    }

    bool wmma_shape_threw = false;
    TestDeviceBuffer<__half> a(17U * 16U);
    TestDeviceBuffer<__half> b(16U * 16U);
    TestDeviceBuffer<float> c(17U * 16U);
    try {
        warpforge::gemm_fp16_cuda(
            a.get(),
            b.get(),
            c.get(),
            {17U, 16U, 16U},
            {warpforge::GemmBackend::custom, warpforge::GemmVariant::wmma_fp16_fp32});
    } catch (const std::invalid_argument&) {
        wmma_shape_threw = true;
    }
    if (!wmma_shape_threw) {
        throw std::runtime_error("WMMA accepted an incompatible shape");
    }
}

}  // namespace

int main() {
    try {
        TestCublasHandle handle;
        constexpr std::array problems{
            warpforge::GemmProblem{1U, 1U, 1U},
            warpforge::GemmProblem{3U, 5U, 7U},
            warpforge::GemmProblem{15U, 17U, 13U},
            warpforge::GemmProblem{16U, 16U, 16U},
            warpforge::GemmProblem{17U, 31U, 33U},
            warpforge::GemmProblem{32U, 32U, 32U},
            warpforge::GemmProblem{65U, 47U, 33U},
        };
        for (const auto& problem : problems) {
            test_fp32_problem(problem, handle.get());
        }
        test_fp16_problem({7U, 19U, 13U}, handle.get(), false);
        test_fp16_problem({16U, 16U, 16U}, handle.get(), true);
        test_fp16_problem({32U, 48U, 64U}, handle.get(), true);
        test_empty_and_invalid();
        std::cout << "Stage 5 GEMM correctness tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 5 GEMM correctness tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
