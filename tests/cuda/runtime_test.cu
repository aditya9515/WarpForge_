#include "warpforge/cuda_check.cuh"
#include "warpforge/gemm.cuh"
#include "warpforge/runtime.cuh"
#include "warpforge/validation.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void expect_throw(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

class CublasHandle final {
public:
    CublasHandle() {
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
    }

    ~CublasHandle() noexcept {
        if (handle_ != nullptr) {
            (void)cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_{nullptr};
};

void test_shape_and_dtype() {
    const warpforge::TensorShape shape{2U, 3U, 5U};
    expect(shape.rank() == 3U, "TensorShape rank mismatch");
    expect(shape.dimension(1U) == 3U, "TensorShape dimension mismatch");
    expect(shape.element_count() == 30U, "TensorShape element count mismatch");
    expect(shape.byte_size(warpforge::DType::fp32) == 120U, "FP32 byte count mismatch");
    expect(shape.byte_size(warpforge::DType::fp16) == 60U, "FP16 byte count mismatch");
    expect(std::string(warpforge::dtype_name(warpforge::DType::fp32)) == "fp32", "dtype name mismatch");

    const warpforge::TensorShape empty;
    const warpforge::TensorShape zero_extent{4U, 0U, 9U};
    expect(empty.empty() && zero_extent.empty(), "zero-element shapes must be empty");
    expect_throw<std::overflow_error>(
        [] {
            const warpforge::TensorShape overflowing{
                std::numeric_limits<std::size_t>::max(), 2U};
            (void)overflowing;
        },
        "shape overflow was not rejected");
}

void test_buffer_stream_event_and_tensor() {
    static_assert(!std::is_copy_constructible_v<warpforge::DeviceBuffer<float>>);
    static_assert(std::is_nothrow_move_constructible_v<warpforge::DeviceBuffer<float>>);
    static_assert(!std::is_copy_constructible_v<warpforge::Tensor>);
    static_assert(std::is_nothrow_move_constructible_v<warpforge::Tensor>);
    static_assert(std::is_nothrow_move_constructible_v<warpforge::CudaStream>);
    static_assert(std::is_nothrow_move_constructible_v<warpforge::CudaEvent>);

    warpforge::DeviceBuffer<float> zero(0U);
    expect(zero.empty() && zero.data() == nullptr, "zero-sized DeviceBuffer must not allocate");
    warpforge::Tensor zero_tensor(warpforge::TensorShape{0U}, warpforge::DType::fp16);
    expect(zero_tensor.empty() && zero_tensor.native_handle() == nullptr, "zero-sized Tensor must not allocate");

    constexpr std::size_t count = 1025U;
    std::vector<float> source(count);
    std::vector<float> destination(count, 0.0F);
    for (std::size_t index = 0; index < count; ++index) {
        source[index] = static_cast<float>(index) * 0.25F - 17.0F;
    }

    warpforge::CudaStream stream;
    warpforge::CudaEvent start;
    warpforge::CudaEvent stop;
    warpforge::DeviceBuffer<float> original(count);
    start.record(stream);
    original.copy_from_host_async(source.data(), source.size(), stream.native_handle());
    original.copy_to_host_async(destination.data(), destination.size(), stream.native_handle());
    stop.record(stream);
    stop.synchronize();
    expect(stop.elapsed_ms_since(start) >= 0.0F, "CUDA event elapsed time must be non-negative");
    expect(destination == source, "asynchronous DeviceBuffer round trip failed");

    float* native_pointer = original.data();
    warpforge::DeviceBuffer<float> moved(std::move(original));
    expect(original.data() == nullptr && original.size() == 0U, "moved-from DeviceBuffer not empty");
    expect(moved.data() == native_pointer && moved.size() == count, "DeviceBuffer move lost ownership");

    warpforge::DeviceBuffer<float> assigned;
    assigned = std::move(moved);
    expect(moved.data() == nullptr && assigned.data() == native_pointer, "DeviceBuffer move assignment failed");
    float* released = assigned.release();
    expect(assigned.empty() && assigned.data() == nullptr, "DeviceBuffer release did not clear ownership");
    CUDA_CHECK(cudaFree(released));

    warpforge::Tensor tensor(warpforge::TensorShape{count}, warpforge::DType::fp32);
    tensor.copy_from_host_async(source.data(), source.size() * sizeof(float), stream.native_handle());
    std::fill(destination.begin(), destination.end(), 0.0F);
    tensor.copy_to_host_async(destination.data(), destination.size() * sizeof(float), stream.native_handle());
    stream.synchronize();
    expect(destination == source, "Tensor asynchronous round trip failed");
    expect(tensor.view().data_as<float>() == tensor.native_handle(), "TensorView native pointer mismatch");
    expect_throw<std::invalid_argument>(
        [&tensor] { (void)tensor.view().data_as<__half>(); },
        "TensorView accepted a mismatched typed access");

    void* tensor_pointer = tensor.native_handle();
    warpforge::Tensor moved_tensor(std::move(tensor));
    expect(tensor.empty() && tensor.element_count() == 0U, "moved-from Tensor retained metadata");
    expect(moved_tensor.native_handle() == tensor_pointer, "Tensor move lost ownership");

    warpforge::CudaStream moved_stream(std::move(stream));
    expect(stream.native_handle() == nullptr, "moved-from CudaStream retained ownership");
    cudaEvent_t event_handle = stop.native_handle();
    warpforge::CudaEvent moved_event(std::move(stop));
    expect(stop.native_handle() == nullptr, "moved-from CudaEvent retained ownership");
    expect(moved_event.native_handle() == event_handle, "CudaEvent move lost ownership");
    moved_event.record(moved_stream);
    moved_stream.synchronize();
}

void test_workspace_reuse_and_alignment() {
    warpforge::DeviceWorkspace workspace(4096U);
    void* first = workspace.allocate_bytes(33U, 256U);
    void* second = workspace.allocate_bytes(17U, 128U);
    expect(reinterpret_cast<std::uintptr_t>(first) % 256U == 0U, "workspace 256-byte alignment failed");
    expect(reinterpret_cast<std::uintptr_t>(second) % 128U == 0U, "workspace 128-byte alignment failed");
    expect(first != second, "workspace allocations overlap");

    const std::size_t capacity = workspace.capacity_bytes();
    workspace.clear();
    void* reused = workspace.allocate_bytes(33U, 256U);
    expect(reused == first, "workspace clear did not reuse the allocation");
    expect(workspace.capacity_bytes() == capacity, "workspace clear changed capacity");

    workspace.clear();
    warpforge::TensorView view = workspace.allocate_view(
        warpforge::TensorShape{8U, 8U}, warpforge::DType::fp16, 256U);
    expect(view.byte_size() == 128U && view.dtype() == warpforge::DType::fp16, "workspace TensorView mismatch");

    expect_throw<std::invalid_argument>(
        [&workspace] { (void)workspace.allocate_bytes(1U, 3U); },
        "non-power-of-two workspace alignment was accepted");
    expect_throw<std::length_error>(
        [&workspace] { (void)workspace.allocate_bytes(8192U); },
        "workspace capacity overrun was accepted");

    workspace.reserve(8192U);
    expect(workspace.capacity_bytes() == 8192U && workspace.used_bytes() == 0U, "workspace reserve failed");
    void* workspace_pointer = workspace.native_handle();
    warpforge::DeviceWorkspace moved_workspace(std::move(workspace));
    expect(
        workspace.capacity_bytes() == 0U && workspace.used_bytes() == 0U,
        "moved-from workspace retained state");
    expect(moved_workspace.native_handle() == workspace_pointer, "workspace move lost ownership");
    moved_workspace.reset();
    expect(
        moved_workspace.capacity_bytes() == 0U && moved_workspace.native_handle() == nullptr,
        "workspace reset failed");
}

void test_allocation_failure() {
    expect_throw<std::runtime_error>(
        [] {
            warpforge::DeviceBuffer<std::byte> impossible(
                std::numeric_limits<std::size_t>::max() / 2U);
            (void)impossible;
        },
        "an impossible CUDA allocation unexpectedly succeeded");
    // cudaMalloc reports the failure directly and may also leave it visible to
    // the next cudaGetLastError-based launch check. Consume that expected state.
    (void)cudaGetLastError();
}

void run_gemm_backend(
    const warpforge::GemmDispatch dispatch,
    const cublasHandle_t handle,
    const warpforge::TensorView& a,
    const warpforge::TensorView& b,
    warpforge::TensorView c,
    const warpforge::GemmProblem& problem,
    warpforge::CudaStream& stream,
    const std::vector<float>& expected) {
    warpforge::gemm_cuda(a, b, c, problem, dispatch, handle, stream.native_handle());
    std::vector<float> actual(expected.size(), 0.0F);
    CUDA_CHECK(cudaMemcpyAsync(
        actual.data(),
        c.data_as<float>(),
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream.native_handle()));
    stream.synchronize();
    const warpforge::ValidationResult validation = warpforge::validate_fp32(
        expected.data(), actual.data(), actual.size(), warpforge::gemm_fp32_tolerance(problem.k));
    expect(validation.passed, std::string("TensorView GEMM dispatch failed for ") + warpforge::gemm_backend_name(dispatch.backend));
}

void test_gemm_backend_dispatch() {
    const warpforge::GemmProblem problem{3U, 5U, 7U};
    std::vector<float> host_a(warpforge::gemm_a_elements(problem));
    std::vector<float> host_b(warpforge::gemm_b_elements(problem));
    std::vector<float> expected(warpforge::gemm_c_elements(problem));
    for (std::size_t index = 0; index < host_a.size(); ++index) {
        host_a[index] = static_cast<float>(static_cast<int>(index % 11U) - 5) / 8.0F;
    }
    for (std::size_t index = 0; index < host_b.size(); ++index) {
        host_b[index] = static_cast<float>(static_cast<int>(index % 7U) - 3) / 6.0F;
    }
    warpforge::gemm_cpu_fp32(host_a.data(), host_b.data(), expected.data(), problem);

    warpforge::CudaStream stream;
    warpforge::Tensor a(warpforge::TensorShape{problem.m, problem.k}, warpforge::DType::fp32);
    warpforge::Tensor b(warpforge::TensorShape{problem.k, problem.n}, warpforge::DType::fp32);
    warpforge::Tensor c(warpforge::TensorShape{problem.m, problem.n}, warpforge::DType::fp32);
    a.copy_from_host_async(host_a.data(), host_a.size() * sizeof(float), stream.native_handle());
    b.copy_from_host_async(host_b.data(), host_b.size() * sizeof(float), stream.native_handle());

    CublasHandle cublas;
    run_gemm_backend(
        {warpforge::GemmBackend::custom, warpforge::GemmVariant::tiled_fp32},
        nullptr,
        a.view(),
        b.view(),
        c.view(),
        problem,
        stream,
        expected);
    run_gemm_backend(
        {warpforge::GemmBackend::cublas, warpforge::GemmVariant::naive_fp32},
        cublas.get(),
        a.view(),
        b.view(),
        c.view(),
        problem,
        stream,
        expected);

    expect_throw<std::invalid_argument>(
        [&] {
            warpforge::Tensor wrong(warpforge::TensorShape{problem.m, problem.n - 1U}, warpforge::DType::fp32);
            warpforge::gemm_cuda(
                a.view(), b.view(), wrong.view(), problem, {}, nullptr, stream.native_handle());
        },
        "GEMM dispatch accepted a mismatched output view");
}

}  // namespace

int main(const int argument_count, char** arguments) {
    try {
        bool skip_allocation_failure = false;
        for (int index = 1; index < argument_count; ++index) {
            const std::string argument = arguments[index];
            if (argument == "--skip-allocation-failure") {
                skip_allocation_failure = true;
            } else {
                throw std::invalid_argument("unknown runtime test argument: " + argument);
            }
        }
        test_shape_and_dtype();
        test_buffer_stream_event_and_tensor();
        test_workspace_reuse_and_alignment();
        if (!skip_allocation_failure) {
            test_allocation_failure();
        }
        test_gemm_backend_dispatch();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "WarpForge runtime tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "WarpForge runtime tests failed: " << error.what() << '\n';
        return 1;
    }
}
