#include <warpforge/cuda_check.cuh>
#include <warpforge/miniinfer.cuh>
#include <warpforge/runtime.cuh>
#include <warpforge/validation.hpp>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check_cublas(const cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}

class CublasHandle final {
  public:
    CublasHandle() {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
        check_cublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
    }
    ~CublasHandle() noexcept {
        if (handle_ != nullptr) {
            (void)cublasDestroy(handle_);
        }
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    [[nodiscard]] cublasHandle_t get() const noexcept {
        return handle_;
    }

  private:
    cublasHandle_t handle_{nullptr};
};

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void validate_backend(warpforge::MiniInferBlock& block, const warpforge::MiniInferFixture& fixture,
                      warpforge::TensorView input, warpforge::TensorView output,
                      const warpforge::GemmBackend backend, const cublasHandle_t handle,
                      warpforge::CudaStream& stream) {
    std::set<std::string> observed;
    const auto observer = [&](const std::string_view name, const warpforge::TensorView& view,
                              const cudaStream_t observation_stream) {
        const auto expected = fixture.intermediates.find(std::string(name));
        if (expected == fixture.intermediates.end()) {
            throw std::runtime_error("unexpected MiniInfer intermediate: " + std::string(name));
        }
        std::vector<float> actual(view.element_count());
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), view.data_as<float>(),
                                   actual.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   observation_stream));
        CUDA_CHECK(cudaStreamSynchronize(observation_stream));
        const warpforge::ValidationResult validation = warpforge::validate_fp32(
            expected->second.data(), actual.data(), actual.size(),
            warpforge::miniinfer_intermediate_tolerance(name, fixture.config));
        if (!validation.passed) {
            throw std::runtime_error(
                std::string(warpforge::gemm_backend_name(backend)) + " backend " +
                std::string(name) + " failed at index " + std::to_string(validation.worst_index) +
                ", max absolute error " + std::to_string(validation.max_absolute_error));
        }
        observed.emplace(name);
    };

    block.forward(input, output, backend, handle, stream.native_handle(), observer);
    stream.synchronize();
    expect(observed.size() == warpforge::miniinfer_intermediate_names().size(),
           "not every MiniInfer intermediate was observed");
}

void test_invalid_config() {
    bool threw = false;
    try {
        warpforge::MiniInferConfig invalid;
        invalid.attention_heads = 7U;
        warpforge::MiniInferBlock block(invalid);
        (void)block;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "invalid MiniInfer attention width was accepted");
}

} // namespace

int main(const int argument_count, char** arguments) {
    try {
        if (argument_count != 2) {
            throw std::invalid_argument("usage: warpforge_miniinfer_test <fixture-directory>");
        }
        const warpforge::MiniInferFixture fixture =
            warpforge::load_miniinfer_fixture(std::filesystem::path(arguments[1]));
        expect(fixture.preset == "small", "CTest fixture must use the small preset");
        expect(fixture.seed == 2027U, "MiniInfer fixture seed mismatch");

        warpforge::CudaStream stream;
        CublasHandle cublas;
        warpforge::MiniInferBlock block(fixture.config);
        block.upload_weights(fixture.weights, stream.native_handle());
        warpforge::Tensor input(
            {fixture.config.batch, fixture.config.sequence, fixture.config.hidden_size},
            warpforge::DType::fp32);
        warpforge::Tensor output(
            {fixture.config.batch, fixture.config.sequence, fixture.config.hidden_size},
            warpforge::DType::fp32);
        input.copy_from_host_async(fixture.input.data(), fixture.input.size() * sizeof(float),
                                   stream.native_handle());
        stream.synchronize();

        void* workspace_pointer = block.workspace().native_handle();
        const std::size_t workspace_capacity = block.workspace().capacity_bytes();
        auto input_view = input.view();
        auto output_view = output.view();
        validate_backend(block, fixture, input_view, output_view, warpforge::GemmBackend::custom,
                         nullptr, stream);
        validate_backend(block, fixture, input_view, output_view, warpforge::GemmBackend::cublas,
                         cublas.get(), stream);

        block.forward(input_view, output_view, warpforge::GemmBackend::custom, nullptr,
                      stream.native_handle());
        stream.synchronize();
        expect(block.workspace().native_handle() == workspace_pointer, "workspace pointer changed");
        expect(block.workspace().capacity_bytes() == workspace_capacity,
               "workspace capacity changed");

        test_invalid_config();
        std::cout << "MiniInfer fixture and backend tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MiniInfer tests failed: " << error.what() << '\n';
        return 1;
    }
}
