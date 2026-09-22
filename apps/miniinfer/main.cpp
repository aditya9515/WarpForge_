#include <warpforge/benchmark.hpp>
#include <warpforge/cuda_check.cuh>
#include <warpforge/miniinfer.cuh>
#include <warpforge/runtime.cuh>
#include <warpforge/validation.hpp>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

enum class BackendSelection { all, custom, cublas };

struct Options final {
    std::filesystem::path fixture_directory{"benchmarks/fixtures/miniinfer_full"};
    std::filesystem::path output_directory{"benchmarks/results/stage9"};
    warpforge::BenchmarkConfig benchmark{};
    BackendSelection backend{BackendSelection::all};
    bool show_help{};
};

struct RecordedResult final {
    std::string filename;
    warpforge::BenchmarkResult result;
};

struct IntermediateValidationRecord final {
    std::string backend;
    std::string name;
    warpforge::Tolerance tolerance;
    warpforge::ValidationResult validation;
};

struct BackendValidation final {
    warpforge::ValidationResult aggregate;
    std::vector<IntermediateValidationRecord> intermediates;
};

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
    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_{nullptr};
};

std::uint64_t parse_unsigned(const std::string& text, const std::string_view option) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(option) + " requires a non-negative integer");
    }
    std::size_t consumed = 0U;
    const std::uint64_t value = std::stoull(text, &consumed, 10);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(option) + " requires an integer value");
    }
    return value;
}

Options parse_options(const int argument_count, char** arguments) {
    Options options;
    for (int index = 1; index < argument_count; ++index) {
        const std::string_view option = arguments[index];
        if (option == "--help" || option == "-h") {
            options.show_help = true;
            continue;
        }
        if (index + 1 >= argument_count) {
            throw std::invalid_argument(std::string(option) + " requires a value");
        }
        const std::string value = arguments[++index];
        if (option == "--fixture-dir") {
            options.fixture_directory = value;
        } else if (option == "--output-dir") {
            options.output_directory = value;
        } else if (option == "--warmups") {
            options.benchmark.warmup_iterations =
                static_cast<std::size_t>(parse_unsigned(value, option));
        } else if (option == "--iterations") {
            options.benchmark.measurement_iterations =
                static_cast<std::size_t>(parse_unsigned(value, option));
        } else if (option == "--backend") {
            if (value == "all") {
                options.backend = BackendSelection::all;
            } else if (value == "custom") {
                options.backend = BackendSelection::custom;
            } else if (value == "cublas") {
                options.backend = BackendSelection::cublas;
            } else {
                throw std::invalid_argument("--backend must be all, custom, or cublas");
            }
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }
    if (options.benchmark.measurement_iterations == 0U) {
        throw std::invalid_argument("--iterations must be greater than zero");
    }
    return options;
}

void print_usage() {
    std::cout
        << "Usage: warpforge-miniinfer [options]\n"
        << "  --fixture-dir <path>   PyTorch fixture directory\n"
        << "  --output-dir <path>    Benchmark result directory\n"
        << "  --backend <name>       all, custom, or cublas (default: all)\n"
        << "  --warmups <count>      Warmup iterations (default: 10)\n"
        << "  --iterations <count>   Measured iterations (default: 100)\n"
        << "  --help                 Show this message\n";
}

[[nodiscard]] std::vector<warpforge::GemmBackend> selected_backends(
    const BackendSelection selection) {
    if (selection == BackendSelection::custom) {
        return {warpforge::GemmBackend::custom};
    }
    if (selection == BackendSelection::cublas) {
        return {warpforge::GemmBackend::cublas};
    }
    return {warpforge::GemmBackend::custom, warpforge::GemmBackend::cublas};
}

void merge_validation(
    warpforge::ValidationResult& aggregate,
    const warpforge::ValidationResult& current,
    const std::size_t index_offset) {
    const double prior_error_sum =
        aggregate.mean_absolute_error * static_cast<double>(aggregate.element_count);
    const double current_error_sum =
        current.mean_absolute_error * static_cast<double>(current.element_count);
    aggregate.passed = aggregate.passed && current.passed;
    aggregate.failure_count += current.failure_count;
    aggregate.element_count += current.element_count;
    aggregate.mean_absolute_error = aggregate.element_count == 0U
        ? 0.0
        : (prior_error_sum + current_error_sum) /
              static_cast<double>(aggregate.element_count);
    aggregate.max_relative_error =
        std::max(aggregate.max_relative_error, current.max_relative_error);
    if (current.max_absolute_error > aggregate.max_absolute_error) {
        aggregate.max_absolute_error = current.max_absolute_error;
        aggregate.worst_index = index_offset + current.worst_index;
        aggregate.expected_at_worst = current.expected_at_worst;
        aggregate.actual_at_worst = current.actual_at_worst;
    }
}

BackendValidation validate_backend(
    warpforge::MiniInferBlock& block,
    const warpforge::MiniInferFixture& fixture,
    warpforge::TensorView& input,
    warpforge::TensorView& output,
    const warpforge::GemmBackend backend,
    const cublasHandle_t cublas_handle,
    warpforge::CudaStream& stream) {
    BackendValidation result;
    result.aggregate.passed = true;
    std::set<std::string> observed;
    std::size_t index_offset = 0U;
    const auto observer = [&](
                              const std::string_view name,
                              const warpforge::TensorView& view,
                              const cudaStream_t observation_stream) {
        const auto expected = fixture.intermediates.find(std::string(name));
        if (expected == fixture.intermediates.end()) {
            throw std::runtime_error("unexpected MiniInfer intermediate: " + std::string(name));
        }
        std::vector<float> actual(view.element_count());
        CUDA_CHECK(cudaMemcpyAsync(
            actual.data(),
            view.data_as<float>(),
            actual.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            observation_stream));
        CUDA_CHECK(cudaStreamSynchronize(observation_stream));
        const warpforge::Tolerance tolerance =
            warpforge::miniinfer_intermediate_tolerance(name, fixture.config);
        const warpforge::ValidationResult current = warpforge::validate_fp32(
            expected->second.data(),
            actual.data(),
            actual.size(),
            tolerance);
        if (!current.passed) {
            throw std::runtime_error(
                std::string(warpforge::gemm_backend_name(backend)) + " backend " +
                std::string(name) + " failed at index " +
                std::to_string(current.worst_index) + ", max absolute error " +
                std::to_string(current.max_absolute_error));
        }
        merge_validation(result.aggregate, current, index_offset);
        index_offset += current.element_count;
        observed.emplace(name);
        result.intermediates.push_back(
            {warpforge::gemm_backend_name(backend), std::string(name), tolerance, current});
    };
    block.forward(
        input,
        output,
        backend,
        cublas_handle,
        stream.native_handle(),
        observer);
    stream.synchronize();
    if (observed.size() != warpforge::miniinfer_intermediate_names().size()) {
        throw std::runtime_error("not every MiniInfer intermediate was observed");
    }
    return result;
}

std::vector<double> measure_end_to_end(
    const warpforge::BenchmarkConfig& config,
    warpforge::MiniInferBlock& block,
    const warpforge::MiniInferFixture& fixture,
    warpforge::Tensor& input,
    warpforge::Tensor& output,
    warpforge::TensorView& input_view,
    warpforge::TensorView& output_view,
    const warpforge::GemmBackend backend,
    const cublasHandle_t cublas_handle,
    warpforge::CudaStream& stream) {
    std::vector<float> host_output(fixture.input.size());
    const auto run_once = [&] {
        input.copy_from_host_async(
            fixture.input.data(), fixture.input.size() * sizeof(float), stream.native_handle());
        block.forward(
            input_view,
            output_view,
            backend,
            cublas_handle,
            stream.native_handle());
        output.copy_to_host_async(
            host_output.data(), host_output.size() * sizeof(float), stream.native_handle());
        stream.synchronize();
    };
    for (std::size_t iteration = 0U; iteration < config.warmup_iterations; ++iteration) {
        run_once();
    }
    std::vector<double> samples;
    samples.reserve(config.measurement_iterations);
    for (std::size_t iteration = 0U; iteration < config.measurement_iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        run_once();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return samples;
}

warpforge::BenchmarkResult make_result(
    const Options& options,
    const warpforge::MiniInferFixture& fixture,
    const warpforge::MiniInferBlock& block,
    const warpforge::GemmBackend backend,
    std::string timing_scope,
    std::vector<double> samples,
    const warpforge::ValidationResult& validation) {
    const std::string implementation = warpforge::gemm_backend_name(backend);
    auto metadata = warpforge::make_benchmark_metadata(
        "miniinfer_block",
        implementation,
        "fp32",
        {{"batch", fixture.config.batch},
         {"sequence", fixture.config.sequence},
         {"hidden_size", fixture.config.hidden_size},
         {"attention_heads", fixture.config.attention_heads},
         {"head_dimension", fixture.config.head_dimension},
         {"intermediate_size", fixture.config.intermediate_size}},
        {});
    metadata.timing_scope = std::move(timing_scope);
    const auto statistics = warpforge::summarize_samples(samples);
    const double tokens_per_second =
        static_cast<double>(warpforge::miniinfer_token_count(fixture.config)) * 1000.0 /
        statistics.median_ms;
    return {
        1,
        options.benchmark,
        std::move(metadata),
        statistics,
        validation,
        {{"tokens_per_second_from_median", tokens_per_second},
         {"parameter_bytes", static_cast<double>(block.weights().parameter_bytes())},
         {"workspace_capacity_bytes", static_cast<double>(block.workspace().capacity_bytes())},
         {"workspace_used_bytes", static_cast<double>(block.workspace().used_bytes())},
         {"validated_intermediate_count",
          static_cast<double>(warpforge::miniinfer_intermediate_names().size())}},
        std::move(samples)};
}

void write_results(
    const std::filesystem::path& directory,
    const std::vector<RecordedResult>& records,
    const std::vector<IntermediateValidationRecord>& validations) {
    std::filesystem::create_directories(directory);
    for (const auto& record : records) {
        warpforge::write_benchmark_json(record.result, directory / record.filename);
    }
    std::ofstream summary(directory / "summary.csv", std::ios::binary | std::ios::trunc);
    if (!summary) {
        throw std::runtime_error("failed to create MiniInfer summary CSV");
    }
    summary << "schema_version,file,backend,timing_scope,median_ms,p95_ms,tokens_per_second,"
               "max_absolute_error,validation_passed,git_commit\n";
    summary << std::setprecision(17);
    for (const auto& record : records) {
        const auto& result = record.result;
        summary << result.schema_version << ','
                << record.filename.substr(0U, record.filename.size() - 5U) << ','
                << result.metadata.implementation << ','
                << result.metadata.timing_scope << ','
                << result.statistics.median_ms << ','
                << result.statistics.p95_ms << ','
                << result.metrics.at("tokens_per_second_from_median") << ','
                << result.validation.max_absolute_error << ','
                << (result.validation.passed ? "true" : "false") << ','
                << result.metadata.git_commit << '\n';
    }

    std::ofstream validation_output(
        directory / "validation.csv", std::ios::binary | std::ios::trunc);
    if (!validation_output) {
        throw std::runtime_error("failed to create MiniInfer validation CSV");
    }
    validation_output
        << "backend,name,element_count,absolute_tolerance,relative_tolerance,"
           "max_absolute_error,mean_absolute_error,max_relative_error,failure_count,passed\n";
    validation_output << std::setprecision(17);
    for (const auto& record : validations) {
        validation_output << record.backend << ',' << record.name << ','
                          << record.validation.element_count << ','
                          << record.tolerance.absolute << ','
                          << record.tolerance.relative << ','
                          << record.validation.max_absolute_error << ','
                          << record.validation.mean_absolute_error << ','
                          << record.validation.max_relative_error << ','
                          << record.validation.failure_count << ','
                          << (record.validation.passed ? "true" : "false") << '\n';
    }
}

}  // namespace

int main(const int argument_count, char** arguments) {
    try {
        const Options options = parse_options(argument_count, arguments);
        if (options.show_help) {
            print_usage();
            return EXIT_SUCCESS;
        }
        warpforge::MiniInferFixture fixture =
            warpforge::load_miniinfer_fixture(options.fixture_directory);
        if (fixture.seed != options.benchmark.seed) {
            throw std::runtime_error("MiniInfer fixture seed does not match benchmark seed 2027");
        }
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
        input.copy_from_host_async(
            fixture.input.data(), fixture.input.size() * sizeof(float), stream.native_handle());
        stream.synchronize();
        auto input_view = input.view();
        auto output_view = output.view();

        std::vector<RecordedResult> records;
        std::vector<IntermediateValidationRecord> validations;
        for (const warpforge::GemmBackend backend : selected_backends(options.backend)) {
            const cublasHandle_t handle =
                backend == warpforge::GemmBackend::cublas ? cublas.get() : nullptr;
            BackendValidation validation = validate_backend(
                block, fixture, input_view, output_view, backend, handle, stream);
            validations.insert(
                validations.end(),
                validation.intermediates.begin(),
                validation.intermediates.end());
            const auto launch = [&](const cudaStream_t launch_stream) {
                block.forward(
                    input_view, output_view, backend, handle, launch_stream);
            };
            auto kernel_samples = warpforge::measure_cuda_kernel(
                options.benchmark, stream.native_handle(), launch);
            auto end_to_end_samples = measure_end_to_end(
                options.benchmark,
                block,
                fixture,
                input,
                output,
                input_view,
                output_view,
                backend,
                handle,
                stream);

            const std::string backend_name = warpforge::gemm_backend_name(backend);
            records.push_back({
                "miniinfer_" + backend_name + "_kernel.json",
                make_result(
                    options,
                    fixture,
                    block,
                    backend,
                    "kernel-sequence-device-resident",
                    std::move(kernel_samples),
                    validation.aggregate)});
            records.push_back({
                "miniinfer_" + backend_name + "_end_to_end.json",
                make_result(
                    options,
                    fixture,
                    block,
                    backend,
                    "end-to-end-h2d-forward-d2h",
                    std::move(end_to_end_samples),
                    validation.aggregate)});
        }
        write_results(options.output_directory, records, validations);
        for (const auto& record : records) {
            std::cout << record.result.metadata.implementation << ' '
                      << record.result.metadata.timing_scope << ": median "
                      << record.result.statistics.median_ms << " ms, max error "
                      << record.result.validation.max_absolute_error << '\n';
        }
        std::cout << "MiniInfer validation: PASS\nResults: "
                  << options.output_directory.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "MiniInfer failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
