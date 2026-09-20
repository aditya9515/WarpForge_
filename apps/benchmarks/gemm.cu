#include <warpforge/benchmark.hpp>
#include <warpforge/cuda_check.cuh>
#include <warpforge/gemm.cuh>
#include <warpforge/validation.hpp>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct CaseDefinition final {
    const char* name;
    bool fp16_input;
    warpforge::GemmDispatch dispatch;
};

constexpr std::array cases{
    CaseDefinition{
        "cublas_fp32",
        false,
        {warpforge::GemmBackend::cublas, warpforge::GemmVariant::naive_fp32}},
    CaseDefinition{
        "custom_naive_fp32",
        false,
        {warpforge::GemmBackend::custom, warpforge::GemmVariant::naive_fp32}},
    CaseDefinition{
        "custom_tiled_fp32",
        false,
        {warpforge::GemmBackend::custom, warpforge::GemmVariant::tiled_fp32}},
    CaseDefinition{
        "custom_coalesced_fp32",
        false,
        {warpforge::GemmBackend::custom, warpforge::GemmVariant::coalesced_fp32}},
    CaseDefinition{
        "custom_register_blocked_fp32",
        false,
        {warpforge::GemmBackend::custom, warpforge::GemmVariant::register_blocked_fp32}},
    CaseDefinition{
        "cublas_fp16_fp32",
        true,
        {warpforge::GemmBackend::cublas, warpforge::GemmVariant::tiled_fp16_fp32}},
    CaseDefinition{
        "custom_tiled_fp16_fp32",
        true,
        {warpforge::GemmBackend::custom, warpforge::GemmVariant::tiled_fp16_fp32}},
    CaseDefinition{
        "custom_wmma_fp16_fp32",
        true,
        {warpforge::GemmBackend::custom, warpforge::GemmVariant::wmma_fp16_fp32}},
};

struct Options final {
    std::vector<std::size_t> sizes{256U, 512U, 1024U, 2048U};
    warpforge::BenchmarkConfig benchmark{};
    std::filesystem::path output_directory{"benchmarks/results/stage5"};
    std::string implementation{"all"};
    bool profile_only{};
    bool show_help{};
};

struct RecordedResult final {
    std::string filename;
    bool fp16_input{};
    warpforge::BenchmarkResult result;
};

template <typename T>
class LocalDeviceBuffer final {
public:
    explicit LocalDeviceBuffer(const std::size_t count) {
        if (count > 0U) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        }
    }

    ~LocalDeviceBuffer() noexcept {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    LocalDeviceBuffer(const LocalDeviceBuffer&) = delete;
    LocalDeviceBuffer& operator=(const LocalDeviceBuffer&) = delete;

    [[nodiscard]] T* get() const noexcept {
        return pointer_;
    }

private:
    T* pointer_{};
};

class LocalStream final {
public:
    LocalStream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~LocalStream() noexcept {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    LocalStream(const LocalStream&) = delete;
    LocalStream& operator=(const LocalStream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept {
        return stream_;
    }

private:
    cudaStream_t stream_{};
};

class LocalCublasHandle final {
public:
    LocalCublasHandle() {
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
        if (cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasSetMathMode failed");
        }
    }

    ~LocalCublasHandle() noexcept {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    LocalCublasHandle(const LocalCublasHandle&) = delete;
    LocalCublasHandle& operator=(const LocalCublasHandle&) = delete;

    [[nodiscard]] cublasHandle_t get() const noexcept {
        return handle_;
    }

private:
    cublasHandle_t handle_{};
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

std::size_t parse_size(const std::string& text, const std::string_view option) {
    const std::uint64_t value = parse_unsigned(text, option);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::out_of_range(std::string(option) + " exceeds the host size range");
    }
    return static_cast<std::size_t>(value);
}

std::vector<std::size_t> parse_sizes(const std::string& text) {
    std::vector<std::size_t> parsed;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const std::size_t value = parse_size(token, "--sizes");
        if (value == 0U) {
            throw std::invalid_argument("--sizes values must be greater than zero");
        }
        parsed.push_back(value);
    }
    if (parsed.empty()) {
        throw std::invalid_argument("--sizes requires at least one value");
    }
    return parsed;
}

Options parse_options(const int argument_count, char** arguments) {
    Options options;
    for (int index = 1; index < argument_count; ++index) {
        const std::string_view option = arguments[index];
        if (option == "--help" || option == "-h") {
            options.show_help = true;
            continue;
        }
        if (option == "--profile-only") {
            options.profile_only = true;
            continue;
        }
        if (index + 1 >= argument_count) {
            throw std::invalid_argument(std::string(option) + " requires a value");
        }
        const std::string value = arguments[++index];
        if (option == "--sizes") {
            options.sizes = parse_sizes(value);
        } else if (option == "--size") {
            options.sizes = {parse_size(value, option)};
        } else if (option == "--warmups") {
            options.benchmark.warmup_iterations = parse_size(value, option);
        } else if (option == "--iterations") {
            options.benchmark.measurement_iterations = parse_size(value, option);
        } else if (option == "--seed") {
            options.benchmark.seed = parse_unsigned(value, option);
        } else if (option == "--output-dir") {
            options.output_directory = value;
        } else if (option == "--implementation") {
            options.implementation = value;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }
    return options;
}

void print_usage() {
    std::cout << "Usage: warpforge-benchmark-gemm [options]\n"
              << "  --sizes <csv>             Square sizes (default: 256,512,1024,2048)\n"
              << "  --size <value>            One square size\n"
              << "  --warmups <count>         Warmup operations (default: 10)\n"
              << "  --iterations <count>      Measured operations (default: 100)\n"
              << "  --seed <value>            Deterministic seed (default: 2027)\n"
              << "  --implementation <name>   all or one reported implementation\n"
              << "  --output-dir <path>       Result directory\n"
              << "  --profile-only            Validate and launch one selected implementation\n"
              << "  --help                    Show this message\n";
}

const CaseDefinition& find_case(const std::string& name) {
    const auto iterator = std::find_if(
        cases.begin(), cases.end(), [&name](const CaseDefinition& candidate) {
            return name == candidate.name;
        });
    if (iterator == cases.end()) {
        throw std::invalid_argument("unknown GEMM implementation: " + name);
    }
    return *iterator;
}

void validate_options(const Options& options) {
    if (options.sizes.empty() ||
        std::any_of(options.sizes.begin(), options.sizes.end(), [](const std::size_t size) {
            return size == 0U;
        })) {
        throw std::invalid_argument("GEMM sizes must be greater than zero");
    }
    if (options.benchmark.measurement_iterations == 0U) {
        throw std::invalid_argument("--iterations must be greater than zero");
    }
    if (options.implementation != "all") {
        static_cast<void>(find_case(options.implementation));
    }
    if (!options.profile_only && options.implementation != "all") {
        throw std::invalid_argument(
            "--implementation selects one kernel only with --profile-only; report runs require all matched baselines");
    }
    if (options.profile_only &&
        (options.implementation == "all" || options.sizes.size() != 1U)) {
        throw std::invalid_argument(
            "--profile-only requires one --size and one --implementation");
    }
    for (const std::size_t size : options.sizes) {
        if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::out_of_range("GEMM size exceeds the supported 32-bit range");
        }
        const warpforge::GemmProblem problem{size, size, size};
        static_cast<void>(warpforge::gemm_a_elements(problem));
        static_cast<void>(warpforge::gemm_b_elements(problem));
        static_cast<void>(warpforge::gemm_c_elements(problem));
    }
}

warpforge::LaunchConfiguration launch_configuration(
    const warpforge::GemmProblem& problem,
    const CaseDefinition& definition) {
    if (definition.dispatch.backend == warpforge::GemmBackend::cublas) {
        return {};
    }
    switch (definition.dispatch.variant) {
        case warpforge::GemmVariant::naive_fp32:
        case warpforge::GemmVariant::tiled_fp32:
        case warpforge::GemmVariant::tiled_fp16_fp32:
            return {
                {static_cast<unsigned int>((problem.n + 15U) / 16U),
                 static_cast<unsigned int>((problem.m + 15U) / 16U),
                 1U},
                {16U, 16U, 1U},
                0U};
        case warpforge::GemmVariant::coalesced_fp32:
            return {
                {static_cast<unsigned int>((problem.n + 31U) / 32U),
                 static_cast<unsigned int>((problem.m + 31U) / 32U),
                 1U},
                {32U, 8U, 1U},
                0U};
        case warpforge::GemmVariant::register_blocked_fp32:
            return {
                {static_cast<unsigned int>((problem.n + 63U) / 64U),
                 static_cast<unsigned int>((problem.m + 63U) / 64U),
                 1U},
                {16U, 16U, 1U},
                0U};
        case warpforge::GemmVariant::wmma_fp16_fp32:
            return {
                {static_cast<unsigned int>(problem.n / 16U),
                 static_cast<unsigned int>(problem.m / 16U),
                 1U},
                {32U, 1U, 1U},
                0U};
    }
    return {};
}

warpforge::ValidationResult trusted_baseline_validation(
    const std::vector<float>& baseline) {
    return warpforge::validate_fp32(
        baseline.data(), baseline.data(), baseline.size(), {0.0, 0.0});
}

double static_shared_memory_bytes(const CaseDefinition& definition) {
    if (definition.dispatch.backend == warpforge::GemmBackend::cublas) {
        return 0.0;
    }
    switch (definition.dispatch.variant) {
        case warpforge::GemmVariant::tiled_fp32:
        case warpforge::GemmVariant::tiled_fp16_fp32:
            return static_cast<double>(2U * 16U * 16U * sizeof(float));
        case warpforge::GemmVariant::coalesced_fp32:
            return static_cast<double>(2U * 32U * 33U * sizeof(float));
        case warpforge::GemmVariant::register_blocked_fp32:
            return static_cast<double>((64U * 17U + 16U * 65U) * sizeof(float));
        default:
            return 0.0;
    }
}

RecordedResult benchmark_case(
    const CaseDefinition& definition,
    const warpforge::GemmProblem& problem,
    const Options& options,
    const LocalDeviceBuffer<float>& device_a_fp32,
    const LocalDeviceBuffer<float>& device_b_fp32,
    const LocalDeviceBuffer<__half>& device_a_fp16,
    const LocalDeviceBuffer<__half>& device_b_fp16,
    LocalDeviceBuffer<float>& device_c,
    const std::vector<float>& reference,
    const std::vector<float>* cpu_reference,
    const cublasHandle_t handle,
    const cudaStream_t stream) {
    auto launch = [&](const cudaStream_t launch_stream) {
        if (definition.fp16_input) {
            warpforge::gemm_fp16_cuda(
                device_a_fp16.get(),
                device_b_fp16.get(),
                device_c.get(),
                problem,
                definition.dispatch,
                handle,
                launch_stream);
        } else {
            warpforge::gemm_fp32_cuda(
                device_a_fp32.get(),
                device_b_fp32.get(),
                device_c.get(),
                problem,
                definition.dispatch,
                handle,
                launch_stream);
        }
    };

    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> actual(warpforge::gemm_c_elements(problem));
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_c.get(),
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    const warpforge::Tolerance tolerance = definition.fp16_input
        ? warpforge::gemm_fp16_tolerance(problem.k)
        : warpforge::gemm_fp32_tolerance(problem.k);
    warpforge::ValidationResult validation;
    if (definition.dispatch.backend == warpforge::GemmBackend::cublas) {
        validation = cpu_reference == nullptr
            ? trusted_baseline_validation(actual)
            : warpforge::validate_fp32(
                  cpu_reference->data(), actual.data(), actual.size(), tolerance);
    } else {
        validation = warpforge::validate_fp32(
            reference.data(), actual.data(), actual.size(), tolerance);
    }
    if (!validation.passed) {
        throw std::runtime_error(
            std::string(definition.name) + " failed validation for size " +
            std::to_string(problem.m) + " at index " +
            std::to_string(validation.worst_index));
    }

    auto samples = warpforge::measure_cuda_kernel(options.benchmark, stream, launch);
    const auto statistics = warpforge::summarize_samples(samples);
    const double gflops = warpforge::gemm_flop_count(problem) /
                          (statistics.median_ms * 1.0e6);
    auto metadata = warpforge::make_benchmark_metadata(
        definition.fp16_input ? "gemm_fp16_fp32" : "gemm_fp32",
        definition.name,
        definition.fp16_input ? "fp16_input_fp32_accumulation" : "fp32",
        {{"m", problem.m}, {"n", problem.n}, {"k", problem.k}},
        launch_configuration(problem, definition));
    warpforge::BenchmarkResult result{
        1,
        options.benchmark,
        std::move(metadata),
        statistics,
        validation,
        {{"gflops_from_median", gflops},
         {"flop_count", warpforge::gemm_flop_count(problem)},
         {"absolute_tolerance", tolerance.absolute},
         {"relative_tolerance", tolerance.relative},
         {"static_shared_memory_bytes", static_shared_memory_bytes(definition)},
         {"validated_against_cpu", cpu_reference == nullptr ? 0.0 : 1.0},
         {"validated_against_cublas",
          definition.dispatch.backend == warpforge::GemmBackend::custom ? 1.0 : 0.0}},
        std::move(samples)};
    const std::string filename =
        "gemm_" + std::to_string(problem.m) + "_" + definition.name + ".json";
    return {filename, definition.fp16_input, std::move(result)};
}

void add_library_relative_metrics(std::vector<RecordedResult>& records) {
    for (auto& record : records) {
        const std::uint64_t size = record.result.metadata.dimensions.at("m");
        const std::string baseline_name =
            record.fp16_input ? "cublas_fp16_fp32" : "cublas_fp32";
        const auto baseline = std::find_if(
            records.begin(), records.end(), [&](const RecordedResult& candidate) {
                return candidate.result.metadata.dimensions.at("m") == size &&
                       candidate.result.metadata.implementation == baseline_name;
            });
        if (baseline == records.end()) {
            throw std::runtime_error("matched cuBLAS result is missing");
        }
        const double baseline_median = baseline->result.statistics.median_ms;
        record.result.metrics["cublas_median_ms"] = baseline_median;
        record.result.metrics["percent_of_cublas"] =
            100.0 * baseline_median / record.result.statistics.median_ms;
    }
}

void write_results(
    const std::filesystem::path& directory,
    const std::vector<RecordedResult>& records) {
    std::filesystem::create_directories(directory);
    for (const auto& record : records) {
        warpforge::write_benchmark_json(record.result, directory / record.filename);
    }
    std::ofstream summary(directory / "summary.csv", std::ios::binary | std::ios::trunc);
    if (!summary) {
        throw std::runtime_error("failed to create GEMM summary CSV");
    }
    summary << "schema_version,file,m,n,k,data_type,implementation,median_ms,p95_ms,gflops,"
               "percent_of_cublas,max_absolute_error,validation_passed,git_commit\n";
    summary << std::setprecision(17);
    for (const auto& record : records) {
        const auto& result = record.result;
        summary << result.schema_version << ','
                << record.filename.substr(0U, record.filename.size() - 5U) << ','
                << result.metadata.dimensions.at("m") << ','
                << result.metadata.dimensions.at("n") << ','
                << result.metadata.dimensions.at("k") << ','
                << result.metadata.data_type << ','
                << result.metadata.implementation << ','
                << result.statistics.median_ms << ','
                << result.statistics.p95_ms << ','
                << result.metrics.at("gflops_from_median") << ','
                << result.metrics.at("percent_of_cublas") << ','
                << result.validation.max_absolute_error << ','
                << (result.validation.passed ? "true" : "false") << ','
                << result.metadata.git_commit << '\n';
    }
}

void check_memory_capacity(const warpforge::GemmProblem& problem) {
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t square = warpforge::gemm_c_elements(problem);
    const std::size_t required = square * (5U * sizeof(float) + 2U * sizeof(__half));
    if (required > free_bytes || free_bytes - required < 64U * 1024U * 1024U) {
        throw std::runtime_error(
            "GEMM size " + std::to_string(problem.m) +
            " does not leave the required 64 MiB device-memory reserve");
    }
}

std::vector<float> cpu_reference_for(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const warpforge::GemmProblem& problem) {
    std::vector<float> output(warpforge::gemm_c_elements(problem));
    warpforge::gemm_cpu_fp32(a.data(), b.data(), output.data(), problem);
    return output;
}

}  // namespace

int main(int argument_count, char** arguments) {
    try {
        const Options options = parse_options(argument_count, arguments);
        if (options.show_help) {
            print_usage();
            return EXIT_SUCCESS;
        }
        validate_options(options);
        CUDA_CHECK(cudaSetDevice(0));
        LocalStream stream;
        LocalCublasHandle handle;
        std::vector<RecordedResult> records;

        for (const std::size_t size : options.sizes) {
            const warpforge::GemmProblem problem{size, size, size};
            check_memory_capacity(problem);
            const std::size_t matrix_elements = warpforge::gemm_c_elements(problem);
            std::vector<float> host_a(matrix_elements);
            std::vector<float> host_b(matrix_elements);
            std::vector<__half> host_a_fp16(matrix_elements);
            std::vector<__half> host_b_fp16(matrix_elements);
            std::vector<float> quantized_a(matrix_elements);
            std::vector<float> quantized_b(matrix_elements);
            std::mt19937_64 generator(options.benchmark.seed + size);
            std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
            for (std::size_t index = 0; index < matrix_elements; ++index) {
                host_a[index] = distribution(generator);
                host_b[index] = distribution(generator);
                host_a_fp16[index] = __float2half(host_a[index]);
                host_b_fp16[index] = __float2half(host_b[index]);
                quantized_a[index] = __half2float(host_a_fp16[index]);
                quantized_b[index] = __half2float(host_b_fp16[index]);
            }

            LocalDeviceBuffer<float> device_a_fp32(matrix_elements);
            LocalDeviceBuffer<float> device_b_fp32(matrix_elements);
            LocalDeviceBuffer<__half> device_a_fp16(matrix_elements);
            LocalDeviceBuffer<__half> device_b_fp16(matrix_elements);
            LocalDeviceBuffer<float> device_c(matrix_elements);
            CUDA_CHECK(cudaMemcpyAsync(
                device_a_fp32.get(),
                host_a.data(),
                matrix_elements * sizeof(float),
                cudaMemcpyHostToDevice,
                stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(
                device_b_fp32.get(),
                host_b.data(),
                matrix_elements * sizeof(float),
                cudaMemcpyHostToDevice,
                stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(
                device_a_fp16.get(),
                host_a_fp16.data(),
                matrix_elements * sizeof(__half),
                cudaMemcpyHostToDevice,
                stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(
                device_b_fp16.get(),
                host_b_fp16.data(),
                matrix_elements * sizeof(__half),
                cudaMemcpyHostToDevice,
                stream.get()));
            CUDA_CHECK(cudaStreamSynchronize(stream.get()));

            std::vector<float> fp32_reference(matrix_elements);
            std::vector<float> fp16_reference(matrix_elements);
            warpforge::gemm_fp32_cuda(
                device_a_fp32.get(),
                device_b_fp32.get(),
                device_c.get(),
                problem,
                {warpforge::GemmBackend::cublas, warpforge::GemmVariant::naive_fp32},
                handle.get(),
                stream.get());
            CUDA_CHECK(cudaStreamSynchronize(stream.get()));
            CUDA_CHECK(cudaMemcpy(
                fp32_reference.data(),
                device_c.get(),
                matrix_elements * sizeof(float),
                cudaMemcpyDeviceToHost));
            warpforge::gemm_fp16_cuda(
                device_a_fp16.get(),
                device_b_fp16.get(),
                device_c.get(),
                problem,
                {warpforge::GemmBackend::cublas, warpforge::GemmVariant::tiled_fp16_fp32},
                handle.get(),
                stream.get());
            CUDA_CHECK(cudaStreamSynchronize(stream.get()));
            CUDA_CHECK(cudaMemcpy(
                fp16_reference.data(),
                device_c.get(),
                matrix_elements * sizeof(float),
                cudaMemcpyDeviceToHost));

            std::vector<float> cpu_fp32;
            std::vector<float> cpu_fp16;
            if (size <= 256U) {
                cpu_fp32 = cpu_reference_for(host_a, host_b, problem);
                cpu_fp16 = cpu_reference_for(quantized_a, quantized_b, problem);
            }

            const auto run_definition = [&](const CaseDefinition& definition) {
                const std::vector<float>& reference =
                    definition.fp16_input ? fp16_reference : fp32_reference;
                const std::vector<float>* cpu_reference = nullptr;
                if (definition.dispatch.backend == warpforge::GemmBackend::cublas &&
                    size <= 256U) {
                    cpu_reference = definition.fp16_input ? &cpu_fp16 : &cpu_fp32;
                }
                return benchmark_case(
                    definition,
                    problem,
                    options,
                    device_a_fp32,
                    device_b_fp32,
                    device_a_fp16,
                    device_b_fp16,
                    device_c,
                    reference,
                    cpu_reference,
                    handle.get(),
                    stream.get());
            };

            if (options.profile_only) {
                const auto result = run_definition(find_case(options.implementation));
                std::cout << "Profile GEMM validation: PASS (" << result.result.metadata.implementation
                          << ", " << size << "x" << size << "x" << size << ")\n";
                return EXIT_SUCCESS;
            }

            for (const auto& definition : cases) {
                if (options.implementation == "all" ||
                    options.implementation == definition.name) {
                    records.push_back(run_definition(definition));
                    const auto& result = records.back().result;
                    std::cout << result.metadata.implementation << " " << size << ": median "
                              << result.statistics.median_ms << " ms, "
                              << result.metrics.at("gflops_from_median") << " GFLOP/s\n";
                }
            }
        }

        add_library_relative_metrics(records);
        write_results(options.output_directory, records);
        std::cout << "Stage 5 GEMM validation: PASS\n"
                  << "Results: " << options.output_directory.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 5 GEMM benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
