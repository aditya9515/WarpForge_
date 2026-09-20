#include <warpforge/benchmark.hpp>
#include <warpforge/cuda_check.cuh>
#include <warpforge/memory.cuh>
#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr float saxpy_alpha = 1.25F;
constexpr std::size_t pipeline_chunk_count = 8U;
constexpr std::size_t maximum_stride_elements = 1U << 20U;
constexpr std::array<unsigned int, 5> block_sizes{32U, 64U, 128U, 256U, 512U};
constexpr std::array<std::size_t, 6> strides{1U, 2U, 4U, 8U, 16U, 32U};

struct Options final {
    std::size_t element_count{1U << 24U};
    std::size_t rows{2048U};
    std::size_t columns{1536U};
    warpforge::BenchmarkConfig benchmark{};
    std::filesystem::path output_directory{"benchmarks/results/stage3"};
    bool profile_only{};
    bool show_help{};
};

struct RecordedResult final {
    std::string filename;
    warpforge::BenchmarkResult result;
};

template <typename T>
class LocalDeviceBuffer final {
public:
    explicit LocalDeviceBuffer(const std::size_t count) {
        if (count > 0) {
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

template <typename T>
class LocalPinnedBuffer final {
public:
    explicit LocalPinnedBuffer(const std::size_t count) : count_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        }
    }

    ~LocalPinnedBuffer() noexcept {
        if (pointer_ != nullptr) {
            cudaFreeHost(pointer_);
        }
    }

    LocalPinnedBuffer(const LocalPinnedBuffer&) = delete;
    LocalPinnedBuffer& operator=(const LocalPinnedBuffer&) = delete;

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

std::uint64_t parse_unsigned(const std::string& text, const std::string_view option) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(option) + " requires a non-negative integer");
    }
    std::size_t consumed = 0;
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
        if (option == "--size") {
            options.element_count = parse_size(value, option);
        } else if (option == "--rows") {
            options.rows = parse_size(value, option);
        } else if (option == "--columns") {
            options.columns = parse_size(value, option);
        } else if (option == "--warmups") {
            options.benchmark.warmup_iterations = parse_size(value, option);
        } else if (option == "--iterations") {
            options.benchmark.measurement_iterations = parse_size(value, option);
        } else if (option == "--seed") {
            options.benchmark.seed = parse_unsigned(value, option);
        } else if (option == "--output-dir") {
            options.output_directory = value;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }
    return options;
}

void print_usage() {
    std::cout << "Usage: warpforge-benchmark-memory [options]\n"
              << "  --size <elements>       1D workload size (default: 16777216)\n"
              << "  --rows <count>          Transpose rows (default: 2048)\n"
              << "  --columns <count>       Transpose columns (default: 1536)\n"
              << "  --warmups <count>       Warmup operations (default: 10)\n"
              << "  --iterations <count>    Measured operations (default: 100)\n"
              << "  --seed <value>          Deterministic input seed (default: 2027)\n"
              << "  --output-dir <path>     Result directory\n"
              << "  --profile-only          Run only one- and two-stream pipeline traces\n"
              << "  --help                  Show this message\n";
}

void validate_options(const Options& options) {
    if (options.element_count == 0 || options.rows == 0 || options.columns == 0) {
        throw std::invalid_argument("size, rows, and columns must be greater than zero");
    }
    if (options.benchmark.measurement_iterations == 0) {
        throw std::invalid_argument("iterations must be greater than zero");
    }
    if (options.element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::out_of_range("size exceeds the addressable FP32 allocation range");
    }
    if (options.rows > std::numeric_limits<std::size_t>::max() / options.columns ||
        options.rows * options.columns >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::out_of_range("matrix dimensions exceed the addressable FP32 range");
    }
}

std::vector<double> measure_host_operation(
    const warpforge::BenchmarkConfig& config,
    const std::function<void()>& work) {
    for (std::size_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
        work();
    }
    std::vector<double> samples_ms;
    samples_ms.reserve(config.measurement_iterations);
    for (std::size_t iteration = 0; iteration < config.measurement_iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        work();
        const auto stop = std::chrono::steady_clock::now();
        samples_ms.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return samples_ms;
}

double bandwidth_gbps(const double logical_bytes, const double latency_ms) {
    return latency_ms > 0.0 ? logical_bytes / (latency_ms * 1.0e6) : 0.0;
}

warpforge::BenchmarkResult make_result(
    const Options& options,
    std::string operation,
    std::string implementation,
    std::map<std::string, std::uint64_t> dimensions,
    const warpforge::LaunchConfiguration launch,
    const warpforge::ValidationResult validation,
    std::vector<double> samples_ms,
    const std::string& timing_scope) {
    warpforge::BenchmarkResult result;
    result.config = options.benchmark;
    result.metadata = warpforge::make_benchmark_metadata(
        std::move(operation),
        std::move(implementation),
        "fp32",
        std::move(dimensions),
        launch);
    result.metadata.timing_scope = timing_scope;
    result.statistics = warpforge::summarize_samples(samples_ms);
    result.validation = validation;
    result.samples_ms = std::move(samples_ms);
    return result;
}

void record_result(
    const Options& options,
    const std::string& filename,
    warpforge::BenchmarkResult result,
    std::vector<RecordedResult>& records) {
    if (!result.validation.passed) {
        throw std::runtime_error(filename + " failed validation");
    }
    warpforge::write_benchmark_json(
        result, options.output_directory / (filename + ".json"));
    records.push_back({filename, std::move(result)});
}

void write_summary(
    const Options& options,
    const std::vector<RecordedResult>& records) {
    std::filesystem::create_directories(options.output_directory);
    std::ofstream output(options.output_directory / "summary.csv", std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create Stage 3 CSV summary");
    }
    output << "schema_version,file,operation,implementation,timing_scope,median_ms,p95_ms,"
              "bandwidth_gbps,validation_passed,git_commit\n";
    output << std::setprecision(17);
    for (const auto& record : records) {
        const auto& result = record.result;
        const auto bandwidth = result.metrics.find("effective_bandwidth_gbps_from_median");
        output << result.schema_version << ',' << record.filename << ','
               << result.metadata.operation << ',' << result.metadata.implementation << ','
               << result.metadata.timing_scope << ',' << result.statistics.median_ms << ','
               << result.statistics.p95_ms << ','
               << (bandwidth == result.metrics.end() ? 0.0 : bandwidth->second) << ','
               << (result.validation.passed ? "true" : "false") << ','
               << result.metadata.git_commit << '\n';
    }
}

void fill_random(
    float* output,
    const std::size_t element_count,
    std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (std::size_t index = 0; index < element_count; ++index) {
        output[index] = distribution(generator);
    }
}

void run_saxpy_sweep(const Options& options, std::vector<RecordedResult>& records) {
    std::vector<float> input(options.element_count);
    std::vector<float> initial(options.element_count);
    std::vector<float> expected(options.element_count);
    std::vector<float> actual(options.element_count);
    std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed));
    fill_random(input.data(), input.size(), generator);
    fill_random(initial.data(), initial.size(), generator);
    expected = initial;
    warpforge::saxpy_cpu(saxpy_alpha, input.data(), expected.data(), expected.size());

    LocalDeviceBuffer<float> device_input(options.element_count);
    LocalDeviceBuffer<float> device_output(options.element_count);
    const std::size_t bytes = options.element_count * sizeof(float);
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice));

    for (const unsigned int block_size : block_sizes) {
        CUDA_CHECK(cudaMemcpy(device_output.get(), initial.data(), bytes, cudaMemcpyHostToDevice));
        warpforge::saxpy_cuda(
            saxpy_alpha,
            device_input.get(),
            device_output.get(),
            options.element_count,
            block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost));
        const auto validation = warpforge::validate_fp32(
            expected.data(), actual.data(), actual.size(), {1.0e-5, 1.0e-5});

        CUDA_CHECK(cudaMemcpy(device_output.get(), initial.data(), bytes, cudaMemcpyHostToDevice));
        auto samples = warpforge::measure_cuda_kernel(
            options.benchmark,
            nullptr,
            [&](cudaStream_t stream) {
                warpforge::saxpy_cuda(
                    saxpy_alpha,
                    device_input.get(),
                    device_output.get(),
                    options.element_count,
                    block_size,
                    stream);
            });
        warpforge::LaunchConfiguration launch;
        launch.grid = {warpforge::memory_grid_size(options.element_count, block_size), 1U, 1U};
        launch.block = {block_size, 1U, 1U};
        auto result = make_result(
            options,
            "saxpy",
            "cuda_block_" + std::to_string(block_size),
            {{"elements", static_cast<std::uint64_t>(options.element_count)}},
            launch,
            validation,
            std::move(samples),
            "kernel-only");
        result.metrics["effective_bandwidth_gbps_from_median"] = bandwidth_gbps(
            static_cast<double>(bytes) * 3.0, result.statistics.median_ms);
        record_result(
            options, "saxpy_block_" + std::to_string(block_size), std::move(result), records);
    }
}

void run_copy_sweep(const Options& options, std::vector<RecordedResult>& records) {
    std::vector<float> input(options.element_count);
    std::vector<float> actual(options.element_count);
    std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed));
    fill_random(input.data(), input.size(), generator);

    LocalDeviceBuffer<float> device_input(options.element_count);
    LocalDeviceBuffer<float> device_output(options.element_count);
    const std::size_t bytes = options.element_count * sizeof(float);
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice));

    for (const unsigned int block_size : block_sizes) {
        warpforge::memory_copy_cuda(
            device_input.get(), device_output.get(), options.element_count, block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost));
        const auto validation = warpforge::validate_fp32(
            input.data(), actual.data(), actual.size(), {0.0, 0.0});

        auto samples = warpforge::measure_cuda_kernel(
            options.benchmark,
            nullptr,
            [&](cudaStream_t stream) {
                warpforge::memory_copy_cuda(
                    device_input.get(),
                    device_output.get(),
                    options.element_count,
                    block_size,
                    stream);
            });
        warpforge::LaunchConfiguration launch;
        launch.grid = {warpforge::memory_grid_size(options.element_count, block_size), 1U, 1U};
        launch.block = {block_size, 1U, 1U};
        auto result = make_result(
            options,
            "memory_copy",
            "cuda_block_" + std::to_string(block_size),
            {{"elements", static_cast<std::uint64_t>(options.element_count)}},
            launch,
            validation,
            std::move(samples),
            "kernel-only");
        result.metrics["effective_bandwidth_gbps_from_median"] = bandwidth_gbps(
            static_cast<double>(bytes) * 2.0, result.statistics.median_ms);
        record_result(
            options, "copy_block_" + std::to_string(block_size), std::move(result), records);
    }
}

void run_stride_sweep(const Options& options, std::vector<RecordedResult>& records) {
    const std::size_t logical_elements =
        std::min(options.element_count, maximum_stride_elements);
    const std::size_t source_elements = (logical_elements - 1U) * strides.back() + 1U;
    std::vector<float> input(source_elements);
    std::vector<float> expected(logical_elements);
    std::vector<float> actual(logical_elements);
    std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed));
    fill_random(input.data(), input.size(), generator);

    LocalDeviceBuffer<float> device_input(source_elements);
    LocalDeviceBuffer<float> device_output(logical_elements);
    CUDA_CHECK(cudaMemcpy(
        device_input.get(),
        input.data(),
        source_elements * sizeof(float),
        cudaMemcpyHostToDevice));

    for (const std::size_t stride : strides) {
        warpforge::strided_copy_cpu(
            input.data(), expected.data(), logical_elements, stride);
        warpforge::strided_copy_cuda(
            device_input.get(), device_output.get(), logical_elements, stride);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            actual.data(),
            device_output.get(),
            logical_elements * sizeof(float),
            cudaMemcpyDeviceToHost));
        const auto validation = warpforge::validate_fp32(
            expected.data(), actual.data(), logical_elements, {0.0, 0.0});

        auto samples = warpforge::measure_cuda_kernel(
            options.benchmark,
            nullptr,
            [&](cudaStream_t stream) {
                warpforge::strided_copy_cuda(
                    device_input.get(),
                    device_output.get(),
                    logical_elements,
                    stride,
                    warpforge::memory_default_block_size,
                    stream);
            });
        warpforge::LaunchConfiguration launch;
        launch.grid = {
            warpforge::memory_grid_size(logical_elements), 1U, 1U};
        launch.block = {warpforge::memory_default_block_size, 1U, 1U};
        auto result = make_result(
            options,
            "strided_copy",
            "stride_" + std::to_string(stride),
            {{"elements", static_cast<std::uint64_t>(logical_elements)},
             {"source_span_elements",
              static_cast<std::uint64_t>((logical_elements - 1U) * stride + 1U)},
             {"stride", static_cast<std::uint64_t>(stride)}},
            launch,
            validation,
            std::move(samples),
            "kernel-only");
        result.metrics["effective_bandwidth_gbps_from_median"] = bandwidth_gbps(
            static_cast<double>(logical_elements * sizeof(float)) * 2.0,
            result.statistics.median_ms);
        record_result(
            options, "strided_stride_" + std::to_string(stride), std::move(result), records);
    }
}

void run_transpose(const Options& options, std::vector<RecordedResult>& records) {
    const std::size_t element_count = options.rows * options.columns;
    const std::size_t bytes = element_count * sizeof(float);
    std::vector<float> input(element_count);
    std::vector<float> expected(element_count);
    std::vector<float> actual(element_count);
    std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed));
    fill_random(input.data(), input.size(), generator);
    warpforge::transpose_cpu(
        input.data(), expected.data(), options.rows, options.columns);

    LocalDeviceBuffer<float> device_input(element_count);
    LocalDeviceBuffer<float> device_output(element_count);
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 naive_block{32U, 8U, 1U};
    const dim3 naive_grid =
        warpforge::transpose_grid_size(options.rows, options.columns, naive_block);
    const dim3 tiled_grid = warpforge::transpose_grid_size(
        options.rows,
        options.columns,
        dim3{warpforge::transpose_tile_dimension, warpforge::transpose_tile_dimension, 1U});
    for (const bool tiled : std::array<bool, 2>{false, true}) {
        if (tiled) {
            warpforge::transpose_tiled_cuda(
                device_input.get(), device_output.get(), options.rows, options.columns);
        } else {
            warpforge::transpose_naive_cuda(
                device_input.get(),
                device_output.get(),
                options.rows,
                options.columns,
                naive_block);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost));
        const auto validation = warpforge::validate_fp32(
            expected.data(), actual.data(), element_count, {0.0, 0.0});

        auto samples = warpforge::measure_cuda_kernel(
            options.benchmark,
            nullptr,
            [&](cudaStream_t stream) {
                if (tiled) {
                    warpforge::transpose_tiled_cuda(
                        device_input.get(),
                        device_output.get(),
                        options.rows,
                        options.columns,
                        stream);
                } else {
                    warpforge::transpose_naive_cuda(
                        device_input.get(),
                        device_output.get(),
                        options.rows,
                        options.columns,
                        naive_block,
                        stream);
                }
            });

        warpforge::LaunchConfiguration launch;
        if (tiled) {
            launch.grid = {tiled_grid.x, tiled_grid.y, tiled_grid.z};
            launch.block = {
                warpforge::transpose_tile_dimension, warpforge::transpose_block_rows, 1U};
        } else {
            launch.grid = {naive_grid.x, naive_grid.y, naive_grid.z};
            launch.block = {naive_block.x, naive_block.y, naive_block.z};
        }
        const std::string implementation = tiled ? "tiled_32x32" : "naive";
        auto result = make_result(
            options,
            "matrix_transpose",
            implementation,
            {{"rows", static_cast<std::uint64_t>(options.rows)},
             {"columns", static_cast<std::uint64_t>(options.columns)}},
            launch,
            validation,
            std::move(samples),
            "kernel-only");
        result.metrics["effective_bandwidth_gbps_from_median"] = bandwidth_gbps(
            static_cast<double>(bytes) * 2.0, result.statistics.median_ms);
        if (tiled) {
            result.metrics["static_shared_memory_bytes"] =
                static_cast<double>(warpforge::transpose_tile_dimension) *
                static_cast<double>(warpforge::transpose_tile_dimension + 1U) *
                sizeof(float);
        }
        record_result(options, "transpose_" + implementation, std::move(result), records);
    }
}

template <typename Work>
void run_transfer_variant(
    const Options& options,
    const std::string& implementation,
    float* expected,
    float* actual,
    const std::size_t element_count,
    Work&& work,
    std::vector<RecordedResult>& records) {
    work();
    const auto validation = warpforge::validate_fp32(
        expected, actual, element_count, {0.0, 0.0});
    auto samples = measure_host_operation(options.benchmark, work);
    const warpforge::LaunchConfiguration launch{};
    auto result = make_result(
        options,
        "transfer_roundtrip",
        implementation,
        {{"elements", static_cast<std::uint64_t>(element_count)}},
        launch,
        validation,
        std::move(samples),
        "end-to-end");
    result.metrics["effective_bandwidth_gbps_from_median"] = bandwidth_gbps(
        static_cast<double>(element_count * sizeof(float)) * 2.0,
        result.statistics.median_ms);
    record_result(options, "transfer_" + implementation, std::move(result), records);
}

void run_transfer_comparisons(const Options& options, std::vector<RecordedResult>& records) {
    std::vector<float> pageable_input(options.element_count);
    std::vector<float> pageable_output(options.element_count);
    LocalPinnedBuffer<float> pinned_input(options.element_count);
    LocalPinnedBuffer<float> pinned_output(options.element_count);
    LocalDeviceBuffer<float> device(options.element_count);
    LocalStream stream;
    std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed));
    fill_random(pageable_input.data(), pageable_input.size(), generator);
    std::copy(pageable_input.begin(), pageable_input.end(), pinned_input.get());
    const std::size_t bytes = options.element_count * sizeof(float);

    run_transfer_variant(
        options,
        "pageable_sync",
        pageable_input.data(),
        pageable_output.data(),
        options.element_count,
        [&]() {
            CUDA_CHECK(cudaMemcpy(
                device.get(), pageable_input.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                pageable_output.data(), device.get(), bytes, cudaMemcpyDeviceToHost));
        },
        records);
    run_transfer_variant(
        options,
        "pinned_sync",
        pinned_input.get(),
        pinned_output.get(),
        options.element_count,
        [&]() {
            CUDA_CHECK(cudaMemcpy(
                device.get(), pinned_input.get(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                pinned_output.get(), device.get(), bytes, cudaMemcpyDeviceToHost));
        },
        records);
    run_transfer_variant(
        options,
        "pageable_async",
        pageable_input.data(),
        pageable_output.data(),
        options.element_count,
        [&]() {
            CUDA_CHECK(cudaMemcpyAsync(
                device.get(),
                pageable_input.data(),
                bytes,
                cudaMemcpyHostToDevice,
                stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(
                pageable_output.data(),
                device.get(),
                bytes,
                cudaMemcpyDeviceToHost,
                stream.get()));
            CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        },
        records);
    run_transfer_variant(
        options,
        "pinned_async",
        pinned_input.get(),
        pinned_output.get(),
        options.element_count,
        [&]() {
            CUDA_CHECK(cudaMemcpyAsync(
                device.get(),
                pinned_input.get(),
                bytes,
                cudaMemcpyHostToDevice,
                stream.get()));
            CUDA_CHECK(cudaMemcpyAsync(
                pinned_output.get(),
                device.get(),
                bytes,
                cudaMemcpyDeviceToHost,
                stream.get()));
            CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        },
        records);
}

void run_pipeline_variant(
    const Options& options,
    const bool two_streams,
    std::vector<RecordedResult>& records) {
    const std::size_t chunk_count = std::min(pipeline_chunk_count, options.element_count);
    const std::size_t chunk_capacity =
        (options.element_count + chunk_count - 1U) / chunk_count;
    LocalPinnedBuffer<float> input(options.element_count);
    LocalPinnedBuffer<float> initial(options.element_count);
    LocalPinnedBuffer<float> output(options.element_count);
    std::vector<float> expected(options.element_count);
    std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed));
    fill_random(input.get(), input.size(), generator);
    fill_random(initial.get(), initial.size(), generator);
    std::copy(initial.get(), initial.get() + initial.size(), expected.begin());
    warpforge::saxpy_cpu(
        saxpy_alpha, input.get(), expected.data(), expected.size());

    std::array<LocalStream, 2> streams{};
    std::array<LocalDeviceBuffer<float>, 2> device_input{
        LocalDeviceBuffer<float>{chunk_capacity}, LocalDeviceBuffer<float>{chunk_capacity}};
    std::array<LocalDeviceBuffer<float>, 2> device_output{
        LocalDeviceBuffer<float>{chunk_capacity}, LocalDeviceBuffer<float>{chunk_capacity}};

    const auto execute = [&]() {
        for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
            const std::size_t stream_index = two_streams ? chunk % streams.size() : 0U;
            const std::size_t offset = chunk * chunk_capacity;
            const std::size_t count =
                std::min(chunk_capacity, options.element_count - offset);
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
                saxpy_alpha,
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
        CUDA_CHECK(cudaStreamSynchronize(streams[0].get()));
        if (two_streams) {
            CUDA_CHECK(cudaStreamSynchronize(streams[1].get()));
        }
    };

    execute();
    const auto validation = warpforge::validate_fp32(
        expected.data(), output.get(), expected.size(), {1.0e-5, 1.0e-5});
    auto samples = measure_host_operation(options.benchmark, execute);
    const std::string implementation = two_streams ? "two_stream_chunked" : "single_stream_chunked";
    const warpforge::LaunchConfiguration launch{};
    auto result = make_result(
        options,
        "saxpy_pipeline",
        implementation,
        {{"chunks", static_cast<std::uint64_t>(chunk_count)},
         {"elements", static_cast<std::uint64_t>(options.element_count)},
         {"streams", two_streams ? 2U : 1U}},
        launch,
        validation,
        std::move(samples),
        "end-to-end");
    result.metrics["effective_bandwidth_gbps_from_median"] = bandwidth_gbps(
        static_cast<double>(options.element_count * sizeof(float)) * 3.0,
        result.statistics.median_ms);
    record_result(options, "pipeline_" + implementation, std::move(result), records);
}

void print_results(const std::vector<RecordedResult>& records) {
    std::cout << std::left << std::setw(34) << "Experiment" << std::right
              << std::setw(14) << "Median ms" << std::setw(14) << "P95 ms"
              << std::setw(16) << "Logical GB/s" << '\n';
    std::cout << std::string(78, '-') << '\n';
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& record : records) {
        const auto bandwidth =
            record.result.metrics.find("effective_bandwidth_gbps_from_median");
        std::cout << std::left << std::setw(34) << record.filename << std::right
                  << std::setw(14) << record.result.statistics.median_ms
                  << std::setw(14) << record.result.statistics.p95_ms
                  << std::setw(16)
                  << (bandwidth == record.result.metrics.end() ? 0.0 : bandwidth->second)
                  << '\n';
    }
}

}  // namespace

int main(const int argument_count, char** arguments) {
    try {
        Options options = parse_options(argument_count, arguments);
        if (options.show_help) {
            print_usage();
            return EXIT_SUCCESS;
        }
        validate_options(options);

        std::vector<RecordedResult> records;
        if (options.profile_only) {
            options.benchmark.warmup_iterations = 0;
            options.benchmark.measurement_iterations = 1;
            run_pipeline_variant(options, false, records);
            run_pipeline_variant(options, true, records);
        } else {
            run_saxpy_sweep(options, records);
            run_copy_sweep(options, records);
            run_stride_sweep(options, records);
            run_transpose(options, records);
            run_transfer_comparisons(options, records);
            run_pipeline_variant(options, false, records);
            run_pipeline_variant(options, true, records);
        }
        write_summary(options, records);
        print_results(records);
        std::cout << "Stage 3 validation: PASS\n";
        std::cout << "Results: " << options.output_directory.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 3 memory benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
