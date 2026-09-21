#include <warpforge/benchmark.hpp>
#include <warpforge/cuda_check.cuh>
#include <warpforge/elementwise.cuh>
#include <warpforge/fusion.cuh>
#include <warpforge/rmsnorm.cuh>
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

enum class RunMode { all, fusion_only, pipeline_only };

struct Options final {
    std::size_t rows{4096U};
    std::size_t columns{1024U};
    std::size_t element_count{1U << 24U};
    std::size_t pipeline_element_count{1U << 24U};
    std::size_t chunk_count{8U};
    warpforge::BenchmarkConfig benchmark{};
    std::filesystem::path output_directory{"benchmarks/results/stage7"};
    RunMode mode{RunMode::all};
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
    [[nodiscard]] T* get() const noexcept { return pointer_; }

private:
    T* pointer_{};
};

template <typename T>
class LocalPinnedBuffer final {
public:
    explicit LocalPinnedBuffer(const std::size_t count) : count_(count) {
        if (count > 0U) {
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        }
    }
    ~LocalPinnedBuffer() noexcept {
        if (pointer_ != nullptr) {
            cudaFreeHost(pointer_);
        }
    }
    LocalPinnedBuffer(const LocalPinnedBuffer&) = delete;
    LocalPinnedBuffer& operator=(const LocalPinnedBuffer&) = delete;
    [[nodiscard]] T* get() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] T& operator[](const std::size_t index) noexcept {
        return pointer_[index];
    }

private:
    T* pointer_{};
    std::size_t count_{};
};

class LocalStream final {
public:
    LocalStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~LocalStream() noexcept {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    LocalStream(const LocalStream&) = delete;
    LocalStream& operator=(const LocalStream&) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_{};
};

class LocalEvent final {
public:
    LocalEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }
    ~LocalEvent() noexcept {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    LocalEvent(const LocalEvent&) = delete;
    LocalEvent& operator=(const LocalEvent&) = delete;
    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_{};
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

Options parse_options(const int argument_count, char** arguments) {
    Options options;
    for (int index = 1; index < argument_count; ++index) {
        const std::string_view option = arguments[index];
        if (option == "--help" || option == "-h") {
            options.show_help = true;
            continue;
        }
        if (option == "--fusion-only") {
            if (options.mode == RunMode::pipeline_only) {
                throw std::invalid_argument("--fusion-only and --pipeline-only are exclusive");
            }
            options.mode = RunMode::fusion_only;
            continue;
        }
        if (option == "--pipeline-only") {
            if (options.mode == RunMode::fusion_only) {
                throw std::invalid_argument("--fusion-only and --pipeline-only are exclusive");
            }
            options.mode = RunMode::pipeline_only;
            continue;
        }
        if (index + 1 >= argument_count) {
            throw std::invalid_argument(std::string(option) + " requires a value");
        }
        const std::string value = arguments[++index];
        if (option == "--rows") {
            options.rows = parse_size(value, option);
        } else if (option == "--columns") {
            options.columns = parse_size(value, option);
        } else if (option == "--elements") {
            options.element_count = parse_size(value, option);
        } else if (option == "--pipeline-elements") {
            options.pipeline_element_count = parse_size(value, option);
        } else if (option == "--chunks") {
            options.chunk_count = parse_size(value, option);
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
    std::cout << "Usage: warpforge-benchmark-fusion [options]\n"
              << "  --rows <count>               Residual RMSNorm rows (default: 4096)\n"
              << "  --columns <count>            Residual RMSNorm width (default: 1024)\n"
              << "  --elements <count>           SwiGLU elements (default: 16777216)\n"
              << "  --pipeline-elements <count>  Pinned pipeline elements (default: 16777216)\n"
              << "  --chunks <count>             Pipeline chunks (default: 8)\n"
              << "  --warmups <count>            Warmups per case (default: 10)\n"
              << "  --iterations <count>         Samples per case (default: 100)\n"
              << "  --seed <value>               Deterministic seed (default: 2027)\n"
              << "  --output-dir <path>          Result directory\n"
              << "  --fusion-only                Run only separate/fused kernel cases\n"
              << "  --pipeline-only              Run only single/two-stream pipelines\n"
              << "  --help                       Show this message\n";
}

void validate_options(const Options& options) {
    if (options.rows == 0U || options.columns == 0U || options.element_count == 0U ||
        options.pipeline_element_count == 0U || options.chunk_count == 0U) {
        throw std::invalid_argument("all workload dimensions and chunk count must be positive");
    }
    if (options.benchmark.measurement_iterations == 0U) {
        throw std::invalid_argument("--iterations must be greater than zero");
    }
    if (options.rows > std::numeric_limits<std::size_t>::max() / options.columns) {
        throw std::overflow_error("residual RMSNorm element count overflows size_t");
    }
    if (options.rows > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("residual RMSNorm row count exceeds the CUDA grid range");
    }
    constexpr std::size_t max_fp32_count =
        std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (options.rows * options.columns > max_fp32_count ||
        options.element_count > max_fp32_count ||
        options.pipeline_element_count > max_fp32_count) {
        throw std::overflow_error("benchmark allocation size overflows size_t");
    }
}

std::vector<float> random_values(
    const std::size_t count,
    const std::uint64_t seed,
    const float minimum,
    const float maximum) {
    std::mt19937 generator(static_cast<std::uint32_t>(seed));
    std::uniform_real_distribution<float> distribution(minimum, maximum);
    std::vector<float> values(count);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

double bandwidth_gbps(const double bytes, const double latency_ms) {
    return latency_ms > 0.0 ? bytes / (latency_ms * 1.0e6) : 0.0;
}

unsigned int checked_grid_size(
    const std::size_t element_count,
    const unsigned int block_size) {
    const std::size_t grid = (element_count + block_size - 1U) / block_size;
    if (grid > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("benchmark grid exceeds the CUDA x-dimension range");
    }
    return static_cast<unsigned int>(grid);
}

std::vector<double> measure_host_operation(
    const warpforge::BenchmarkConfig& config,
    const std::function<void()>& work) {
    for (std::size_t iteration = 0U; iteration < config.warmup_iterations; ++iteration) {
        work();
    }
    std::vector<double> samples;
    samples.reserve(config.measurement_iterations);
    for (std::size_t iteration = 0U;
         iteration < config.measurement_iterations;
         ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        work();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return samples;
}

RecordedResult measure_cuda_case(
    const std::string& filename,
    warpforge::BenchmarkMetadata metadata,
    const warpforge::BenchmarkConfig& config,
    const warpforge::Tolerance tolerance,
    const std::vector<float>& expected,
    float* device_output,
    const std::size_t logical_reads,
    const std::size_t logical_writes,
    const warpforge::CudaWork& work,
    const cudaStream_t stream) {
    work(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_output,
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    const auto validation = warpforge::validate_fp32(
        expected.data(), actual.data(), expected.size(), tolerance);
    if (!validation.passed) {
        throw std::runtime_error(
            metadata.operation + "/" + metadata.implementation +
            " failed validation at index " + std::to_string(validation.worst_index));
    }
    auto samples = warpforge::measure_cuda_kernel(config, stream, work);
    const auto statistics = warpforge::summarize_samples(samples);
    const double logical_bytes = static_cast<double>(expected.size()) * sizeof(float) *
                                 static_cast<double>(logical_reads + logical_writes);
    warpforge::BenchmarkResult result{
        1,
        config,
        std::move(metadata),
        statistics,
        validation,
        {{"absolute_tolerance", tolerance.absolute},
         {"effective_bandwidth_gbps_from_median",
          bandwidth_gbps(logical_bytes, statistics.median_ms)},
         {"logical_bytes", logical_bytes},
         {"logical_reads_per_element", static_cast<double>(logical_reads)},
         {"logical_writes_per_element", static_cast<double>(logical_writes)},
         {"relative_tolerance", tolerance.relative}},
        std::move(samples)};
    return {filename, std::move(result)};
}

void benchmark_residual_rmsnorm(
    const Options& options,
    std::vector<RecordedResult>& records) {
    const std::size_t count = options.rows * options.columns;
    const auto input = random_values(count, options.benchmark.seed, -3.0F, 3.0F);
    const auto residual = random_values(count, options.benchmark.seed + 1U, -3.0F, 3.0F);
    const auto weight = random_values(
        options.columns, options.benchmark.seed + 2U, 0.5F, 1.5F);
    std::vector<float> expected(count);
    constexpr float epsilon = 1.0e-5F;
    warpforge::residual_rmsnorm_cpu(
        input.data(), residual.data(), weight.data(), expected.data(),
        options.rows, options.columns, epsilon);
    LocalDeviceBuffer<float> device_input(count);
    LocalDeviceBuffer<float> device_residual(count);
    LocalDeviceBuffer<float> device_weight(options.columns);
    LocalDeviceBuffer<float> device_intermediate(count);
    LocalDeviceBuffer<float> device_output(count);
    LocalStream stream;
    CUDA_CHECK(cudaMemcpy(
        device_input.get(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_residual.get(), residual.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_weight.get(),
        weight.data(),
        weight.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    const warpforge::LaunchConfiguration launch{
        {static_cast<unsigned int>(options.rows), 1U, 1U},
        {warpforge::fusion_default_block_size, 1U, 1U},
        warpforge::residual_rmsnorm_dynamic_shared_memory_bytes()};
    auto separate_metadata = warpforge::make_benchmark_metadata(
        "residual_rmsnorm",
        "separate",
        "fp32",
        {{"columns", options.columns}, {"kernel_launches", 2U}, {"rows", options.rows}},
        launch);
    records.push_back(measure_cuda_case(
        "residual_rmsnorm_separate.json",
        std::move(separate_metadata),
        options.benchmark,
        warpforge::rmsnorm_tolerance(options.columns),
        expected,
        device_output.get(),
        4U,
        2U,
        [&](const cudaStream_t launch_stream) {
            warpforge::add_cuda(
                device_input.get(),
                device_residual.get(),
                device_intermediate.get(),
                count,
                warpforge::fusion_default_block_size,
                launch_stream);
            warpforge::rmsnorm_cuda(
                device_intermediate.get(),
                device_weight.get(),
                device_output.get(),
                options.rows,
                options.columns,
                epsilon,
                warpforge::RmsNormVariant::block,
                warpforge::fusion_default_block_size,
                launch_stream);
        },
        stream.get()));

    auto fused_metadata = warpforge::make_benchmark_metadata(
        "residual_rmsnorm",
        "fused",
        "fp32",
        {{"columns", options.columns}, {"kernel_launches", 1U}, {"rows", options.rows}},
        launch);
    records.push_back(measure_cuda_case(
        "residual_rmsnorm_fused.json",
        std::move(fused_metadata),
        options.benchmark,
        warpforge::rmsnorm_tolerance(options.columns),
        expected,
        device_output.get(),
        3U,
        1U,
        [&](const cudaStream_t launch_stream) {
            warpforge::residual_rmsnorm_fused_cuda(
                device_input.get(),
                device_residual.get(),
                device_weight.get(),
                device_output.get(),
                options.rows,
                options.columns,
                epsilon,
                warpforge::fusion_default_block_size,
                launch_stream);
        },
        stream.get()));
}

void benchmark_swiglu(
    const Options& options,
    std::vector<RecordedResult>& records) {
    const std::size_t count = options.element_count;
    const auto gate = random_values(count, options.benchmark.seed + 3U, -5.0F, 5.0F);
    const auto up = random_values(count, options.benchmark.seed + 4U, -3.0F, 3.0F);
    std::vector<float> expected(count);
    warpforge::swiglu_fused_cpu(gate.data(), up.data(), expected.data(), count);
    LocalDeviceBuffer<float> device_gate(count);
    LocalDeviceBuffer<float> device_up(count);
    LocalDeviceBuffer<float> device_intermediate(count);
    LocalDeviceBuffer<float> device_output(count);
    LocalStream stream;
    CUDA_CHECK(cudaMemcpy(
        device_gate.get(), gate.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_up.get(), up.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    const auto grid = checked_grid_size(count, warpforge::fusion_default_block_size);
    const warpforge::LaunchConfiguration launch{
        {grid, 1U, 1U}, {warpforge::fusion_default_block_size, 1U, 1U}, 0U};
    auto separate_metadata = warpforge::make_benchmark_metadata(
        "swiglu",
        "separate",
        "fp32",
        {{"elements", count}, {"kernel_launches", 2U}},
        launch);
    records.push_back(measure_cuda_case(
        "swiglu_separate.json",
        std::move(separate_metadata),
        options.benchmark,
        {1.0e-5, 1.0e-5},
        expected,
        device_output.get(),
        3U,
        2U,
        [&](const cudaStream_t launch_stream) {
            warpforge::swiglu_unfused_cuda(
                device_gate.get(),
                device_up.get(),
                device_intermediate.get(),
                device_output.get(),
                count,
                warpforge::fusion_default_block_size,
                launch_stream);
        },
        stream.get()));

    auto fused_metadata = warpforge::make_benchmark_metadata(
        "swiglu",
        "fused",
        "fp32",
        {{"elements", count}, {"kernel_launches", 1U}},
        launch);
    records.push_back(measure_cuda_case(
        "swiglu_fused.json",
        std::move(fused_metadata),
        options.benchmark,
        {1.0e-5, 1.0e-5},
        expected,
        device_output.get(),
        2U,
        1U,
        [&](const cudaStream_t launch_stream) {
            warpforge::swiglu_fused_cuda(
                device_gate.get(),
                device_up.get(),
                device_output.get(),
                count,
                warpforge::fusion_default_block_size,
                launch_stream);
        },
        stream.get()));
}

void benchmark_pipeline_variant(
    const Options& options,
    const bool two_streams,
    std::vector<RecordedResult>& records) {
    const std::size_t count = options.pipeline_element_count;
    const std::size_t chunks = std::min(options.chunk_count, count);
    const std::size_t chunk_capacity = (count + chunks - 1U) / chunks;
    LocalPinnedBuffer<float> input(count);
    LocalPinnedBuffer<float> output(count);
    std::vector<float> expected(count);
    std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed + 5U));
    std::uniform_real_distribution<float> distribution(-5.0F, 5.0F);
    for (std::size_t index = 0U; index < count; ++index) {
        input[index] = distribution(generator);
    }
    warpforge::silu_cpu(input.get(), expected.data(), count);

    std::array<LocalStream, 2> streams{};
    std::array<LocalEvent, 2> completion_events{};
    std::array<LocalDeviceBuffer<float>, 2> buffers{
        LocalDeviceBuffer<float>{chunk_capacity}, LocalDeviceBuffer<float>{chunk_capacity}};
    const std::size_t active_streams = two_streams ? std::min<std::size_t>(2U, chunks) : 1U;
    const auto execute = [&]() {
        for (std::size_t chunk = 0U; chunk < chunks; ++chunk) {
            const std::size_t stream_index = two_streams ? chunk % active_streams : 0U;
            const std::size_t offset = chunk * chunk_capacity;
            const std::size_t chunk_elements = std::min(chunk_capacity, count - offset);
            const std::size_t bytes = chunk_elements * sizeof(float);
            const cudaStream_t stream = streams[stream_index].get();
            CUDA_CHECK(cudaMemcpyAsync(
                buffers[stream_index].get(),
                input.get() + offset,
                bytes,
                cudaMemcpyHostToDevice,
                stream));
            warpforge::silu_cuda(
                buffers[stream_index].get(),
                buffers[stream_index].get(),
                chunk_elements,
                warpforge::fusion_default_block_size,
                stream);
            CUDA_CHECK(cudaMemcpyAsync(
                output.get() + offset,
                buffers[stream_index].get(),
                bytes,
                cudaMemcpyDeviceToHost,
                stream));
        }
        for (std::size_t index = 0U; index < active_streams; ++index) {
            CUDA_CHECK(cudaEventRecord(completion_events[index].get(), streams[index].get()));
        }
        for (std::size_t index = 0U; index < active_streams; ++index) {
            CUDA_CHECK(cudaEventSynchronize(completion_events[index].get()));
        }
    };

    execute();
    const auto validation = warpforge::validate_fp32(
        expected.data(), output.get(), count, {1.0e-5, 1.0e-5});
    if (!validation.passed) {
        throw std::runtime_error(
            std::string(two_streams ? "two" : "single") +
            "-stream pipeline failed validation at index " +
            std::to_string(validation.worst_index));
    }
    auto samples = measure_host_operation(options.benchmark, execute);
    const auto statistics = warpforge::summarize_samples(samples);
    const std::string implementation = two_streams ? "two_stream" : "single_stream";
    auto metadata = warpforge::make_benchmark_metadata(
        "pinned_silu_pipeline",
        implementation,
        "fp32",
        {{"chunks", chunks},
         {"elements", count},
         {"streams", active_streams}},
        {{checked_grid_size(chunk_capacity, warpforge::fusion_default_block_size),
          1U,
          1U},
         {warpforge::fusion_default_block_size, 1U, 1U},
         0U});
    metadata.timing_scope = "end-to-end";
    const double transferred_bytes = static_cast<double>(2U * count * sizeof(float));
    warpforge::BenchmarkResult result{
        1,
        options.benchmark,
        std::move(metadata),
        statistics,
        validation,
        {{"absolute_tolerance", 1.0e-5},
         {"effective_bandwidth_gbps_from_median",
          bandwidth_gbps(transferred_bytes, statistics.median_ms)},
         {"logical_bytes", transferred_bytes},
         {"logical_reads_per_element", 1.0},
         {"logical_writes_per_element", 1.0},
         {"relative_tolerance", 1.0e-5}},
        std::move(samples)};
    records.push_back(
        {"pipeline_" + implementation + ".json", std::move(result)});
}

void add_speedups(std::vector<RecordedResult>& records) {
    for (auto& record : records) {
        const auto baseline = std::find_if(
            records.begin(), records.end(), [&](const RecordedResult& candidate) {
                if (candidate.result.metadata.operation != record.result.metadata.operation) {
                    return false;
                }
                const std::string& implementation = candidate.result.metadata.implementation;
                return implementation == "separate" || implementation == "single_stream";
            });
        if (baseline == records.end()) {
            throw std::runtime_error("fusion benchmark baseline is missing");
        }
        record.result.metrics["speedup_vs_operation_baseline"] =
            baseline->result.statistics.median_ms / record.result.statistics.median_ms;
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
        throw std::runtime_error("failed to create Stage 7 summary CSV");
    }
    summary << "schema_version,file,operation,implementation,timing_scope,median_ms,p95_ms,"
               "effective_bandwidth_gbps,speedup,logical_reads,logical_writes,"
               "max_absolute_error,validation_passed,git_commit\n";
    summary << std::setprecision(17);
    for (const auto& record : records) {
        const auto& result = record.result;
        summary << result.schema_version << ','
                << record.filename << ','
                << result.metadata.operation << ','
                << result.metadata.implementation << ','
                << result.metadata.timing_scope << ','
                << result.statistics.median_ms << ','
                << result.statistics.p95_ms << ','
                << result.metrics.at("effective_bandwidth_gbps_from_median") << ','
                << result.metrics.at("speedup_vs_operation_baseline") << ','
                << result.metrics.at("logical_reads_per_element") << ','
                << result.metrics.at("logical_writes_per_element") << ','
                << result.validation.max_absolute_error << ','
                << (result.validation.passed ? "true" : "false") << ','
                << result.metadata.git_commit << '\n';
    }
}

void print_results(const std::vector<RecordedResult>& records) {
    for (const auto& record : records) {
        std::cout << record.result.metadata.operation << '/'
                  << record.result.metadata.implementation << ": median "
                  << record.result.statistics.median_ms << " ms, "
                  << record.result.metrics.at("speedup_vs_operation_baseline")
                  << "x baseline\n";
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
        validate_options(options);
        std::vector<RecordedResult> records;
        records.reserve(6U);
        if (options.mode != RunMode::pipeline_only) {
            benchmark_residual_rmsnorm(options, records);
            benchmark_swiglu(options, records);
        }
        if (options.mode != RunMode::fusion_only) {
            benchmark_pipeline_variant(options, false, records);
            benchmark_pipeline_variant(options, true, records);
        }
        add_speedups(records);
        write_results(options.output_directory, records);
        print_results(records);
        std::cout << "Stage 7 fusion and pipeline validation: PASS\n"
                  << "Results: " << options.output_directory.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 7 fusion benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
