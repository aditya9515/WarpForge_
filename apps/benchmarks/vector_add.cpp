#include <warpforge/benchmark.hpp>
#include <warpforge/cuda_check.cuh>
#include <warpforge/runtime.cuh>
#include <warpforge/validation.hpp>
#include <warpforge/vector_add.cuh>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Options final {
    std::size_t element_count{1U << 20U};
    warpforge::BenchmarkConfig benchmark;
    std::filesystem::path output_path{"benchmarks/results/vector_add.json"};
    bool show_help{};
};

template <typename T> using LocalDeviceAllocation = warpforge::DeviceBuffer<T>;

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
        if (option == "--size") {
            const std::uint64_t parsed = parse_unsigned(value, option);
            if (parsed > std::numeric_limits<std::size_t>::max()) {
                throw std::out_of_range("--size exceeds the host size range");
            }
            options.element_count = static_cast<std::size_t>(parsed);
        } else if (option == "--warmups") {
            options.benchmark.warmup_iterations =
                static_cast<std::size_t>(parse_unsigned(value, option));
        } else if (option == "--iterations") {
            options.benchmark.measurement_iterations =
                static_cast<std::size_t>(parse_unsigned(value, option));
        } else if (option == "--seed") {
            options.benchmark.seed = parse_unsigned(value, option);
        } else if (option == "--output") {
            options.output_path = value;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }
    return options;
}

void print_usage() {
    std::cout << "Usage: warpforge-benchmark-vector-add [options]\n"
              << "  --size <elements>       Vector length (default: 1048576)\n"
              << "  --warmups <count>       Warmup launches (default: 10)\n"
              << "  --iterations <count>    Measured launches (default: 100)\n"
              << "  --seed <value>          Deterministic input seed (default: 2027)\n"
              << "  --output <path>         JSON output path\n"
              << "  --help                  Show this message\n";
}

double effective_bandwidth_gbps(const std::size_t element_count, const double latency_ms) {
    if (latency_ms <= 0.0) {
        return 0.0;
    }
    constexpr std::size_t arrays_accessed = 3;
    const double logical_bytes =
        static_cast<double>(element_count) * arrays_accessed * sizeof(float);
    return logical_bytes / (latency_ms * 1.0e6);
}

} // namespace

int main(const int argument_count, char** arguments) {
    try {
        const Options options = parse_options(argument_count, arguments);
        if (options.show_help) {
            print_usage();
            return EXIT_SUCCESS;
        }
        if (options.element_count == 0) {
            throw std::invalid_argument("--size must be greater than zero for a benchmark run");
        }
        if (options.benchmark.measurement_iterations == 0) {
            throw std::invalid_argument("--iterations must be greater than zero");
        }
        if (options.element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::out_of_range("--size exceeds the addressable FP32 allocation range");
        }

        std::vector<float> left(options.element_count);
        std::vector<float> right(options.element_count);
        std::vector<float> expected(options.element_count);
        std::vector<float> actual(options.element_count);

        std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed));
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        for (std::size_t index = 0; index < options.element_count; ++index) {
            left[index] = distribution(generator);
            right[index] = distribution(generator);
        }
        warpforge::vector_add_cpu(left.data(), right.data(), expected.data(),
                                  options.element_count);

        LocalDeviceAllocation<float> device_left(options.element_count);
        LocalDeviceAllocation<float> device_right(options.element_count);
        LocalDeviceAllocation<float> device_output(options.element_count);
        const std::size_t bytes = options.element_count * sizeof(float);

        CUDA_CHECK(cudaMemcpy(device_left.get(), left.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_right.get(), right.data(), bytes, cudaMemcpyHostToDevice));

        warpforge::vector_add_cuda(device_left.get(), device_right.get(), device_output.get(),
                                   options.element_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost));

        const warpforge::ValidationResult validation = warpforge::validate_fp32(
            expected.data(), actual.data(), options.element_count, {1.0e-5, 1.0e-5});
        if (!validation.passed) {
            std::cerr << "VectorAdd validation failed at index " << validation.worst_index
                      << ": expected " << validation.expected_at_worst << ", actual "
                      << validation.actual_at_worst << ", max absolute error "
                      << validation.max_absolute_error << '\n';
            return EXIT_FAILURE;
        }

        const std::vector<double> samples_ms =
            warpforge::measure_cuda_kernel(options.benchmark, nullptr, [&](cudaStream_t stream) {
                warpforge::vector_add_cuda(device_left.get(), device_right.get(),
                                           device_output.get(), options.element_count,
                                           warpforge::vector_add_default_block_size, stream);
            });
        const warpforge::BenchmarkStatistics statistics = warpforge::summarize_samples(samples_ms);

        const unsigned int grid_size = warpforge::vector_add_grid_size(options.element_count);
        warpforge::LaunchConfiguration launch;
        launch.grid = {grid_size, 1U, 1U};
        launch.block = {warpforge::vector_add_default_block_size, 1U, 1U};

        warpforge::BenchmarkResult result;
        result.config = options.benchmark;
        result.metadata = warpforge::make_benchmark_metadata(
            "vector_add", "cuda_baseline", "fp32",
            {{"elements", static_cast<std::uint64_t>(options.element_count)}}, launch);
        result.statistics = statistics;
        result.validation = validation;
        result.metrics["effective_bandwidth_gbps_from_median"] =
            effective_bandwidth_gbps(options.element_count, statistics.median_ms);
        result.samples_ms = samples_ms;
        warpforge::write_benchmark_json(result, options.output_path);

        std::cout << "VectorAdd validation: PASS\n";
        std::cout << "Elements:             " << options.element_count << '\n';
        std::cout << "Samples:              " << statistics.sample_count << '\n';
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Minimum latency:      " << statistics.minimum_ms << " ms\n";
        std::cout << "Mean latency:         " << statistics.mean_ms << " ms\n";
        std::cout << "Median latency:       " << statistics.median_ms << " ms\n";
        std::cout << "P95 latency:          " << statistics.p95_ms << " ms\n";
        std::cout << "Standard deviation:   " << statistics.standard_deviation_ms << " ms\n";
        std::cout << "Effective bandwidth:  "
                  << result.metrics.at("effective_bandwidth_gbps_from_median") << " GB/s\n";
        std::cout << "Maximum abs error:    " << validation.max_absolute_error << '\n';
        std::cout << "JSON output:          " << options.output_path.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "VectorAdd benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
