#include <warpforge/benchmark.hpp>
#include <warpforge/cuda_check.cuh>
#include <warpforge/reduction.cuh>
#include <warpforge/runtime.cuh>
#include <warpforge/validation.hpp>

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
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::array<warpforge::ReductionOperation, 2> operations{
    warpforge::ReductionOperation::sum,
    warpforge::ReductionOperation::maximum,
};

constexpr std::array<warpforge::ReductionVariant, 5> variants{
    warpforge::ReductionVariant::naive_interleaved,
    warpforge::ReductionVariant::shared_memory,
    warpforge::ReductionVariant::reduced_divergence,
    warpforge::ReductionVariant::unrolled,
    warpforge::ReductionVariant::warp_shuffle,
};

struct Options final {
    std::size_t element_count{1U << 24U};
    unsigned int block_size{warpforge::reduction_default_block_size};
    warpforge::BenchmarkConfig benchmark{};
    std::filesystem::path output_directory{"benchmarks/results/stage4"};
    std::string operation{"all"};
    std::string variant{"all"};
    bool profile_only{};
    bool show_help{};
};

struct RecordedResult final {
    std::string filename;
    warpforge::BenchmarkResult result;
};

template <typename T>
using LocalDeviceBuffer = warpforge::DeviceBuffer<T>;

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
        } else if (option == "--block-size") {
            const std::uint64_t parsed = parse_unsigned(value, option);
            if (parsed > std::numeric_limits<unsigned int>::max()) {
                throw std::out_of_range("--block-size exceeds unsigned int");
            }
            options.block_size = static_cast<unsigned int>(parsed);
        } else if (option == "--warmups") {
            options.benchmark.warmup_iterations = parse_size(value, option);
        } else if (option == "--iterations") {
            options.benchmark.measurement_iterations = parse_size(value, option);
        } else if (option == "--seed") {
            options.benchmark.seed = parse_unsigned(value, option);
        } else if (option == "--output-dir") {
            options.output_directory = value;
        } else if (option == "--operation") {
            options.operation = value;
        } else if (option == "--variant") {
            options.variant = value;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }
    return options;
}

void print_usage() {
    std::cout << "Usage: warpforge-benchmark-reduction [options]\n"
              << "  --size <elements>       Input length (default: 16777216)\n"
              << "  --block-size <threads>  Power of two from 32 to 1024 (default: 256)\n"
              << "  --warmups <count>       Warmup reductions (default: 10)\n"
              << "  --iterations <count>    Measured reductions (default: 100)\n"
              << "  --seed <value>          Deterministic input seed (default: 2027)\n"
              << "  --operation <value>     all, sum, or maximum\n"
              << "  --variant <value>       all, naive_interleaved, shared_memory,\n"
              << "                          reduced_divergence, unrolled, or warp_shuffle\n"
              << "  --output-dir <path>     Result directory\n"
              << "  --profile-only          Execute one selected reduction without timing\n"
              << "  --help                  Show this message\n";
}

warpforge::ReductionOperation parse_operation(const std::string& name) {
    if (name == "sum") {
        return warpforge::ReductionOperation::sum;
    }
    if (name == "maximum" || name == "max") {
        return warpforge::ReductionOperation::maximum;
    }
    throw std::invalid_argument("unknown reduction operation: " + name);
}

warpforge::ReductionVariant parse_variant(const std::string& name) {
    for (const auto variant : variants) {
        if (name == warpforge::reduction_variant_name(variant)) {
            return variant;
        }
    }
    throw std::invalid_argument("unknown reduction variant: " + name);
}

void validate_options(const Options& options) {
    if (options.element_count == 0) {
        throw std::invalid_argument("--size must be greater than zero");
    }
    if (options.benchmark.measurement_iterations == 0) {
        throw std::invalid_argument("--iterations must be greater than zero");
    }
    if (options.element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::out_of_range("input exceeds the addressable FP32 allocation range");
    }
    static_cast<void>(warpforge::reduction_elements_per_block(
        warpforge::ReductionVariant::warp_shuffle, options.block_size));
    if (options.operation != "all") {
        static_cast<void>(parse_operation(options.operation));
    }
    if (options.variant != "all") {
        static_cast<void>(parse_variant(options.variant));
    }
    if (options.profile_only && (options.operation == "all" || options.variant == "all")) {
        throw std::invalid_argument(
            "--profile-only requires one --operation and one --variant");
    }
}

std::vector<warpforge::ReductionOperation> selected_operations(const Options& options) {
    if (options.operation == "all") {
        return {operations.begin(), operations.end()};
    }
    return {parse_operation(options.operation)};
}

std::vector<warpforge::ReductionVariant> selected_variants(const Options& options) {
    if (options.variant == "all") {
        return {variants.begin(), variants.end()};
    }
    return {parse_variant(options.variant)};
}

warpforge::ValidationResult validate_result(
    const warpforge::ReductionOperation operation,
    const double sum_reference,
    const float maximum_reference,
    const float actual,
    const std::size_t input_count) {
    if (operation == warpforge::ReductionOperation::sum) {
        return warpforge::validate_reduction_sum(sum_reference, actual, input_count);
    }
    return warpforge::validate_fp32(
        &maximum_reference, &actual, 1U, {0.0, 0.0});
}

double expected_value(
    const warpforge::ReductionOperation operation,
    const double sum_reference,
    const float maximum_reference) {
    return operation == warpforge::ReductionOperation::sum
        ? sum_reference
        : static_cast<double>(maximum_reference);
}

RecordedResult run_experiment(
    const Options& options,
    const warpforge::ReductionOperation operation,
    const warpforge::ReductionVariant variant,
    const float* device_input,
    const double sum_reference,
    const float maximum_reference) {
    const std::size_t workspace_elements = warpforge::reduction_workspace_elements(
        options.element_count, variant, options.block_size);
    LocalDeviceBuffer<float> workspace_a(workspace_elements);
    LocalDeviceBuffer<float> workspace_b(workspace_elements);
    LocalDeviceBuffer<float> device_output(1U);

    const auto launch = [&](cudaStream_t stream) {
        warpforge::reduce_cuda(
            device_input,
            options.element_count,
            operation,
            variant,
            workspace_a.get(),
            workspace_b.get(),
            workspace_elements,
            device_output.get(),
            options.block_size,
            stream);
    };

    launch(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    float actual = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &actual, device_output.get(), sizeof(float), cudaMemcpyDeviceToHost));
    const auto validation = validate_result(
        operation, sum_reference, maximum_reference, actual, options.element_count);
    if (!validation.passed) {
        throw std::runtime_error(
            std::string(warpforge::reduction_operation_name(operation)) + "/" +
            warpforge::reduction_variant_name(variant) + " failed validation");
    }

    auto samples = warpforge::measure_cuda_kernel(
        options.benchmark, nullptr, launch);
    const auto statistics = warpforge::summarize_samples(samples);
    const std::size_t first_output = warpforge::reduction_output_count(
        options.element_count, variant, options.block_size);
    if (first_output > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("first reduction grid exceeds unsigned int");
    }

    warpforge::LaunchConfiguration launch_configuration;
    launch_configuration.grid = {
        static_cast<unsigned int>(first_output), 1U, 1U};
    launch_configuration.block = {options.block_size, 1U, 1U};
    launch_configuration.dynamic_shared_memory_bytes =
        warpforge::reduction_dynamic_shared_memory_bytes(variant, options.block_size);

    warpforge::BenchmarkResult result;
    result.config = options.benchmark;
    result.metadata = warpforge::make_benchmark_metadata(
        std::string("reduction_") + warpforge::reduction_operation_name(operation),
        warpforge::reduction_variant_name(variant),
        "fp32",
        {{"elements", static_cast<std::uint64_t>(options.element_count)},
         {"passes", static_cast<std::uint64_t>(warpforge::reduction_pass_count(
                        options.element_count, variant, options.block_size))},
         {"workspace_elements", static_cast<std::uint64_t>(workspace_elements)}},
        launch_configuration);
    result.statistics = statistics;
    result.validation = validation;
    result.metrics["absolute_error"] = validation.max_absolute_error;
    result.metrics["cpu_reference"] =
        expected_value(operation, sum_reference, maximum_reference);
    result.metrics["gpu_result"] = static_cast<double>(actual);
    result.metrics["elements_per_second_from_median"] =
        static_cast<double>(options.element_count) / (statistics.median_ms * 1.0e-3);
    result.metrics["effective_input_bandwidth_gbps_from_median"] =
        static_cast<double>(options.element_count * sizeof(float)) /
        (statistics.median_ms * 1.0e6);
    if (operation == warpforge::ReductionOperation::sum) {
        const auto tolerance = warpforge::reduction_sum_tolerance(options.element_count);
        result.metrics["absolute_tolerance"] = tolerance.absolute;
        result.metrics["relative_tolerance"] = tolerance.relative;
    }
    result.samples_ms = std::move(samples);

    const std::string filename =
        std::string(warpforge::reduction_operation_name(operation)) + "_" +
        warpforge::reduction_variant_name(variant);
    return {filename, std::move(result)};
}

void add_speedups_and_write(
    const Options& options,
    std::vector<RecordedResult>& records) {
    for (const auto operation : selected_operations(options)) {
        const std::string operation_name = warpforge::reduction_operation_name(operation);
        const auto baseline = std::find_if(
            records.begin(), records.end(), [&](const RecordedResult& record) {
                return record.result.metadata.operation == "reduction_" + operation_name &&
                    record.result.metadata.implementation == "naive_interleaved";
            });
        if (baseline == records.end()) {
            continue;
        }
        const double baseline_median = baseline->result.statistics.median_ms;
        for (auto& record : records) {
            if (record.result.metadata.operation == "reduction_" + operation_name) {
                record.result.metrics["speedup_vs_naive_interleaved"] =
                    baseline_median / record.result.statistics.median_ms;
            }
        }
    }

    std::filesystem::create_directories(options.output_directory);
    for (const auto& record : records) {
        warpforge::write_benchmark_json(
            record.result, options.output_directory / (record.filename + ".json"));
    }

    std::ofstream summary(options.output_directory / "summary.csv", std::ios::trunc);
    if (!summary) {
        throw std::runtime_error("failed to create reduction summary CSV");
    }
    summary << "schema_version,file,operation,variant,median_ms,p95_ms,"
               "elements_per_second,input_bandwidth_gbps,speedup_vs_naive,"
               "absolute_error,validation_passed,git_commit\n";
    summary << std::setprecision(17);
    for (const auto& record : records) {
        const auto speedup = record.result.metrics.find("speedup_vs_naive_interleaved");
        summary << record.result.schema_version << ',' << record.filename << ','
                << record.result.metadata.operation << ','
                << record.result.metadata.implementation << ','
                << record.result.statistics.median_ms << ','
                << record.result.statistics.p95_ms << ','
                << record.result.metrics.at("elements_per_second_from_median") << ','
                << record.result.metrics.at("effective_input_bandwidth_gbps_from_median") << ','
                << (speedup == record.result.metrics.end() ? 0.0 : speedup->second) << ','
                << record.result.validation.max_absolute_error << ','
                << (record.result.validation.passed ? "true" : "false") << ','
                << record.result.metadata.git_commit << '\n';
    }
}

void print_results(const std::vector<RecordedResult>& records) {
    std::cout << std::left << std::setw(34) << "Experiment" << std::right
              << std::setw(13) << "Median ms" << std::setw(13) << "P95 ms"
              << std::setw(13) << "GB/s" << std::setw(11) << "Speedup" << '\n';
    std::cout << std::string(84, '-') << '\n';
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& record : records) {
        const auto speedup = record.result.metrics.find("speedup_vs_naive_interleaved");
        std::cout << std::left << std::setw(34) << record.filename << std::right
                  << std::setw(13) << record.result.statistics.median_ms
                  << std::setw(13) << record.result.statistics.p95_ms
                  << std::setw(13)
                  << record.result.metrics.at("effective_input_bandwidth_gbps_from_median")
                  << std::setw(11)
                  << (speedup == record.result.metrics.end() ? 0.0 : speedup->second)
                  << '\n';
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

        std::vector<float> input(options.element_count);
        std::mt19937 generator(static_cast<std::uint32_t>(options.benchmark.seed));
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        for (float& value : input) {
            value = distribution(generator);
        }
        input[options.element_count / 2U] = 7.25F;
        const double sum_reference =
            warpforge::reduction_sum_cpu(input.data(), input.size());
        const float maximum_reference =
            warpforge::reduction_max_cpu(input.data(), input.size());

        LocalDeviceBuffer<float> device_input(options.element_count);
        CUDA_CHECK(cudaMemcpy(
            device_input.get(),
            input.data(),
            options.element_count * sizeof(float),
            cudaMemcpyHostToDevice));

        std::vector<RecordedResult> records;
        for (const auto operation : selected_operations(options)) {
            for (const auto variant : selected_variants(options)) {
                if (options.profile_only) {
                    const std::size_t workspace_elements =
                        warpforge::reduction_workspace_elements(
                            options.element_count, variant, options.block_size);
                    LocalDeviceBuffer<float> workspace_a(workspace_elements);
                    LocalDeviceBuffer<float> workspace_b(workspace_elements);
                    LocalDeviceBuffer<float> output(1U);
                    warpforge::reduce_cuda(
                        device_input.get(),
                        options.element_count,
                        operation,
                        variant,
                        workspace_a.get(),
                        workspace_b.get(),
                        workspace_elements,
                        output.get(),
                        options.block_size);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    float actual = 0.0F;
                    CUDA_CHECK(cudaMemcpy(
                        &actual, output.get(), sizeof(float), cudaMemcpyDeviceToHost));
                    const auto validation = validate_result(
                        operation,
                        sum_reference,
                        maximum_reference,
                        actual,
                        options.element_count);
                    if (!validation.passed) {
                        throw std::runtime_error("profile-only reduction failed validation");
                    }
                    std::cout << "Profile reduction validation: PASS\n";
                    return EXIT_SUCCESS;
                }
                records.push_back(run_experiment(
                    options,
                    operation,
                    variant,
                    device_input.get(),
                    sum_reference,
                    maximum_reference));
            }
        }

        add_speedups_and_write(options, records);
        print_results(records);
        std::cout << "Stage 4 reduction validation: PASS\n";
        std::cout << "Results: " << options.output_directory.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 4 reduction benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
