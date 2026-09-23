#include <warpforge/benchmark.hpp>
#include <warpforge/causal_mask.cuh>
#include <warpforge/cuda_check.cuh>
#include <warpforge/elementwise.cuh>
#include <warpforge/rmsnorm.cuh>
#include <warpforge/rope.cuh>
#include <warpforge/runtime.cuh>
#include <warpforge/softmax.cuh>
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

struct Options final {
    std::size_t rows{4096U};
    std::size_t columns{1024U};
    std::size_t element_count{1U << 24U};
    std::size_t sequence{2048U};
    std::size_t heads{8U};
    std::size_t head_dimension{64U};
    std::size_t mask_length{512U};
    warpforge::BenchmarkConfig benchmark{};
    std::filesystem::path output_directory{"benchmarks/results/stage6"};
    bool show_help{};
};

struct RecordedResult final {
    std::string filename;
    warpforge::BenchmarkResult result;
};

template <typename T> using LocalDeviceBuffer = warpforge::DeviceBuffer<T>;
using LocalStream = warpforge::CudaStream;

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
        } else if (option == "--sequence") {
            options.sequence = parse_size(value, option);
        } else if (option == "--heads") {
            options.heads = parse_size(value, option);
        } else if (option == "--head-dim") {
            options.head_dimension = parse_size(value, option);
        } else if (option == "--mask-length") {
            options.mask_length = parse_size(value, option);
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
    std::cout << "Usage: warpforge-benchmark-transformer [options]\n"
              << "  --rows <count>          Softmax/RMSNorm rows (default: 4096)\n"
              << "  --columns <count>       Softmax/RMSNorm width (default: 1024)\n"
              << "  --elements <count>      Elementwise width (default: 16777216)\n"
              << "  --sequence <count>      RoPE sequence (default: 2048)\n"
              << "  --heads <count>         RoPE/mask heads (default: 8)\n"
              << "  --head-dim <count>      RoPE head dimension (default: 64)\n"
              << "  --mask-length <count>   Square causal-mask length (default: 512)\n"
              << "  --warmups <count>       Warmups per case (default: 10)\n"
              << "  --iterations <count>    Samples per case (default: 100)\n"
              << "  --seed <value>          Deterministic seed (default: 2027)\n"
              << "  --output-dir <path>     Result directory\n"
              << "  --help                  Show this message\n";
}

void validate_options(const Options& options) {
    if (options.rows == 0U || options.columns == 0U || options.element_count == 0U ||
        options.sequence == 0U || options.heads == 0U || options.head_dimension == 0U ||
        options.mask_length == 0U) {
        throw std::invalid_argument("all benchmark dimensions must be greater than zero");
    }
    if (options.head_dimension % 2U != 0U) {
        throw std::invalid_argument("--head-dim must be even");
    }
    if (options.benchmark.measurement_iterations == 0U) {
        throw std::invalid_argument("--iterations must be greater than zero");
    }
    const warpforge::RopeProblem rope{1U,  options.sequence, options.heads, options.head_dimension,
                                      17U, 10000.0F};
    static_cast<void>(warpforge::rope_element_count(rope));
    const warpforge::CausalMaskProblem mask{1U, options.heads, options.mask_length,
                                            options.mask_length, 0U};
    static_cast<void>(warpforge::causal_mask_element_count(mask));
    if (options.rows > std::numeric_limits<std::size_t>::max() / options.columns) {
        throw std::overflow_error("row-wise benchmark element count overflows size_t");
    }
}

std::vector<float> random_values(const std::size_t count, const std::uint64_t seed,
                                 const float minimum, const float maximum) {
    std::vector<float> values(count);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution(minimum, maximum);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

RecordedResult measure_case(const std::string& filename, warpforge::BenchmarkMetadata metadata,
                            const warpforge::BenchmarkConfig& config,
                            const warpforge::Tolerance tolerance,
                            const std::vector<float>& expected, const float* device_output,
                            const std::size_t logical_bytes, const warpforge::CudaWork& work,
                            const cudaStream_t stream) {
    work(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const auto validation =
        warpforge::validate_fp32(expected.data(), actual.data(), actual.size(), tolerance);
    if (!validation.passed) {
        throw std::runtime_error(metadata.operation + "/" + metadata.implementation +
                                 " failed validation at index " +
                                 std::to_string(validation.worst_index));
    }
    auto samples = warpforge::measure_cuda_kernel(config, stream, work);
    const auto statistics = warpforge::summarize_samples(samples);
    const double bandwidth = static_cast<double>(logical_bytes) / (statistics.median_ms * 1.0e6);
    warpforge::BenchmarkResult result{1,
                                      config,
                                      std::move(metadata),
                                      statistics,
                                      validation,
                                      {{"logical_bytes", static_cast<double>(logical_bytes)},
                                       {"effective_bandwidth_gbps_from_median", bandwidth},
                                       {"absolute_tolerance", tolerance.absolute},
                                       {"relative_tolerance", tolerance.relative}},
                                      std::move(samples)};
    return {filename, std::move(result)};
}

void benchmark_softmax(const Options& options, const cudaStream_t stream,
                       std::vector<RecordedResult>& records) {
    const std::size_t count = options.rows * options.columns;
    const auto input = random_values(count, options.benchmark.seed, -12.0F, 12.0F);
    std::vector<float> expected(count);
    warpforge::softmax_cpu(input.data(), expected.data(), options.rows, options.columns);
    LocalDeviceBuffer<float> device_input(count);
    LocalDeviceBuffer<float> device_output(count);
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    constexpr std::array variants{warpforge::SoftmaxVariant::naive,
                                  warpforge::SoftmaxVariant::block,
                                  warpforge::SoftmaxVariant::warp};
    for (const auto variant : variants) {
        const auto block = warpforge::softmax_default_block_size;
        const unsigned int grid =
            variant == warpforge::SoftmaxVariant::naive
                ? static_cast<unsigned int>((options.rows + block - 1U) / block)
                : static_cast<unsigned int>(options.rows);
        const std::string name = warpforge::softmax_variant_name(variant);
        auto metadata = warpforge::make_benchmark_metadata(
            "softmax", name, "fp32", {{"rows", options.rows}, {"columns", options.columns}},
            {{grid, 1U, 1U},
             {block, 1U, 1U},
             warpforge::softmax_dynamic_shared_memory_bytes(variant, block)});
        records.push_back(measure_case(
            "softmax_" + name + ".json", std::move(metadata), options.benchmark, {1.0e-5, 1.0e-5},
            expected, device_output.get(), 2U * count * sizeof(float),
            [&](const cudaStream_t launch_stream) {
                warpforge::softmax_cuda(device_input.get(), device_output.get(), options.rows,
                                        options.columns, variant, block, launch_stream);
            },
            stream));
    }
}

void benchmark_rmsnorm(const Options& options, const cudaStream_t stream,
                       std::vector<RecordedResult>& records) {
    const std::size_t count = options.rows * options.columns;
    const auto input = random_values(count, options.benchmark.seed + 1U, -3.0F, 3.0F);
    const auto weight = random_values(options.columns, options.benchmark.seed + 2U, 0.5F, 1.5F);
    std::vector<float> expected(count);
    warpforge::rmsnorm_cpu(input.data(), weight.data(), expected.data(), options.rows,
                           options.columns, 1.0e-5F);
    LocalDeviceBuffer<float> device_input(count);
    LocalDeviceBuffer<float> device_weight(options.columns);
    LocalDeviceBuffer<float> device_output(count);
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_weight.get(), weight.data(), options.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    constexpr std::array variants{warpforge::RmsNormVariant::naive,
                                  warpforge::RmsNormVariant::block};
    for (const auto variant : variants) {
        const auto block = warpforge::rmsnorm_default_block_size;
        const unsigned int grid =
            variant == warpforge::RmsNormVariant::naive
                ? static_cast<unsigned int>((options.rows + block - 1U) / block)
                : static_cast<unsigned int>(options.rows);
        const std::string name = warpforge::rmsnorm_variant_name(variant);
        auto metadata = warpforge::make_benchmark_metadata(
            "rmsnorm", name, "fp32", {{"rows", options.rows}, {"columns", options.columns}},
            {{grid, 1U, 1U},
             {block, 1U, 1U},
             warpforge::rmsnorm_dynamic_shared_memory_bytes(variant, block)});
        records.push_back(measure_case(
            "rmsnorm_" + name + ".json", std::move(metadata), options.benchmark,
            warpforge::rmsnorm_tolerance(options.columns), expected, device_output.get(),
            3U * count * sizeof(float),
            [&](const cudaStream_t launch_stream) {
                warpforge::rmsnorm_cuda(device_input.get(), device_weight.get(),
                                        device_output.get(), options.rows, options.columns, 1.0e-5F,
                                        variant, block, launch_stream);
            },
            stream));
    }
}

void benchmark_rope(const Options& options, const cudaStream_t stream,
                    std::vector<RecordedResult>& records) {
    const warpforge::RopeProblem problem{
        1U, options.sequence, options.heads, options.head_dimension, 17U, 10000.0F};
    const std::size_t count = warpforge::rope_element_count(problem);
    const auto input = random_values(count, options.benchmark.seed + 3U, -2.0F, 2.0F);
    std::vector<float> expected(count);
    warpforge::rope_cpu(input.data(), expected.data(), problem);
    LocalDeviceBuffer<float> device_input(count);
    LocalDeviceBuffer<float> device_output(count);
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    const unsigned int grid = static_cast<unsigned int>(
        (warpforge::rope_pair_count(problem) + warpforge::rope_default_block_size - 1U) /
        warpforge::rope_default_block_size);
    auto metadata = warpforge::make_benchmark_metadata(
        "rope", "interleaved_pairs", "fp32",
        {{"batch", problem.batch},
         {"sequence", problem.sequence},
         {"heads", problem.heads},
         {"head_dimension", problem.head_dimension},
         {"position_offset", problem.position_offset}},
        {{grid, 1U, 1U}, {warpforge::rope_default_block_size, 1U, 1U}, 0U});
    records.push_back(measure_case(
        "rope_interleaved_pairs.json", std::move(metadata), options.benchmark,
        // FP32 inverse-frequency rounding is amplified by long sequence positions.
        {2.5e-4, 1.0e-5}, expected, device_output.get(), 2U * count * sizeof(float),
        [&](const cudaStream_t launch_stream) {
            warpforge::rope_cuda(device_input.get(), device_output.get(), problem,
                                 warpforge::rope_default_block_size, launch_stream);
        },
        stream));
}

enum class ElementwiseCase { silu, add, multiply, scale, swiglu };

void benchmark_elementwise_case(const Options& options, const cudaStream_t stream,
                                const ElementwiseCase selected,
                                std::vector<RecordedResult>& records) {
    const std::size_t count = options.element_count;
    const auto left = random_values(count, options.benchmark.seed + 4U, -5.0F, 5.0F);
    const auto right = random_values(count, options.benchmark.seed + 5U, -3.0F, 3.0F);
    std::vector<float> expected(count);
    std::vector<float> intermediate(count);
    LocalDeviceBuffer<float> device_left(count);
    LocalDeviceBuffer<float> device_right(count);
    LocalDeviceBuffer<float> device_intermediate(count);
    LocalDeviceBuffer<float> device_output(count);
    CUDA_CHECK(
        cudaMemcpy(device_left.get(), left.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_right.get(), right.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    std::string operation;
    std::string implementation{"baseline"};
    std::size_t logical_arrays = 2U;
    std::size_t kernel_launches = 1U;
    warpforge::CudaWork work;
    if (selected == ElementwiseCase::silu) {
        operation = "silu";
        warpforge::silu_cpu(left.data(), expected.data(), count);
        work = [&](const cudaStream_t launch_stream) {
            warpforge::silu_cuda(device_left.get(), device_output.get(), count,
                                 warpforge::elementwise_default_block_size, launch_stream);
        };
    } else if (selected == ElementwiseCase::add) {
        operation = "add";
        logical_arrays = 3U;
        warpforge::add_cpu(left.data(), right.data(), expected.data(), count);
        work = [&](const cudaStream_t launch_stream) {
            warpforge::add_cuda(device_left.get(), device_right.get(), device_output.get(), count,
                                warpforge::elementwise_default_block_size, launch_stream);
        };
    } else if (selected == ElementwiseCase::multiply) {
        operation = "multiply";
        logical_arrays = 3U;
        warpforge::multiply_cpu(left.data(), right.data(), expected.data(), count);
        work = [&](const cudaStream_t launch_stream) {
            warpforge::multiply_cuda(device_left.get(), device_right.get(), device_output.get(),
                                     count, warpforge::elementwise_default_block_size,
                                     launch_stream);
        };
    } else if (selected == ElementwiseCase::scale) {
        operation = "scale";
        warpforge::scale_cpu(left.data(), 0.375F, expected.data(), count);
        work = [&](const cudaStream_t launch_stream) {
            warpforge::scale_cuda(device_left.get(), 0.375F, device_output.get(), count,
                                  warpforge::elementwise_default_block_size, launch_stream);
        };
    } else {
        operation = "swiglu";
        implementation = "unfused";
        logical_arrays = 5U;
        kernel_launches = 2U;
        warpforge::swiglu_unfused_cpu(left.data(), right.data(), intermediate.data(),
                                      expected.data(), count);
        work = [&](const cudaStream_t launch_stream) {
            warpforge::swiglu_unfused_cuda(device_left.get(), device_right.get(),
                                           device_intermediate.get(), device_output.get(), count,
                                           warpforge::elementwise_default_block_size,
                                           launch_stream);
        };
    }
    const unsigned int grid =
        static_cast<unsigned int>((count + warpforge::elementwise_default_block_size - 1U) /
                                  warpforge::elementwise_default_block_size);
    auto metadata = warpforge::make_benchmark_metadata(
        operation, implementation, "fp32",
        {{"elements", count}, {"kernel_launches", kernel_launches}},
        {{grid, 1U, 1U}, {warpforge::elementwise_default_block_size, 1U, 1U}, 0U});
    records.push_back(measure_case(operation + "_" + implementation + ".json", std::move(metadata),
                                   options.benchmark, {1.0e-5, 1.0e-5}, expected,
                                   device_output.get(), logical_arrays * count * sizeof(float),
                                   work, stream));
}

void benchmark_causal_mask(const Options& options, const cudaStream_t stream,
                           std::vector<RecordedResult>& records) {
    const warpforge::CausalMaskProblem problem{1U, options.heads, options.mask_length,
                                               options.mask_length, 0U};
    const std::size_t count = warpforge::causal_mask_element_count(problem);
    const auto input = random_values(count, options.benchmark.seed + 6U, -4.0F, 4.0F);
    std::vector<float> expected(count);
    warpforge::causal_mask_cpu(input.data(), expected.data(), problem);
    LocalDeviceBuffer<float> device_input(count);
    LocalDeviceBuffer<float> device_output(count);
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    const unsigned int grid =
        static_cast<unsigned int>((count + warpforge::causal_mask_default_block_size - 1U) /
                                  warpforge::causal_mask_default_block_size);
    auto metadata = warpforge::make_benchmark_metadata(
        "causal_mask", "baseline", "fp32",
        {{"batch", problem.batch},
         {"heads", problem.heads},
         {"query_length", problem.query_length},
         {"key_length", problem.key_length},
         {"query_position_offset", problem.query_position_offset}},
        {{grid, 1U, 1U}, {warpforge::causal_mask_default_block_size, 1U, 1U}, 0U});
    records.push_back(measure_case(
        "causal_mask_baseline.json", std::move(metadata), options.benchmark, {0.0, 0.0}, expected,
        device_output.get(), 2U * count * sizeof(float),
        [&](const cudaStream_t launch_stream) {
            warpforge::causal_mask_cuda(device_input.get(), device_output.get(), problem,
                                        warpforge::causal_mask_default_block_size, launch_stream);
        },
        stream));
}

void add_speedups(std::vector<RecordedResult>& records) {
    for (auto& record : records) {
        const auto baseline =
            std::find_if(records.begin(), records.end(), [&](const RecordedResult& candidate) {
                if (candidate.result.metadata.operation != record.result.metadata.operation) {
                    return false;
                }
                if (record.result.metadata.operation == "softmax" ||
                    record.result.metadata.operation == "rmsnorm") {
                    return candidate.result.metadata.implementation == "naive";
                }
                return candidate.result.metadata.implementation ==
                       record.result.metadata.implementation;
            });
        if (baseline == records.end()) {
            throw std::runtime_error("operation baseline is missing");
        }
        record.result.metrics["speedup_vs_operation_baseline"] =
            baseline->result.statistics.median_ms / record.result.statistics.median_ms;
    }
}

void write_results(const std::filesystem::path& directory,
                   const std::vector<RecordedResult>& records) {
    std::filesystem::create_directories(directory);
    for (const auto& record : records) {
        warpforge::write_benchmark_json(record.result, directory / record.filename);
    }
    std::ofstream summary(directory / "summary.csv", std::ios::binary | std::ios::trunc);
    if (!summary) {
        throw std::runtime_error("failed to create Transformer summary CSV");
    }
    summary << "schema_version,file,operation,implementation,median_ms,p95_ms,"
               "effective_bandwidth_gbps,speedup,max_absolute_error,validation_passed,git_commit\n";
    summary << std::setprecision(17);
    for (const auto& record : records) {
        const auto& result = record.result;
        summary << result.schema_version << ','
                << record.filename.substr(0U, record.filename.size() - 5U) << ','
                << result.metadata.operation << ',' << result.metadata.implementation << ','
                << result.statistics.median_ms << ',' << result.statistics.p95_ms << ','
                << result.metrics.at("effective_bandwidth_gbps_from_median") << ','
                << result.metrics.at("speedup_vs_operation_baseline") << ','
                << result.validation.max_absolute_error << ','
                << (result.validation.passed ? "true" : "false") << ','
                << result.metadata.git_commit << '\n';
    }
}

} // namespace

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
        std::vector<RecordedResult> records;
        records.reserve(12U);
        benchmark_softmax(options, stream.get(), records);
        benchmark_rmsnorm(options, stream.get(), records);
        benchmark_rope(options, stream.get(), records);
        benchmark_elementwise_case(options, stream.get(), ElementwiseCase::silu, records);
        benchmark_elementwise_case(options, stream.get(), ElementwiseCase::add, records);
        benchmark_elementwise_case(options, stream.get(), ElementwiseCase::multiply, records);
        benchmark_elementwise_case(options, stream.get(), ElementwiseCase::scale, records);
        benchmark_elementwise_case(options, stream.get(), ElementwiseCase::swiglu, records);
        benchmark_causal_mask(options, stream.get(), records);
        add_speedups(records);
        write_results(options.output_directory, records);
        for (const auto& record : records) {
            std::cout << record.result.metadata.operation << '/'
                      << record.result.metadata.implementation << ": median "
                      << record.result.statistics.median_ms << " ms, "
                      << record.result.metrics.at("effective_bandwidth_gbps_from_median")
                      << " logical GB/s\n";
        }
        std::cout << "Stage 6 Transformer-kernel validation: PASS\n"
                  << "Results: " << options.output_directory.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Stage 6 Transformer benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
