#include <warpforge/benchmark.hpp>
#include <warpforge/device.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef WARPFORGE_BUILD_TYPE
#define WARPFORGE_BUILD_TYPE "unknown"
#endif

#ifndef WARPFORGE_COMPILER
#define WARPFORGE_COMPILER "unknown"
#endif

#ifndef WARPFORGE_GIT_COMMIT
#define WARPFORGE_GIT_COMMIT "unknown"
#endif

namespace warpforge {
namespace {

double percentile(const std::vector<double>& sorted_samples, const double probability) {
    if (sorted_samples.size() == 1) {
        return sorted_samples.front();
    }

    const double position = probability * static_cast<double>(sorted_samples.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted_samples[lower] + fraction * (sorted_samples[upper] - sorted_samples[lower]);
}

std::string format_cuda_version(const int version) {
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string current_utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif

    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

void write_number(std::ostream& output, const double value) {
    if (std::isfinite(value)) {
        output << std::setprecision(17) << value;
    } else {
        output << "null";
    }
}

void write_dimensions(
    std::ostream& output,
    const std::map<std::string, std::uint64_t>& dimensions) {
    output << "{";
    bool first = true;
    for (const auto& [name, value] : dimensions) {
        if (!first) {
            output << ',';
        }
        output << "\n      \"" << escape_json(name) << "\": " << value;
        first = false;
    }
    if (!dimensions.empty()) {
        output << '\n' << "    ";
    }
    output << '}';
}

void write_metrics(std::ostream& output, const std::map<std::string, double>& metrics) {
    output << "{";
    bool first = true;
    for (const auto& [name, value] : metrics) {
        if (!first) {
            output << ',';
        }
        output << "\n    \"" << escape_json(name) << "\": ";
        write_number(output, value);
        first = false;
    }
    if (!metrics.empty()) {
        output << '\n' << "  ";
    }
    output << '}';
}

}  // namespace

BenchmarkStatistics summarize_samples(const std::vector<double>& samples_ms) {
    if (samples_ms.empty()) {
        throw std::invalid_argument("cannot summarize an empty benchmark sample set");
    }
    if (std::any_of(samples_ms.begin(), samples_ms.end(), [](const double value) {
            return !std::isfinite(value) || value < 0.0;
        })) {
        throw std::invalid_argument("benchmark samples must be finite and non-negative");
    }

    std::vector<double> sorted_samples = samples_ms;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    BenchmarkStatistics statistics;
    statistics.sample_count = samples_ms.size();
    statistics.minimum_ms = sorted_samples.front();
    statistics.mean_ms =
        std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
        static_cast<double>(samples_ms.size());
    statistics.median_ms = percentile(sorted_samples, 0.5);
    statistics.p95_ms = percentile(sorted_samples, 0.95);

    if (samples_ms.size() > 1) {
        double squared_difference_sum = 0.0;
        for (const double sample : samples_ms) {
            const double difference = sample - statistics.mean_ms;
            squared_difference_sum += difference * difference;
        }
        statistics.standard_deviation_ms = std::sqrt(
            squared_difference_sum / static_cast<double>(samples_ms.size() - 1));
    }
    return statistics;
}

BenchmarkMetadata make_benchmark_metadata(
    std::string operation,
    std::string implementation,
    std::string data_type,
    std::map<std::string, std::uint64_t> dimensions,
    const LaunchConfiguration launch,
    const int device_index) {
    const DeviceInfo device = query_device(device_index);
    const CudaVersions versions = query_cuda_versions();

    BenchmarkMetadata metadata;
    metadata.operation = std::move(operation);
    metadata.implementation = std::move(implementation);
    metadata.data_type = std::move(data_type);
    metadata.dimensions = std::move(dimensions);
    metadata.launch = launch;
    metadata.gpu_name = device.name;
    metadata.compute_capability_major = device.compute_capability_major;
    metadata.compute_capability_minor = device.compute_capability_minor;
    metadata.gpu_global_memory_bytes = device.global_memory_bytes;
    metadata.cuda_driver_version = format_cuda_version(versions.driver);
    metadata.cuda_runtime_version = format_cuda_version(versions.runtime);
    metadata.build_type = WARPFORGE_BUILD_TYPE;
    metadata.compiler = WARPFORGE_COMPILER;
    metadata.git_commit = WARPFORGE_GIT_COMMIT;
    metadata.timestamp_utc = current_utc_timestamp();
    return metadata;
}

void write_benchmark_json(
    const BenchmarkResult& result,
    const std::filesystem::path& output_path) {
    if (output_path.empty()) {
        throw std::invalid_argument("benchmark output path must not be empty");
    }
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open benchmark output: " + output_path.string());
    }

    const auto& metadata = result.metadata;
    const auto& launch = metadata.launch;
    const auto& statistics = result.statistics;
    const auto& validation = result.validation;

    output << "{\n";
    output << "  \"schema_version\": " << result.schema_version << ",\n";
    output << "  \"metadata\": {\n";
    output << "    \"operation\": \"" << escape_json(metadata.operation) << "\",\n";
    output << "    \"implementation\": \"" << escape_json(metadata.implementation) << "\",\n";
    output << "    \"data_type\": \"" << escape_json(metadata.data_type) << "\",\n";
    output << "    \"timing_scope\": \"" << escape_json(metadata.timing_scope) << "\",\n";
    output << "    \"dimensions\": ";
    write_dimensions(output, metadata.dimensions);
    output << ",\n";
    output << "    \"launch\": {\n";
    output << "      \"grid\": [" << launch.grid[0] << ", " << launch.grid[1] << ", "
           << launch.grid[2] << "],\n";
    output << "      \"block\": [" << launch.block[0] << ", " << launch.block[1] << ", "
           << launch.block[2] << "],\n";
    output << "      \"dynamic_shared_memory_bytes\": "
           << launch.dynamic_shared_memory_bytes << "\n";
    output << "    },\n";
    output << "    \"gpu_name\": \"" << escape_json(metadata.gpu_name) << "\",\n";
    output << "    \"compute_capability\": \"" << metadata.compute_capability_major << '.'
           << metadata.compute_capability_minor << "\",\n";
    output << "    \"gpu_global_memory_bytes\": " << metadata.gpu_global_memory_bytes << ",\n";
    output << "    \"cuda_driver_version\": \"" << escape_json(metadata.cuda_driver_version)
           << "\",\n";
    output << "    \"cuda_runtime_version\": \"" << escape_json(metadata.cuda_runtime_version)
           << "\",\n";
    output << "    \"build_type\": \"" << escape_json(metadata.build_type) << "\",\n";
    output << "    \"compiler\": \"" << escape_json(metadata.compiler) << "\",\n";
    output << "    \"git_commit\": \"" << escape_json(metadata.git_commit) << "\",\n";
    output << "    \"timestamp_utc\": \"" << escape_json(metadata.timestamp_utc) << "\"\n";
    output << "  },\n";
    output << "  \"config\": {\n";
    output << "    \"warmup_iterations\": " << result.config.warmup_iterations << ",\n";
    output << "    \"measurement_iterations\": " << result.config.measurement_iterations << ",\n";
    output << "    \"seed\": " << result.config.seed << "\n";
    output << "  },\n";
    output << "  \"statistics_ms\": {\n";
    output << "    \"sample_count\": " << statistics.sample_count << ",\n";
    output << "    \"minimum\": ";
    write_number(output, statistics.minimum_ms);
    output << ",\n    \"mean\": ";
    write_number(output, statistics.mean_ms);
    output << ",\n    \"median\": ";
    write_number(output, statistics.median_ms);
    output << ",\n    \"p95\": ";
    write_number(output, statistics.p95_ms);
    output << ",\n    \"standard_deviation\": ";
    write_number(output, statistics.standard_deviation_ms);
    output << "\n  },\n";
    output << "  \"correctness\": {\n";
    output << "    \"passed\": " << (validation.passed ? "true" : "false") << ",\n";
    output << "    \"element_count\": " << validation.element_count << ",\n";
    output << "    \"failure_count\": " << validation.failure_count << ",\n";
    output << "    \"worst_index\": " << validation.worst_index << ",\n";
    output << "    \"max_absolute_error\": ";
    write_number(output, validation.max_absolute_error);
    output << ",\n    \"mean_absolute_error\": ";
    write_number(output, validation.mean_absolute_error);
    output << ",\n    \"max_relative_error\": ";
    write_number(output, validation.max_relative_error);
    output << ",\n    \"expected_at_worst\": ";
    write_number(output, validation.expected_at_worst);
    output << ",\n    \"actual_at_worst\": ";
    write_number(output, validation.actual_at_worst);
    output << "\n  },\n";
    output << "  \"metrics\": ";
    write_metrics(output, result.metrics);
    output << ",\n";
    output << "  \"samples_ms\": [";
    for (std::size_t index = 0; index < result.samples_ms.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        write_number(output, result.samples_ms[index]);
    }
    output << "]\n";
    output << "}\n";

    if (!output) {
        throw std::runtime_error("failed while writing benchmark output: " + output_path.string());
    }
}

}  // namespace warpforge
