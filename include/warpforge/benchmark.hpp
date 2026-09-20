#pragma once

#include <warpforge/validation.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace warpforge {

struct BenchmarkConfig final {
    std::size_t warmup_iterations{10};
    std::size_t measurement_iterations{100};
    std::uint64_t seed{2027};
};

struct BenchmarkStatistics final {
    std::size_t sample_count{};
    double minimum_ms{};
    double mean_ms{};
    double median_ms{};
    double p95_ms{};
    double standard_deviation_ms{};
};

struct LaunchConfiguration final {
    std::array<unsigned int, 3> grid{1U, 1U, 1U};
    std::array<unsigned int, 3> block{1U, 1U, 1U};
    std::size_t dynamic_shared_memory_bytes{};
};

struct BenchmarkMetadata final {
    std::string operation;
    std::string implementation;
    std::string data_type;
    std::string timing_scope{"kernel-only"};
    std::map<std::string, std::uint64_t> dimensions;
    LaunchConfiguration launch;
    std::string gpu_name;
    int compute_capability_major{};
    int compute_capability_minor{};
    std::size_t gpu_global_memory_bytes{};
    std::string cuda_driver_version;
    std::string cuda_runtime_version;
    std::string build_type;
    std::string compiler;
    std::string git_commit;
    std::string timestamp_utc;
};

struct BenchmarkResult final {
    int schema_version{1};
    BenchmarkConfig config;
    BenchmarkMetadata metadata;
    BenchmarkStatistics statistics;
    ValidationResult validation;
    std::map<std::string, double> metrics;
    std::vector<double> samples_ms;
};

using CudaWork = std::function<void(cudaStream_t)>;

[[nodiscard]] BenchmarkStatistics summarize_samples(const std::vector<double>& samples_ms);

[[nodiscard]] std::vector<double> measure_cuda_kernel(
    const BenchmarkConfig& config,
    cudaStream_t stream,
    const CudaWork& work);

[[nodiscard]] BenchmarkMetadata make_benchmark_metadata(
    std::string operation,
    std::string implementation,
    std::string data_type,
    std::map<std::string, std::uint64_t> dimensions,
    LaunchConfiguration launch,
    int device_index = 0);

void write_benchmark_json(const BenchmarkResult& result, const std::filesystem::path& output_path);

}  // namespace warpforge
