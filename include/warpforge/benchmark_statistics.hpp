#pragma once

#include <cstddef>
#include <vector>

namespace warpforge {

struct BenchmarkStatistics final {
    std::size_t sample_count{};
    double minimum_ms{};
    double mean_ms{};
    double median_ms{};
    double p95_ms{};
    double standard_deviation_ms{};
};

[[nodiscard]] BenchmarkStatistics summarize_samples(const std::vector<double>& samples_ms);

} // namespace warpforge
