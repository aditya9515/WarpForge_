#include <warpforge/benchmark_statistics.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

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

}  // namespace warpforge
