#include <warpforge/benchmark_statistics.hpp>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require_close(const double actual, const double expected, const double tolerance,
                   const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        const auto statistics = warpforge::summarize_samples({1.0, 2.0, 3.0, 4.0});
        if (statistics.sample_count != 4) {
            throw std::runtime_error("sample count mismatch");
        }
        require_close(statistics.minimum_ms, 1.0, 1.0e-12, "minimum mismatch");
        require_close(statistics.mean_ms, 2.5, 1.0e-12, "mean mismatch");
        require_close(statistics.median_ms, 2.5, 1.0e-12, "median mismatch");
        require_close(statistics.p95_ms, 3.85, 1.0e-12, "p95 mismatch");
        require_close(statistics.standard_deviation_ms, std::sqrt(5.0 / 3.0), 1.0e-12,
                      "sample standard deviation mismatch");

        const auto single = warpforge::summarize_samples({7.0});
        require_close(single.standard_deviation_ms, 0.0, 1.0e-12, "single-sample deviation");

        bool empty_threw = false;
        try {
            static_cast<void>(warpforge::summarize_samples({}));
        } catch (const std::invalid_argument&) {
            empty_threw = true;
        }
        if (!empty_threw) {
            throw std::runtime_error("empty sample sets should throw");
        }

        bool negative_threw = false;
        try {
            static_cast<void>(warpforge::summarize_samples({1.0, -1.0}));
        } catch (const std::invalid_argument&) {
            negative_threw = true;
        }
        if (!negative_threw) {
            throw std::runtime_error("negative samples should throw");
        }

        const auto zeros = warpforge::summarize_samples({0.0, 0.0, 0.0});
        require_close(zeros.p95_ms, 0.0, 0.0, "zero samples should remain zero");
        require_close(zeros.standard_deviation_ms, 0.0, 0.0, "zero deviation mismatch");

        for (const double invalid :
             {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            bool invalid_threw = false;
            try {
                static_cast<void>(warpforge::summarize_samples({1.0, invalid}));
            } catch (const std::invalid_argument&) {
                invalid_threw = true;
            }
            if (!invalid_threw) {
                throw std::runtime_error("non-finite samples should throw");
            }
        }

        std::cout << "Benchmark statistics tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark statistics tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
