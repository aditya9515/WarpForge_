#pragma once

#include <cstddef>

namespace warpforge {

struct Tolerance final {
    double absolute{1.0e-5};
    double relative{1.0e-5};
};

struct ValidationResult final {
    bool passed{};
    std::size_t element_count{};
    std::size_t failure_count{};
    std::size_t worst_index{};
    double max_absolute_error{};
    double mean_absolute_error{};
    double max_relative_error{};
    float expected_at_worst{};
    float actual_at_worst{};
};

[[nodiscard]] ValidationResult validate_fp32(
    const float* expected,
    const float* actual,
    std::size_t element_count,
    Tolerance tolerance = {});

}  // namespace warpforge
