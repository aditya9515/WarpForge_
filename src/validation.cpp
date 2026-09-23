#include <warpforge/validation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace warpforge {

ValidationResult validate_fp32(const float* expected, const float* actual,
                               const std::size_t element_count, const Tolerance tolerance) {
    if (!std::isfinite(tolerance.absolute) || !std::isfinite(tolerance.relative) ||
        tolerance.absolute < 0.0 || tolerance.relative < 0.0) {
        throw std::invalid_argument("validation tolerances must be finite and non-negative");
    }
    if (element_count > 0 && (expected == nullptr || actual == nullptr)) {
        throw std::invalid_argument("validation inputs must not be null for a non-empty range");
    }

    ValidationResult result;
    result.passed = true;
    result.element_count = element_count;

    double absolute_error_sum = 0.0;
    for (std::size_t index = 0; index < element_count; ++index) {
        const double expected_value = static_cast<double>(expected[index]);
        const double actual_value = static_cast<double>(actual[index]);

        double absolute_error = 0.0;
        double relative_error = 0.0;
        bool value_passed = false;

        if (expected_value == actual_value) {
            value_passed = true;
        } else if (std::isfinite(expected_value) && std::isfinite(actual_value)) {
            absolute_error = std::abs(actual_value - expected_value);
            const double allowed_error =
                tolerance.absolute + tolerance.relative * std::abs(expected_value);
            value_passed = absolute_error <= allowed_error;

            if (expected_value == 0.0) {
                relative_error =
                    absolute_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
            } else {
                relative_error = absolute_error / std::abs(expected_value);
            }
        } else {
            absolute_error = std::numeric_limits<double>::infinity();
            relative_error = std::numeric_limits<double>::infinity();
        }

        absolute_error_sum += absolute_error;
        if (absolute_error > result.max_absolute_error) {
            result.max_absolute_error = absolute_error;
            result.worst_index = index;
            result.expected_at_worst = expected[index];
            result.actual_at_worst = actual[index];
        }
        result.max_relative_error = std::max(result.max_relative_error, relative_error);

        if (!value_passed) {
            result.passed = false;
            ++result.failure_count;
        }
    }

    if (element_count > 0) {
        result.mean_absolute_error = absolute_error_sum / static_cast<double>(element_count);
    }
    return result;
}

} // namespace warpforge
