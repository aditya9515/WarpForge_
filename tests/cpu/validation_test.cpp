#include <warpforge/validation.hpp>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_exact_and_empty_inputs() {
    const float values[] = {1.0F, -2.0F, 0.0F};
    const auto exact = warpforge::validate_fp32(values, values, 3);
    require(exact.passed, "exact values should pass");
    require(exact.failure_count == 0, "exact values should not report failures");
    require(exact.max_absolute_error == 0.0, "exact values should have zero error");

    const auto empty = warpforge::validate_fp32(nullptr, nullptr, 0);
    require(empty.passed, "empty ranges should pass");
    require(empty.element_count == 0, "empty range size should be zero");
}

void test_tolerance_and_error_summary() {
    const float expected[] = {100.0F, 1.0F, -4.0F};
    const float within_tolerance[] = {100.0005F, 1.000001F, -4.000001F};
    const auto passing =
        warpforge::validate_fp32(expected, within_tolerance, 3, {1.0e-5, 1.0e-5});
    require(passing.passed, "combined absolute/relative tolerance should pass");

    const float failing_values[] = {101.0F, 1.0F, -3.5F};
    const auto failing = warpforge::validate_fp32(expected, failing_values, 3, {1.0e-5, 1.0e-5});
    require(!failing.passed, "out-of-tolerance values should fail");
    require(failing.failure_count == 2, "two values should fail");
    require(failing.worst_index == 0, "largest absolute error should be at index zero");
    require(std::abs(failing.max_absolute_error - 1.0) < 1.0e-12, "maximum error mismatch");
    require(failing.mean_absolute_error > 0.0, "mean error should be reported");
}

void test_non_finite_and_invalid_inputs() {
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float expected[] = {infinity, nan};
    const float actual[] = {infinity, nan};
    const auto result = warpforge::validate_fp32(expected, actual, 2);
    require(!result.passed, "matching NaNs are not accepted as numerically equal");
    require(result.failure_count == 1, "only the NaN pair should fail");

    bool invalid_tolerance_threw = false;
    try {
        static_cast<void>(warpforge::validate_fp32(expected, actual, 2, {-1.0, 0.0}));
    } catch (const std::invalid_argument&) {
        invalid_tolerance_threw = true;
    }
    require(invalid_tolerance_threw, "negative tolerances should throw");

    bool null_input_threw = false;
    try {
        static_cast<void>(warpforge::validate_fp32(nullptr, actual, 1));
    } catch (const std::invalid_argument&) {
        null_input_threw = true;
    }
    require(null_input_threw, "null non-empty inputs should throw");
}

}  // namespace

int main() {
    try {
        test_exact_and_empty_inputs();
        test_tolerance_and_error_summary();
        test_non_finite_and_invalid_inputs();
        std::cout << "FP32 validation tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FP32 validation tests failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
