#include <warpforge/device.hpp>

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

std::string format_cuda_version(const int version) {
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + '.' + std::to_string(minor);
}

double bytes_to_mib(const std::size_t bytes) {
    constexpr double bytes_per_mib = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / bytes_per_mib;
}

double bytes_to_kib(const std::size_t bytes) {
    constexpr double bytes_per_kib = 1024.0;
    return static_cast<double>(bytes) / bytes_per_kib;
}

const char* yes_no(const bool value) {
    return value ? "yes" : "no";
}

void print_dimensions(const std::array<int, 3>& dimensions) {
    std::cout << dimensions[0] << " x " << dimensions[1] << " x " << dimensions[2];
}

void print_device(const warpforge::DeviceInfo& device) {
    std::cout << "\nDevice " << device.index << ": " << device.name << '\n';
    std::cout << "  Compute capability:                " << device.compute_capability_major << '.'
              << device.compute_capability_minor << '\n';
    std::cout << "  Global memory:                     " << std::fixed << std::setprecision(0)
              << bytes_to_mib(device.global_memory_bytes) << " MiB\n";
    std::cout << "  Streaming multiprocessors:         " << device.multiprocessor_count << '\n';
    std::cout << "  Warp size:                         " << device.warp_size << '\n';
    std::cout << "  Maximum threads per block:         " << device.max_threads_per_block << '\n';
    std::cout << "  Maximum threads per multiprocessor:" << ' '
              << device.max_threads_per_multiprocessor << '\n';
    std::cout << "  Maximum block dimensions:          ";
    print_dimensions(device.max_block_dimensions);
    std::cout << '\n';
    std::cout << "  Maximum grid dimensions:           ";
    print_dimensions(device.max_grid_dimensions);
    std::cout << '\n';
    std::cout << "  Shared memory per block:           "
              << bytes_to_kib(device.shared_memory_per_block_bytes) << " KiB\n";
    std::cout << "  Opt-in shared memory per block:    "
              << bytes_to_kib(device.shared_memory_per_block_optin_bytes) << " KiB\n";
    std::cout << "  Shared memory per multiprocessor:  "
              << bytes_to_kib(device.shared_memory_per_multiprocessor_bytes) << " KiB\n";
    std::cout << "  Concurrent kernels:                " << yes_no(device.concurrent_kernels) << '\n';
    std::cout << "  Unified addressing:                " << yes_no(device.unified_addressing) << '\n';
    std::cout << "  Managed memory:                    " << yes_no(device.managed_memory) << '\n';
    std::cout << "  Cooperative launch:                " << yes_no(device.cooperative_launch) << '\n';
}

}  // namespace

int main() {
    try {
        const auto versions = warpforge::query_cuda_versions();
        const int count = warpforge::device_count();

        std::cout << "WarpForge CUDA device information\n";
        std::cout << "CUDA driver API version:  " << format_cuda_version(versions.driver) << '\n';
        std::cout << "CUDA runtime API version: " << format_cuda_version(versions.runtime) << '\n';
        std::cout << "CUDA device count:        " << count << '\n';

        for (int index = 0; index < count; ++index) {
            print_device(warpforge::query_device(index));
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "device_info failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
