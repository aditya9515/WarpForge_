#include <warpforge/cuda_check.cuh>
#include <warpforge/device.hpp>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        const auto versions = warpforge::query_cuda_versions();
        require(versions.driver > 0, "CUDA driver API version was not reported");
        require(versions.runtime >= 12060, "CUDA runtime is older than the 12.6 project baseline");

        const int count = warpforge::device_count();
        require(count > 0, "No CUDA devices were detected");

        const auto device = warpforge::query_device(0);
        require(!device.name.empty(), "CUDA device name is empty");
        require(device.compute_capability_major > 0, "Invalid CUDA compute capability");
        require(device.global_memory_bytes > 0, "CUDA device reports no global memory");
        require(device.multiprocessor_count > 0, "CUDA device reports no multiprocessors");
        require(device.warp_size > 0, "CUDA device reports an invalid warp size");
        require(device.max_threads_per_block > 0, "CUDA device reports no threads per block");

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "CUDA runtime smoke test passed on " << device.name << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "CUDA runtime smoke test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
