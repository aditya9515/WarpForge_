#include <warpforge/cuda_check.cuh>
#include <warpforge/device.hpp>

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace warpforge {

CudaVersions query_cuda_versions() {
    CudaVersions versions;
    CUDA_CHECK(cudaDriverGetVersion(&versions.driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&versions.runtime));
    return versions;
}

int device_count() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceInfo query_device(const int device_index) {
    const int count = device_count();
    if (device_index < 0 || device_index >= count) {
        std::ostringstream message;
        message << "CUDA device index " << device_index << " is out of range [0, " << count << ')';
        throw std::out_of_range(message.str());
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));

    DeviceInfo info;
    info.index = device_index;
    info.name = properties.name;
    info.compute_capability_major = properties.major;
    info.compute_capability_minor = properties.minor;
    info.global_memory_bytes = properties.totalGlobalMem;
    info.multiprocessor_count = properties.multiProcessorCount;
    info.warp_size = properties.warpSize;
    info.max_threads_per_block = properties.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = properties.maxThreadsPerMultiProcessor;
    info.max_block_dimensions = {
        properties.maxThreadsDim[0],
        properties.maxThreadsDim[1],
        properties.maxThreadsDim[2],
    };
    info.max_grid_dimensions = {
        properties.maxGridSize[0],
        properties.maxGridSize[1],
        properties.maxGridSize[2],
    };
    info.shared_memory_per_block_bytes = properties.sharedMemPerBlock;
    info.shared_memory_per_block_optin_bytes = properties.sharedMemPerBlockOptin;
    info.shared_memory_per_multiprocessor_bytes = properties.sharedMemPerMultiprocessor;
    info.concurrent_kernels = properties.concurrentKernels != 0;
    info.unified_addressing = properties.unifiedAddressing != 0;
    info.managed_memory = properties.managedMemory != 0;
    info.cooperative_launch = properties.cooperativeLaunch != 0;
    return info;
}

} // namespace warpforge
