#pragma once

#include <array>
#include <cstddef>
#include <string>

namespace warpforge {

struct CudaVersions final {
    int driver{};
    int runtime{};
};

struct DeviceInfo final {
    int index{};
    std::string name;
    int compute_capability_major{};
    int compute_capability_minor{};
    std::size_t global_memory_bytes{};
    int multiprocessor_count{};
    int warp_size{};
    int max_threads_per_block{};
    int max_threads_per_multiprocessor{};
    std::array<int, 3> max_block_dimensions{};
    std::array<int, 3> max_grid_dimensions{};
    std::size_t shared_memory_per_block_bytes{};
    std::size_t shared_memory_per_block_optin_bytes{};
    std::size_t shared_memory_per_multiprocessor_bytes{};
    bool concurrent_kernels{};
    bool unified_addressing{};
    bool managed_memory{};
    bool cooperative_launch{};
};

[[nodiscard]] CudaVersions query_cuda_versions();
[[nodiscard]] int device_count();
[[nodiscard]] DeviceInfo query_device(int device_index);

} // namespace warpforge
