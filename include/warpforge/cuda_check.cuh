#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string_view>

namespace warpforge::detail {

inline void check_cuda(
    const cudaError_t status,
    const std::string_view expression,
    const std::string_view file,
    const int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA call failed: " << expression << " at " << file << ':' << line
            << " [" << cudaGetErrorName(status) << ", code " << static_cast<int>(status)
            << "]: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

}  // namespace warpforge::detail

// A successful kernel launch only confirms that work was enqueued. Launch errors
// can be checked immediately with CUDA_CHECK(cudaGetLastError()), while execution
// errors surface at a later synchronization or another synchronizing API call.
#define CUDA_CHECK(expression) \
    ::warpforge::detail::check_cuda((expression), #expression, __FILE__, __LINE__)
