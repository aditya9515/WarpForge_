#include <warpforge/benchmark.hpp>
#include <warpforge/cuda_check.cuh>

#include <stdexcept>
#include <utility>
#include <vector>

namespace warpforge {
namespace {

class EventPair final {
public:
    EventPair() {
        CUDA_CHECK(cudaEventCreate(&start_));
        try {
            CUDA_CHECK(cudaEventCreate(&stop_));
        } catch (...) {
            cudaEventDestroy(start_);
            throw;
        }
    }

    ~EventPair() noexcept {
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    [[nodiscard]] cudaEvent_t start() const noexcept {
        return start_;
    }

    [[nodiscard]] cudaEvent_t stop() const noexcept {
        return stop_;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

}  // namespace

std::vector<double> measure_cuda_kernel(
    const BenchmarkConfig& config,
    cudaStream_t stream,
    const CudaWork& work) {
    if (config.measurement_iterations == 0) {
        throw std::invalid_argument("measurement_iterations must be greater than zero");
    }
    if (!work) {
        throw std::invalid_argument("CUDA benchmark work callback must not be empty");
    }

    EventPair events;

    for (std::size_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
        work(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples_ms;
    samples_ms.reserve(config.measurement_iterations);
    for (std::size_t iteration = 0; iteration < config.measurement_iterations; ++iteration) {
        CUDA_CHECK(cudaEventRecord(events.start(), stream));
        work(stream);
        CUDA_CHECK(cudaEventRecord(events.stop(), stream));
        CUDA_CHECK(cudaEventSynchronize(events.stop()));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()));
        samples_ms.push_back(static_cast<double>(elapsed_ms));
    }
    return samples_ms;
}

}  // namespace warpforge
