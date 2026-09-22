#include "warpforge/runtime.cuh"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace warpforge {
namespace {

[[nodiscard]] std::size_t checked_element_count(
    const std::vector<std::size_t>& dimensions) {
    if (dimensions.empty()) {
        return 0;
    }

    std::size_t count = 1;
    for (const std::size_t dimension : dimensions) {
        if (dimension == 0) {
            return 0;
        }
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor shape element count overflow");
        }
        count *= dimension;
    }
    return count;
}

void require_stream(const cudaStream_t stream) {
    if (stream == nullptr) {
        throw std::logic_error("CUDA stream has no native handle");
    }
}

void require_event(const cudaEvent_t event) {
    if (event == nullptr) {
        throw std::logic_error("CUDA event has no native handle");
    }
}

}  // namespace

const char* dtype_name(const DType dtype) noexcept {
    switch (dtype) {
        case DType::fp32:
            return "fp32";
        case DType::fp16:
            return "fp16";
    }
    return "unknown";
}

std::size_t dtype_size(const DType dtype) {
    switch (dtype) {
        case DType::fp32:
            return sizeof(float);
        case DType::fp16:
            return sizeof(__half);
    }
    throw std::invalid_argument("unsupported WarpForge dtype");
}

TensorShape::TensorShape(const std::initializer_list<std::size_t> dimensions)
    : dimensions_(dimensions) {
    update_element_count();
}

TensorShape::TensorShape(std::vector<std::size_t> dimensions)
    : dimensions_(std::move(dimensions)) {
    update_element_count();
}

std::size_t TensorShape::dimension(const std::size_t index) const {
    return dimensions_.at(index);
}

std::size_t TensorShape::byte_size(const DType dtype) const {
    return detail::checked_byte_count(element_count_, dtype_size(dtype));
}

void TensorShape::update_element_count() {
    element_count_ = checked_element_count(dimensions_);
}

CudaStream::CudaStream() : CudaStream(cudaStreamNonBlocking) {}

CudaStream::CudaStream(const unsigned int flags) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

CudaStream::~CudaStream() noexcept {
    reset_noexcept();
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        reset_noexcept();
        stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
}

void CudaStream::synchronize() const {
    require_stream(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CudaStream::wait(const CudaEvent& event) const {
    require_stream(stream_);
    require_event(event.native_handle());
    CUDA_CHECK(cudaStreamWaitEvent(stream_, event.native_handle(), 0));
}

void CudaStream::reset() {
    if (stream_ == nullptr) {
        return;
    }
    CUDA_CHECK(cudaStreamDestroy(stream_));
    stream_ = nullptr;
}

cudaStream_t CudaStream::release() noexcept {
    return std::exchange(stream_, nullptr);
}

void CudaStream::reset_noexcept() noexcept {
    if (stream_ != nullptr) {
        (void)cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

CudaEvent::CudaEvent() : CudaEvent(cudaEventDefault) {}

CudaEvent::CudaEvent(const unsigned int flags) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

CudaEvent::~CudaEvent() noexcept {
    reset_noexcept();
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)) {}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
        reset_noexcept();
        event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
}

void CudaEvent::record(const cudaStream_t stream) const {
    require_event(event_);
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaEvent::synchronize() const {
    require_event(event_);
    CUDA_CHECK(cudaEventSynchronize(event_));
}

float CudaEvent::elapsed_ms_since(const CudaEvent& start) const {
    require_event(start.event_);
    require_event(event_);
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.event_, event_));
    return elapsed_ms;
}

void CudaEvent::reset() {
    if (event_ == nullptr) {
        return;
    }
    CUDA_CHECK(cudaEventDestroy(event_));
    event_ = nullptr;
}

cudaEvent_t CudaEvent::release() noexcept {
    return std::exchange(event_, nullptr);
}

void CudaEvent::reset_noexcept() noexcept {
    if (event_ != nullptr) {
        (void)cudaEventDestroy(event_);
        event_ = nullptr;
    }
}

TensorView::TensorView(void* data, TensorShape shape, const DType dtype)
    : data_(data), shape_(std::move(shape)), dtype_(dtype) {
    if (!shape_.empty() && data_ == nullptr) {
        throw std::invalid_argument("non-empty TensorView requires a non-null device pointer");
    }
}

Tensor::Tensor(TensorShape shape, const DType dtype) {
    resize(std::move(shape), dtype);
}

Tensor::Tensor(Tensor&& other) noexcept
    : storage_(std::move(other.storage_)),
      shape_(std::move(other.shape_)),
      dtype_(other.dtype_) {
    other.shape_ = TensorShape{};
    other.dtype_ = DType::fp32;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        storage_ = std::move(other.storage_);
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        other.shape_ = TensorShape{};
        other.dtype_ = DType::fp32;
    }
    return *this;
}

void Tensor::resize(TensorShape shape, const DType dtype) {
    const std::size_t bytes = shape.byte_size(dtype);
    DeviceBuffer<std::byte> replacement(bytes);
    storage_.swap(replacement);
    shape_ = std::move(shape);
    dtype_ = dtype;
    replacement.reset();
}

void Tensor::reset() {
    storage_.reset();
    shape_ = TensorShape{};
    dtype_ = DType::fp32;
}

void Tensor::copy_from_host_async(
    const void* source,
    const std::size_t bytes,
    const cudaStream_t stream) {
    if (bytes > byte_size()) {
        throw std::out_of_range("host-to-tensor copy exceeds tensor storage");
    }
    if (bytes == 0) {
        return;
    }
    if (source == nullptr) {
        throw std::invalid_argument("host source is null for a non-empty tensor copy");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        storage_.data(), source, bytes, cudaMemcpyHostToDevice, stream));
}

void Tensor::copy_to_host_async(
    void* destination,
    const std::size_t bytes,
    const cudaStream_t stream) const {
    if (bytes > byte_size()) {
        throw std::out_of_range("tensor-to-host copy exceeds tensor storage");
    }
    if (bytes == 0) {
        return;
    }
    if (destination == nullptr) {
        throw std::invalid_argument("host destination is null for a non-empty tensor copy");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        destination, storage_.data(), bytes, cudaMemcpyDeviceToHost, stream));
}

DeviceWorkspace::DeviceWorkspace(const std::size_t capacity_bytes)
    : storage_(capacity_bytes) {}

DeviceWorkspace::DeviceWorkspace(DeviceWorkspace&& other) noexcept
    : storage_(std::move(other.storage_)),
      used_bytes_(std::exchange(other.used_bytes_, 0U)) {}

DeviceWorkspace& DeviceWorkspace::operator=(DeviceWorkspace&& other) noexcept {
    if (this != &other) {
        storage_ = std::move(other.storage_);
        used_bytes_ = std::exchange(other.used_bytes_, 0U);
    }
    return *this;
}

void DeviceWorkspace::reserve(const std::size_t capacity_bytes) {
    if (capacity_bytes <= storage_.size()) {
        return;
    }
    DeviceBuffer<std::byte> replacement(capacity_bytes);
    storage_.swap(replacement);
    used_bytes_ = 0;
    replacement.reset();
}

void DeviceWorkspace::reset() {
    storage_.reset();
    used_bytes_ = 0;
}

void* DeviceWorkspace::allocate_bytes(
    const std::size_t bytes,
    const std::size_t alignment) {
    if (bytes == 0) {
        return nullptr;
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("workspace alignment must be a non-zero power of two");
    }
    if (used_bytes_ > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("workspace alignment offset overflow");
    }

    const std::size_t aligned_offset = (used_bytes_ + alignment - 1) & ~(alignment - 1);
    if (aligned_offset > storage_.size() || bytes > storage_.size() - aligned_offset) {
        throw std::length_error("device workspace capacity exceeded");
    }

    std::byte* pointer = storage_.data() + aligned_offset;
    used_bytes_ = aligned_offset + bytes;
    return pointer;
}

TensorView DeviceWorkspace::allocate_view(
    const TensorShape& shape,
    const DType dtype,
    const std::size_t alignment) {
    return TensorView(allocate_bytes(shape.byte_size(dtype), alignment), shape, dtype);
}

}  // namespace warpforge
