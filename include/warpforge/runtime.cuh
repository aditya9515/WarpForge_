#pragma once

#include "warpforge/cuda_check.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace warpforge {

enum class DType {
    fp32,
    fp16,
};

[[nodiscard]] const char* dtype_name(DType dtype) noexcept;
[[nodiscard]] std::size_t dtype_size(DType dtype);

class TensorShape final {
  public:
    TensorShape() = default;
    TensorShape(std::initializer_list<std::size_t> dimensions);
    explicit TensorShape(std::vector<std::size_t> dimensions);

    [[nodiscard]] bool empty() const noexcept {
        return element_count_ == 0;
    }
    [[nodiscard]] std::size_t rank() const noexcept {
        return dimensions_.size();
    }
    [[nodiscard]] std::size_t dimension(std::size_t index) const;
    [[nodiscard]] const std::vector<std::size_t>& dimensions() const noexcept {
        return dimensions_;
    }
    [[nodiscard]] std::size_t element_count() const noexcept {
        return element_count_;
    }
    [[nodiscard]] std::size_t byte_size(DType dtype) const;

  private:
    void update_element_count();

    std::vector<std::size_t> dimensions_;
    std::size_t element_count_{0};
};

namespace detail {

inline std::size_t checked_byte_count(const std::size_t count, const std::size_t element_size) {
    if (element_size != 0 && count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("device allocation byte count overflow");
    }
    return count * element_size;
}

template <typename T> struct DTypeFor;

template <> struct DTypeFor<float> {
    static constexpr DType value = DType::fp32;
};

template <> struct DTypeFor<__half> {
    static constexpr DType value = DType::fp16;
};

} // namespace detail

template <typename T> class DeviceBuffer final {
  public:
    static_assert(std::is_trivially_copyable_v<T>,
                  "DeviceBuffer requires trivially copyable elements");

    DeviceBuffer() noexcept = default;

    explicit DeviceBuffer(const std::size_t count) {
        allocate(count);
    }

    ~DeviceBuffer() noexcept {
        reset_noexcept();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)), count_(std::exchange(other.count_, 0)) {
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset_noexcept();
            pointer_ = std::exchange(other.pointer_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(const std::size_t count) {
        if (count == 0) {
            reset();
            return;
        }

        const std::size_t requested_bytes = detail::checked_byte_count(count, sizeof(T));
        T* replacement = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replacement), requested_bytes));

        DeviceBuffer temporary;
        temporary.pointer_ = replacement;
        temporary.count_ = count;
        swap(temporary);
        temporary.reset();
    }

    // Unlike destruction, reset reports cudaFree failures to the caller.
    void reset() {
        if (pointer_ == nullptr) {
            count_ = 0;
            return;
        }
        CUDA_CHECK(cudaFree(pointer_));
        pointer_ = nullptr;
        count_ = 0;
    }

    [[nodiscard]] T* release() noexcept {
        count_ = 0;
        return std::exchange(pointer_, nullptr);
    }

    void swap(DeviceBuffer& other) noexcept {
        std::swap(pointer_, other.pointer_);
        std::swap(count_, other.count_);
    }

    [[nodiscard]] T* data() noexcept {
        return pointer_;
    }
    [[nodiscard]] const T* data() const noexcept {
        return pointer_;
    }
    [[nodiscard]] T* get() noexcept {
        return pointer_;
    }
    [[nodiscard]] const T* get() const noexcept {
        return pointer_;
    }
    [[nodiscard]] T* native_handle() noexcept {
        return pointer_;
    }
    [[nodiscard]] const T* native_handle() const noexcept {
        return pointer_;
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }
    [[nodiscard]] std::size_t byte_size() const noexcept {
        return count_ * sizeof(T);
    }
    [[nodiscard]] bool empty() const noexcept {
        return count_ == 0;
    }

    void copy_from_host_async(const T* source, const std::size_t count, const cudaStream_t stream,
                              const std::size_t destination_offset = 0) {
        check_range(destination_offset, count);
        if (count == 0) {
            return;
        }
        if (source == nullptr) {
            throw std::invalid_argument("host source is null for a non-empty copy");
        }
        CUDA_CHECK(cudaMemcpyAsync(pointer_ + destination_offset, source,
                                   detail::checked_byte_count(count, sizeof(T)),
                                   cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host_async(T* destination, const std::size_t count, const cudaStream_t stream,
                            const std::size_t source_offset = 0) const {
        check_range(source_offset, count);
        if (count == 0) {
            return;
        }
        if (destination == nullptr) {
            throw std::invalid_argument("host destination is null for a non-empty copy");
        }
        CUDA_CHECK(cudaMemcpyAsync(destination, pointer_ + source_offset,
                                   detail::checked_byte_count(count, sizeof(T)),
                                   cudaMemcpyDeviceToHost, stream));
    }

  private:
    void check_range(const std::size_t offset, const std::size_t count) const {
        if (offset > count_ || count > count_ - offset) {
            throw std::out_of_range("device buffer copy exceeds allocation");
        }
    }

    void reset_noexcept() noexcept {
        if (pointer_ != nullptr) {
            (void)cudaFree(pointer_);
            pointer_ = nullptr;
            count_ = 0;
        }
    }

    T* pointer_{nullptr};
    std::size_t count_{0};
};

class CudaEvent;

class CudaStream final {
  public:
    CudaStream();
    explicit CudaStream(unsigned int flags);
    ~CudaStream() noexcept;

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    void synchronize() const;
    void wait(const CudaEvent& event) const;
    void reset();
    [[nodiscard]] cudaStream_t release() noexcept;
    [[nodiscard]] cudaStream_t get() const noexcept {
        return stream_;
    }
    [[nodiscard]] cudaStream_t native_handle() const noexcept {
        return stream_;
    }

  private:
    void reset_noexcept() noexcept;
    cudaStream_t stream_{nullptr};
};

class CudaEvent final {
  public:
    CudaEvent();
    explicit CudaEvent(unsigned int flags);
    ~CudaEvent() noexcept;

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    void record(cudaStream_t stream = nullptr) const;
    void record(const CudaStream& stream) const {
        record(stream.native_handle());
    }
    void synchronize() const;
    [[nodiscard]] float elapsed_ms_since(const CudaEvent& start) const;
    void reset();
    [[nodiscard]] cudaEvent_t release() noexcept;
    [[nodiscard]] cudaEvent_t get() const noexcept {
        return event_;
    }
    [[nodiscard]] cudaEvent_t native_handle() const noexcept {
        return event_;
    }

  private:
    void reset_noexcept() noexcept;
    cudaEvent_t event_{nullptr};
};

class TensorView final {
  public:
    TensorView() = default;
    TensorView(void* data, TensorShape shape, DType dtype);

    [[nodiscard]] void* data() noexcept {
        return data_;
    }
    [[nodiscard]] const void* data() const noexcept {
        return data_;
    }
    [[nodiscard]] void* native_handle() noexcept {
        return data_;
    }
    [[nodiscard]] const void* native_handle() const noexcept {
        return data_;
    }
    [[nodiscard]] const TensorShape& shape() const noexcept {
        return shape_;
    }
    [[nodiscard]] DType dtype() const noexcept {
        return dtype_;
    }
    [[nodiscard]] std::size_t element_count() const noexcept {
        return shape_.element_count();
    }
    [[nodiscard]] std::size_t byte_size() const {
        return shape_.byte_size(dtype_);
    }
    [[nodiscard]] bool empty() const noexcept {
        return shape_.empty();
    }

    template <typename T> [[nodiscard]] T* data_as() {
        validate_type<T>();
        return static_cast<T*>(data_);
    }

    template <typename T> [[nodiscard]] const T* data_as() const {
        validate_type<T>();
        return static_cast<const T*>(data_);
    }

  private:
    template <typename T> void validate_type() const {
        using Element = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Element, float> || std::is_same_v<Element, __half>) {
            if (detail::DTypeFor<Element>::value != dtype_) {
                throw std::invalid_argument(
                    "TensorView dtype does not match requested element type");
            }
        } else {
            static_assert(std::is_same_v<Element, float> || std::is_same_v<Element, __half>,
                          "TensorView supports only float and __half");
        }
    }

    void* data_{nullptr};
    TensorShape shape_;
    DType dtype_{DType::fp32};
};

class Tensor final {
  public:
    Tensor() = default;
    Tensor(TensorShape shape, DType dtype);
    ~Tensor() noexcept = default;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void resize(TensorShape shape, DType dtype);
    void reset();

    [[nodiscard]] TensorView view() {
        return TensorView(storage_.data(), shape_, dtype_);
    }
    [[nodiscard]] void* native_handle() noexcept {
        return storage_.data();
    }
    [[nodiscard]] const void* native_handle() const noexcept {
        return storage_.data();
    }
    [[nodiscard]] const TensorShape& shape() const noexcept {
        return shape_;
    }
    [[nodiscard]] DType dtype() const noexcept {
        return dtype_;
    }
    [[nodiscard]] std::size_t element_count() const noexcept {
        return shape_.element_count();
    }
    [[nodiscard]] std::size_t byte_size() const noexcept {
        return storage_.size();
    }
    [[nodiscard]] bool empty() const noexcept {
        return storage_.empty();
    }

    void copy_from_host_async(const void* source, std::size_t bytes, cudaStream_t stream);
    void copy_to_host_async(void* destination, std::size_t bytes, cudaStream_t stream) const;

  private:
    DeviceBuffer<std::byte> storage_;
    TensorShape shape_;
    DType dtype_{DType::fp32};
};

class DeviceWorkspace final {
  public:
    explicit DeviceWorkspace(std::size_t capacity_bytes = 0);
    ~DeviceWorkspace() noexcept = default;

    DeviceWorkspace(const DeviceWorkspace&) = delete;
    DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;
    DeviceWorkspace(DeviceWorkspace&& other) noexcept;
    DeviceWorkspace& operator=(DeviceWorkspace&& other) noexcept;

    void reserve(std::size_t capacity_bytes);
    void clear() noexcept {
        used_bytes_ = 0;
    }
    void reset();

    [[nodiscard]] void* allocate_bytes(std::size_t bytes, std::size_t alignment = 256);
    [[nodiscard]] TensorView allocate_view(const TensorShape& shape, DType dtype,
                                           std::size_t alignment = 256);

    [[nodiscard]] void* native_handle() noexcept {
        return storage_.data();
    }
    [[nodiscard]] const void* native_handle() const noexcept {
        return storage_.data();
    }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept {
        return storage_.size();
    }
    [[nodiscard]] std::size_t used_bytes() const noexcept {
        return used_bytes_;
    }
    [[nodiscard]] std::size_t remaining_bytes() const noexcept {
        return storage_.size() - used_bytes_;
    }

  private:
    DeviceBuffer<std::byte> storage_;
    std::size_t used_bytes_{0};
};

static_assert(std::is_nothrow_destructible_v<DeviceBuffer<float>>);
static_assert(std::is_nothrow_destructible_v<CudaStream>);
static_assert(std::is_nothrow_destructible_v<CudaEvent>);
static_assert(std::is_nothrow_destructible_v<Tensor>);
static_assert(std::is_nothrow_destructible_v<DeviceWorkspace>);

} // namespace warpforge
