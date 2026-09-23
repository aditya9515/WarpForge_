#include <warpforge/cuda_check.cuh>
#include <warpforge/gemm.cuh>

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace warpforge {
namespace {

constexpr unsigned int tile_16 = 16U;
constexpr unsigned int tile_32 = 32U;
constexpr unsigned int register_tile = 64U;
constexpr unsigned int register_k = 16U;

void check_cublas(const cublasStatus_t status, const char* expression) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS call failed: ") + expression + " (status " +
                                 std::to_string(static_cast<int>(status)) + ")");
    }
}

[[nodiscard]] std::size_t checked_product(const std::size_t left, const std::size_t right,
                                          const char* label) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(label) + " element count overflows size_t");
    }
    return left * right;
}

void validate_problem_dimensions(const GemmProblem& problem) {
    constexpr std::size_t cublas_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (problem.m > cublas_limit || problem.n > cublas_limit || problem.k > cublas_limit) {
        throw std::out_of_range("GEMM dimensions exceed the supported 32-bit launch range");
    }
    static_cast<void>(gemm_a_elements(problem));
    static_cast<void>(gemm_b_elements(problem));
    static_cast<void>(gemm_c_elements(problem));
}

void validate_pointers(const void* a, const void* b, const void* c, const GemmProblem& problem) {
    validate_problem_dimensions(problem);
    if (!gemm_problem_is_empty(problem) && (a == nullptr || b == nullptr || c == nullptr)) {
        throw std::invalid_argument("non-empty GEMM requires non-null A, B, and C pointers");
    }
}

__global__ void naive_fp32_kernel(const float* a, const float* b, float* c, const std::size_t m,
                                  const std::size_t n, const std::size_t k) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= m || column >= n) {
        return;
    }
    float accumulator = 0.0F;
    for (std::size_t inner = 0; inner < k; ++inner) {
        accumulator = fmaf(a[row * k + inner], b[inner * n + column], accumulator);
    }
    c[row * n + column] = accumulator;
}

template <typename Input> __device__ __forceinline__ float to_float(const Input value) {
    return static_cast<float>(value);
}

template <> __device__ __forceinline__ float to_float<__half>(const __half value) {
    return __half2float(value);
}

template <typename Input>
__global__ void tiled_16_kernel(const Input* a, const Input* b, float* c, const std::size_t m,
                                const std::size_t n, const std::size_t k) {
    __shared__ float a_tile[tile_16][tile_16];
    __shared__ float b_tile[tile_16][tile_16];
    const unsigned int local_row = threadIdx.y;
    const unsigned int local_column = threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * tile_16 + local_row;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * tile_16 + local_column;
    float accumulator = 0.0F;

    for (std::size_t tile_begin = 0; tile_begin < k; tile_begin += tile_16) {
        const std::size_t a_column = tile_begin + local_column;
        const std::size_t b_row = tile_begin + local_row;
        a_tile[local_row][local_column] =
            row < m && a_column < k ? to_float(a[row * k + a_column]) : 0.0F;
        b_tile[local_row][local_column] =
            b_row < k && column < n ? to_float(b[b_row * n + column]) : 0.0F;
        __syncthreads();
#pragma unroll
        for (unsigned int inner = 0; inner < tile_16; ++inner) {
            accumulator = fmaf(a_tile[local_row][inner], b_tile[inner][local_column], accumulator);
        }
        __syncthreads();
    }
    if (row < m && column < n) {
        c[row * n + column] = accumulator;
    }
}

__global__ void coalesced_fp32_kernel(const float* a, const float* b, float* c, const std::size_t m,
                                      const std::size_t n, const std::size_t k) {
    __shared__ float a_tile[tile_32][tile_32 + 1U];
    __shared__ float b_tile[tile_32][tile_32 + 1U];
    constexpr unsigned int rows_per_thread = 4U;
    float accumulators[rows_per_thread]{};
    const unsigned int local_column = threadIdx.x;
    const unsigned int local_row_base = threadIdx.y;
    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * tile_32;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * tile_32 + local_column;

    for (std::size_t tile_begin = 0; tile_begin < k; tile_begin += tile_32) {
#pragma unroll
        for (unsigned int offset = 0; offset < rows_per_thread; ++offset) {
            const unsigned int local_row = local_row_base + offset * blockDim.y;
            const std::size_t row = block_row + local_row;
            const std::size_t inner = tile_begin + local_column;
            a_tile[local_row][local_column] = row < m && inner < k ? a[row * k + inner] : 0.0F;
            const std::size_t b_row = tile_begin + local_row;
            b_tile[local_row][local_column] =
                b_row < k && column < n ? b[b_row * n + column] : 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (unsigned int inner = 0; inner < tile_32; ++inner) {
            const float b_value = b_tile[inner][local_column];
#pragma unroll
            for (unsigned int offset = 0; offset < rows_per_thread; ++offset) {
                const unsigned int local_row = local_row_base + offset * blockDim.y;
                accumulators[offset] =
                    fmaf(a_tile[local_row][inner], b_value, accumulators[offset]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int offset = 0; offset < rows_per_thread; ++offset) {
        const std::size_t row = block_row + local_row_base + offset * blockDim.y;
        if (row < m && column < n) {
            c[row * n + column] = accumulators[offset];
        }
    }
}

__global__ void register_blocked_fp32_kernel(const float* a, const float* b, float* c,
                                             const std::size_t m, const std::size_t n,
                                             const std::size_t k) {
    __shared__ float a_tile[register_tile][register_k + 1U];
    __shared__ float b_tile[register_k][register_tile + 1U];
    constexpr unsigned int thread_tile = 4U;
    float accumulators[thread_tile][thread_tile]{};
    const unsigned int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * register_tile;
    const std::size_t block_column = static_cast<std::size_t>(blockIdx.x) * register_tile;

    for (std::size_t tile_begin = 0; tile_begin < k; tile_begin += register_k) {
        for (unsigned int index = linear_thread; index < register_tile * register_k;
             index += blockDim.x * blockDim.y) {
            const unsigned int local_row = index / register_k;
            const unsigned int local_inner = index % register_k;
            const std::size_t row = block_row + local_row;
            const std::size_t inner = tile_begin + local_inner;
            a_tile[local_row][local_inner] = row < m && inner < k ? a[row * k + inner] : 0.0F;
        }
        for (unsigned int index = linear_thread; index < register_k * register_tile;
             index += blockDim.x * blockDim.y) {
            const unsigned int local_inner = index / register_tile;
            const unsigned int local_column = index % register_tile;
            const std::size_t inner = tile_begin + local_inner;
            const std::size_t column = block_column + local_column;
            b_tile[local_inner][local_column] =
                inner < k && column < n ? b[inner * n + column] : 0.0F;
        }
        __syncthreads();

#pragma unroll
        for (unsigned int inner = 0; inner < register_k; ++inner) {
            float a_values[thread_tile];
            float b_values[thread_tile];
#pragma unroll
            for (unsigned int row = 0; row < thread_tile; ++row) {
                a_values[row] = a_tile[threadIdx.y * thread_tile + row][inner];
            }
#pragma unroll
            for (unsigned int column = 0; column < thread_tile; ++column) {
                b_values[column] = b_tile[inner][threadIdx.x * thread_tile + column];
            }
#pragma unroll
            for (unsigned int row = 0; row < thread_tile; ++row) {
#pragma unroll
                for (unsigned int column = 0; column < thread_tile; ++column) {
                    accumulators[row][column] =
                        fmaf(a_values[row], b_values[column], accumulators[row][column]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int row_offset = 0; row_offset < thread_tile; ++row_offset) {
        const std::size_t row = block_row + threadIdx.y * thread_tile + row_offset;
#pragma unroll
        for (unsigned int column_offset = 0; column_offset < thread_tile; ++column_offset) {
            const std::size_t column = block_column + threadIdx.x * thread_tile + column_offset;
            if (row < m && column < n) {
                c[row * n + column] = accumulators[row_offset][column_offset];
            }
        }
    }
}

__global__ void wmma_fp16_fp32_kernel(const __half* a, const __half* b, float* c, const int m,
                                      const int n, const int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const int tile_row = static_cast<int>(blockIdx.y) * 16;
    const int tile_column = static_cast<int>(blockIdx.x) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0F);
    for (int inner = 0; inner < k; inner += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_fragment;
        wmma::load_matrix_sync(a_fragment, a + tile_row * k + inner, k);
        wmma::load_matrix_sync(b_fragment, b + inner * n + tile_column, n);
        wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
    wmma::store_matrix_sync(c + tile_row * n + tile_column, accumulator, n, wmma::mem_row_major);
#else
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(c);
    static_cast<void>(m);
    static_cast<void>(n);
    static_cast<void>(k);
#endif
}

void launch_custom_fp32(const float* a, const float* b, float* c, const GemmProblem& problem,
                        const GemmVariant variant, const cudaStream_t stream) {
    const auto grid_16 = dim3{static_cast<unsigned int>((problem.n + tile_16 - 1U) / tile_16),
                              static_cast<unsigned int>((problem.m + tile_16 - 1U) / tile_16), 1U};
    switch (variant) {
    case GemmVariant::naive_fp32:
        naive_fp32_kernel<<<grid_16, dim3{tile_16, tile_16, 1U}, 0, stream>>>(a, b, c, problem.m,
                                                                              problem.n, problem.k);
        break;
    case GemmVariant::tiled_fp32:
        tiled_16_kernel<float><<<grid_16, dim3{tile_16, tile_16, 1U}, 0, stream>>>(
            a, b, c, problem.m, problem.n, problem.k);
        break;
    case GemmVariant::coalesced_fp32: {
        const dim3 grid{static_cast<unsigned int>((problem.n + tile_32 - 1U) / tile_32),
                        static_cast<unsigned int>((problem.m + tile_32 - 1U) / tile_32), 1U};
        coalesced_fp32_kernel<<<grid, dim3{tile_32, 8U, 1U}, 0, stream>>>(a, b, c, problem.m,
                                                                          problem.n, problem.k);
        break;
    }
    case GemmVariant::register_blocked_fp32: {
        const dim3 grid{static_cast<unsigned int>((problem.n + register_tile - 1U) / register_tile),
                        static_cast<unsigned int>((problem.m + register_tile - 1U) / register_tile),
                        1U};
        register_blocked_fp32_kernel<<<grid, dim3{16U, 16U, 1U}, 0, stream>>>(a, b, c, problem.m,
                                                                              problem.n, problem.k);
        break;
    }
    default:
        throw std::invalid_argument("selected GEMM variant does not accept FP32 input");
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_cublas_fp32(const float* a, const float* b, float* c, const GemmProblem& problem,
                        const cublasHandle_t handle, const cudaStream_t stream) {
    if (handle == nullptr) {
        throw std::invalid_argument("cuBLAS GEMM requires a valid handle");
    }
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(problem.n),
                             static_cast<int>(problem.m), static_cast<int>(problem.k), &alpha, b,
                             static_cast<int>(problem.n), a, static_cast<int>(problem.k), &beta, c,
                             static_cast<int>(problem.n)),
                 "cublasSgemm");
}

void launch_cublas_fp16(const __half* a, const __half* b, float* c, const GemmProblem& problem,
                        const cublasHandle_t handle, const cudaStream_t stream) {
    if (handle == nullptr) {
        throw std::invalid_argument("cuBLAS GEMM requires a valid handle");
    }
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(problem.n),
                              static_cast<int>(problem.m), static_cast<int>(problem.k), &alpha, b,
                              CUDA_R_16F, static_cast<int>(problem.n), a, CUDA_R_16F,
                              static_cast<int>(problem.k), &beta, c, CUDA_R_32F,
                              static_cast<int>(problem.n), CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "cublasGemmEx");
}

} // namespace

const char* gemm_variant_name(const GemmVariant variant) noexcept {
    switch (variant) {
    case GemmVariant::naive_fp32:
        return "naive_fp32";
    case GemmVariant::tiled_fp32:
        return "tiled_fp32";
    case GemmVariant::coalesced_fp32:
        return "coalesced_fp32";
    case GemmVariant::register_blocked_fp32:
        return "register_blocked_fp32";
    case GemmVariant::tiled_fp16_fp32:
        return "tiled_fp16_fp32";
    case GemmVariant::wmma_fp16_fp32:
        return "wmma_fp16_fp32";
    }
    return "unknown";
}

const char* gemm_backend_name(const GemmBackend backend) noexcept {
    switch (backend) {
    case GemmBackend::custom:
        return "custom";
    case GemmBackend::cublas:
        return "cublas";
    }
    return "unknown";
}

bool gemm_problem_is_empty(const GemmProblem& problem) noexcept {
    return problem.m == 0U || problem.n == 0U || problem.k == 0U;
}

bool gemm_variant_supports_problem(const GemmVariant variant, const GemmProblem& problem) noexcept {
    if (variant != GemmVariant::wmma_fp16_fp32) {
        return true;
    }
    return problem.m % 16U == 0U && problem.n % 16U == 0U && problem.k % 16U == 0U;
}

std::size_t gemm_a_elements(const GemmProblem& problem) {
    return checked_product(problem.m, problem.k, "A");
}

std::size_t gemm_b_elements(const GemmProblem& problem) {
    return checked_product(problem.k, problem.n, "B");
}

std::size_t gemm_c_elements(const GemmProblem& problem) {
    return checked_product(problem.m, problem.n, "C");
}

double gemm_flop_count(const GemmProblem& problem) noexcept {
    return 2.0 * static_cast<double>(problem.m) * static_cast<double>(problem.n) *
           static_cast<double>(problem.k);
}

Tolerance gemm_fp32_tolerance(const std::size_t inner_dimension) noexcept {
    const double scale = std::max(
        1.0, std::ceil(std::log2(static_cast<double>(std::max<std::size_t>(1U, inner_dimension)))));
    return Tolerance{1.0e-5 * scale, 1.0e-4};
}

Tolerance gemm_fp16_tolerance(const std::size_t inner_dimension) noexcept {
    const double scale = std::max(
        1.0, std::ceil(std::log2(static_cast<double>(std::max<std::size_t>(1U, inner_dimension)))));
    return Tolerance{1.0e-3 * scale, 1.0e-2};
}

void gemm_cpu_fp32(const float* a, const float* b, float* c, const GemmProblem& problem) {
    validate_pointers(a, b, c, problem);
    if (gemm_problem_is_empty(problem)) {
        return;
    }
    for (std::size_t row = 0; row < problem.m; ++row) {
        for (std::size_t column = 0; column < problem.n; ++column) {
            double accumulator = 0.0;
            for (std::size_t inner = 0; inner < problem.k; ++inner) {
                accumulator += static_cast<double>(a[row * problem.k + inner]) *
                               static_cast<double>(b[inner * problem.n + column]);
            }
            c[row * problem.n + column] = static_cast<float>(accumulator);
        }
    }
}

void gemm_fp32_cuda(const float* a, const float* b, float* c, const GemmProblem& problem,
                    const GemmDispatch dispatch, const cublasHandle_t cublas_handle,
                    const cudaStream_t stream) {
    validate_pointers(a, b, c, problem);
    if (gemm_problem_is_empty(problem)) {
        return;
    }
    if (dispatch.backend == GemmBackend::custom) {
        launch_custom_fp32(a, b, c, problem, dispatch.variant, stream);
        return;
    }
    launch_cublas_fp32(a, b, c, problem, cublas_handle, stream);
}

void gemm_fp16_cuda(const __half* a, const __half* b, float* c, const GemmProblem& problem,
                    const GemmDispatch dispatch, const cublasHandle_t cublas_handle,
                    const cudaStream_t stream) {
    validate_pointers(a, b, c, problem);
    if (gemm_problem_is_empty(problem)) {
        return;
    }
    if (dispatch.backend == GemmBackend::cublas) {
        launch_cublas_fp16(a, b, c, problem, cublas_handle, stream);
        return;
    }
    if (dispatch.variant == GemmVariant::tiled_fp16_fp32) {
        const dim3 grid{static_cast<unsigned int>((problem.n + tile_16 - 1U) / tile_16),
                        static_cast<unsigned int>((problem.m + tile_16 - 1U) / tile_16), 1U};
        tiled_16_kernel<__half><<<grid, dim3{tile_16, tile_16, 1U}, 0, stream>>>(
            a, b, c, problem.m, problem.n, problem.k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (dispatch.variant == GemmVariant::wmma_fp16_fp32) {
        if (!gemm_variant_supports_problem(dispatch.variant, problem)) {
            throw std::invalid_argument("WMMA GEMM requires M, N, and K to be multiples of 16");
        }
        const dim3 grid{static_cast<unsigned int>(problem.n / 16U),
                        static_cast<unsigned int>(problem.m / 16U), 1U};
        wmma_fp16_fp32_kernel<<<grid, dim3{32U, 1U, 1U}, 0, stream>>>(
            a, b, c, static_cast<int>(problem.m), static_cast<int>(problem.n),
            static_cast<int>(problem.k));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    throw std::invalid_argument("selected GEMM variant does not accept FP16 input");
}

void gemm_cuda(const TensorView& a, const TensorView& b, TensorView& c, const GemmProblem& problem,
               const GemmDispatch dispatch, const cublasHandle_t cublas_handle,
               const cudaStream_t stream) {
    const std::size_t expected_a = gemm_a_elements(problem);
    const std::size_t expected_b = gemm_b_elements(problem);
    const std::size_t expected_c = gemm_c_elements(problem);
    if (a.element_count() != expected_a || b.element_count() != expected_b ||
        c.element_count() != expected_c) {
        throw std::invalid_argument("GEMM TensorView element counts do not match the problem");
    }
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("GEMM input TensorView dtypes must match");
    }
    if (c.dtype() != DType::fp32) {
        throw std::invalid_argument("GEMM output TensorView must be fp32");
    }

    if (a.dtype() == DType::fp32) {
        gemm_fp32_cuda(a.data_as<float>(), b.data_as<float>(), c.data_as<float>(), problem,
                       dispatch, cublas_handle, stream);
        return;
    }
    if (a.dtype() == DType::fp16) {
        gemm_fp16_cuda(a.data_as<__half>(), b.data_as<__half>(), c.data_as<float>(), problem,
                       dispatch, cublas_handle, stream);
        return;
    }
    throw std::invalid_argument("unsupported GEMM TensorView dtype");
}

} // namespace warpforge
