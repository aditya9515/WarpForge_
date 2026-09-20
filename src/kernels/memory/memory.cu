#include <warpforge/cuda_check.cuh>
#include <warpforge/memory.cuh>

#include <limits>
#include <stdexcept>
#include <string>

namespace warpforge {
namespace {

__global__ void saxpy_kernel(
    const float alpha,
    const float* input,
    float* inout,
    const std::size_t element_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        inout[index] = alpha * input[index] + inout[index];
    }
}

__global__ void memory_copy_kernel(
    const float* input,
    float* output,
    const std::size_t element_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = input[index];
    }
}

__global__ void strided_copy_kernel(
    const float* input,
    float* output,
    const std::size_t element_count,
    const std::size_t stride) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = input[index * stride];
    }
}

__global__ void transpose_naive_kernel(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns) {
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < rows && column < columns) {
        output[column * rows + row] = input[row * columns + column];
    }
}

__global__ void transpose_tiled_kernel(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns) {
    __shared__ float tile[transpose_tile_dimension][transpose_tile_dimension + 1U];

    const std::size_t input_column =
        static_cast<std::size_t>(blockIdx.x) * transpose_tile_dimension + threadIdx.x;
    const std::size_t input_row =
        static_cast<std::size_t>(blockIdx.y) * transpose_tile_dimension + threadIdx.y;

    for (unsigned int offset = 0; offset < transpose_tile_dimension;
         offset += transpose_block_rows) {
        if (input_column < columns && input_row + offset < rows) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[(input_row + offset) * columns + input_column];
        }
    }
    __syncthreads();

    const std::size_t output_column =
        static_cast<std::size_t>(blockIdx.y) * transpose_tile_dimension + threadIdx.x;
    const std::size_t output_row =
        static_cast<std::size_t>(blockIdx.x) * transpose_tile_dimension + threadIdx.y;

    for (unsigned int offset = 0; offset < transpose_tile_dimension;
         offset += transpose_block_rows) {
        if (output_column < rows && output_row + offset < columns) {
            output[(output_row + offset) * rows + output_column] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

void require_non_null(
    const float* input,
    const float* output,
    const std::size_t element_count,
    const char* operation) {
    if (element_count > 0 && (input == nullptr || output == nullptr)) {
        throw std::invalid_argument(std::string(operation) +
                                    " pointers must not be null for non-empty input");
    }
}

std::size_t checked_matrix_elements(const std::size_t rows, const std::size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::overflow_error("matrix element count exceeds size_t");
    }
    return rows * columns;
}

}  // namespace

unsigned int memory_grid_size(
    const std::size_t element_count,
    const unsigned int block_size) {
    if (block_size == 0) {
        throw std::invalid_argument("block size must be greater than zero");
    }
    if (block_size > 1024U) {
        throw std::invalid_argument("block size exceeds the CUDA per-block limit");
    }
    if (element_count == 0) {
        return 0;
    }

    const std::size_t blocks = 1U + (element_count - 1U) / block_size;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("grid size exceeds the CUDA x-dimension range");
    }
    return static_cast<unsigned int>(blocks);
}

void saxpy_cpu(
    const float alpha,
    const float* input,
    float* inout,
    const std::size_t element_count) {
    require_non_null(input, inout, element_count, "SAXPY");
    for (std::size_t index = 0; index < element_count; ++index) {
        inout[index] = alpha * input[index] + inout[index];
    }
}

void saxpy_cuda(
    const float alpha,
    const float* input,
    float* inout,
    const std::size_t element_count,
    const unsigned int block_size,
    cudaStream_t stream) {
    require_non_null(input, inout, element_count, "SAXPY");
    const unsigned int grid_size = memory_grid_size(element_count, block_size);
    if (grid_size == 0) {
        return;
    }
    saxpy_kernel<<<grid_size, block_size, 0, stream>>>(alpha, input, inout, element_count);
    CUDA_CHECK(cudaGetLastError());
}

void memory_copy_cuda(
    const float* input,
    float* output,
    const std::size_t element_count,
    const unsigned int block_size,
    cudaStream_t stream) {
    require_non_null(input, output, element_count, "memory copy");
    const unsigned int grid_size = memory_grid_size(element_count, block_size);
    if (grid_size == 0) {
        return;
    }
    memory_copy_kernel<<<grid_size, block_size, 0, stream>>>(input, output, element_count);
    CUDA_CHECK(cudaGetLastError());
}

void strided_copy_cpu(
    const float* input,
    float* output,
    const std::size_t element_count,
    const std::size_t stride) {
    if (stride == 0) {
        throw std::invalid_argument("stride must be greater than zero");
    }
    require_non_null(input, output, element_count, "strided copy");
    if (element_count > 0 && element_count - 1U >
            std::numeric_limits<std::size_t>::max() / stride) {
        throw std::overflow_error("strided input index exceeds size_t");
    }
    for (std::size_t index = 0; index < element_count; ++index) {
        output[index] = input[index * stride];
    }
}

void strided_copy_cuda(
    const float* input,
    float* output,
    const std::size_t element_count,
    const std::size_t stride,
    const unsigned int block_size,
    cudaStream_t stream) {
    if (stride == 0) {
        throw std::invalid_argument("stride must be greater than zero");
    }
    require_non_null(input, output, element_count, "strided copy");
    if (element_count > 0 && element_count - 1U >
            std::numeric_limits<std::size_t>::max() / stride) {
        throw std::overflow_error("strided input index exceeds size_t");
    }
    const unsigned int grid_size = memory_grid_size(element_count, block_size);
    if (grid_size == 0) {
        return;
    }
    strided_copy_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, element_count, stride);
    CUDA_CHECK(cudaGetLastError());
}

void transpose_cpu(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns) {
    const std::size_t elements = checked_matrix_elements(rows, columns);
    require_non_null(input, output, elements, "transpose");
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            output[column * rows + row] = input[row * columns + column];
        }
    }
}

dim3 transpose_grid_size(
    const std::size_t rows,
    const std::size_t columns,
    const dim3 block) {
    if (block.x == 0 || block.y == 0 || block.z != 1U) {
        throw std::invalid_argument("transpose block dimensions must be non-zero with z equal to one");
    }
    if (static_cast<std::size_t>(block.x) * block.y > 1024U) {
        throw std::invalid_argument("transpose block exceeds the CUDA per-block limit");
    }
    if (rows == 0 || columns == 0) {
        return dim3{0U, 0U, 1U};
    }

    const std::size_t grid_x = 1U + (columns - 1U) / block.x;
    const std::size_t grid_y = 1U + (rows - 1U) / block.y;
    if (grid_x > std::numeric_limits<unsigned int>::max() ||
        grid_y > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("transpose grid exceeds CUDA dimension range");
    }
    return dim3{static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), 1U};
}

void transpose_naive_cuda(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    const dim3 block,
    cudaStream_t stream) {
    const std::size_t elements = checked_matrix_elements(rows, columns);
    require_non_null(input, output, elements, "transpose");
    const dim3 grid = transpose_grid_size(rows, columns, block);
    if (grid.x == 0 || grid.y == 0) {
        return;
    }
    transpose_naive_kernel<<<grid, block, 0, stream>>>(input, output, rows, columns);
    CUDA_CHECK(cudaGetLastError());
}

void transpose_tiled_cuda(
    const float* input,
    float* output,
    const std::size_t rows,
    const std::size_t columns,
    cudaStream_t stream) {
    const std::size_t elements = checked_matrix_elements(rows, columns);
    require_non_null(input, output, elements, "transpose");
    if (elements == 0) {
        return;
    }
    const dim3 block{transpose_tile_dimension, transpose_block_rows, 1U};
    const dim3 grid = transpose_grid_size(
        rows,
        columns,
        dim3{transpose_tile_dimension, transpose_tile_dimension, 1U});
    transpose_tiled_kernel<<<grid, block, 0, stream>>>(input, output, rows, columns);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace warpforge
