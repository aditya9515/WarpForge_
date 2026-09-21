cmake_minimum_required(VERSION 3.24)

project(
    WarpForge
    VERSION 0.1.0
    DESCRIPTION "CUDA kernel engineering and compact inference runtime"
    LANGUAGES CXX CUDA
)

include(CTest)
find_package(CUDAToolkit 12.6 REQUIRED)

set(WARPFORGE_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --always --dirty
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE WARPFORGE_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build for")
endif()

function(warpforge_enable_warnings target)
    if(MSVC)
        target_compile_options(
            ${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    else()
        target_compile_options(
            ${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
        )
    endif()
endfunction()

add_library(
    warpforge
    STATIC
        src/benchmark.cpp
        src/benchmark_cuda.cu
        src/kernels/gemm/gemm.cu
        src/kernels/memory/memory.cu
        src/kernels/reduction/reduction.cu
        src/kernels/transformer/causal_mask.cu
        src/kernels/transformer/elementwise.cu
        src/kernels/transformer/rmsnorm.cu
        src/kernels/transformer/rope.cu
        src/kernels/transformer/softmax.cu
        src/kernels/vector/vector_add.cu
        src/runtime/device.cu
        src/validation.cpp
)
add_library(WarpForge::warpforge ALIAS warpforge)

target_compile_features(warpforge PUBLIC cxx_std_17 cuda_std_17)
target_include_directories(
    warpforge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(warpforge PUBLIC CUDA::cudart CUDA::cublas)
target_compile_definitions(
    warpforge
    PRIVATE
        WARPFORGE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        WARPFORGE_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        WARPFORGE_GIT_COMMIT="${WARPFORGE_GIT_COMMIT}"
)
warpforge_enable_warnings(warpforge)

add_executable(warpforge_device_info apps/device_info/main.cpp)
set_target_properties(warpforge_device_info PROPERTIES OUTPUT_NAME "warpforge-device-info")
target_link_libraries(warpforge_device_info PRIVATE WarpForge::warpforge)
warpforge_enable_warnings(warpforge_device_info)

add_executable(warpforge_benchmark_vector_add apps/benchmarks/vector_add.cpp)
set_target_properties(
    warpforge_benchmark_vector_add
    PROPERTIES OUTPUT_NAME "warpforge-benchmark-vector-add"
)
target_link_libraries(warpforge_benchmark_vector_add PRIVATE WarpForge::warpforge)
warpforge_enable_warnings(warpforge_benchmark_vector_add)

add_executable(warpforge_benchmark_memory apps/benchmarks/memory.cpp)
set_target_properties(
    warpforge_benchmark_memory
    PROPERTIES OUTPUT_NAME "warpforge-benchmark-memory"
)
target_link_libraries(warpforge_benchmark_memory PRIVATE WarpForge::warpforge)
warpforge_enable_warnings(warpforge_benchmark_memory)

add_executable(warpforge_benchmark_reduction apps/benchmarks/reduction.cpp)
set_target_properties(
    warpforge_benchmark_reduction
    PROPERTIES OUTPUT_NAME "warpforge-benchmark-reduction"
)
target_link_libraries(warpforge_benchmark_reduction PRIVATE WarpForge::warpforge)
warpforge_enable_warnings(warpforge_benchmark_reduction)

add_executable(warpforge_benchmark_gemm apps/benchmarks/gemm.cu)
set_target_properties(
    warpforge_benchmark_gemm
    PROPERTIES OUTPUT_NAME "warpforge-benchmark-gemm"
)
target_link_libraries(warpforge_benchmark_gemm PRIVATE WarpForge::warpforge)
warpforge_enable_warnings(warpforge_benchmark_gemm)

add_executable(warpforge_benchmark_transformer apps/benchmarks/transformer.cpp)
set_target_properties(
    warpforge_benchmark_transformer
    PROPERTIES OUTPUT_NAME "warpforge-benchmark-transformer"
)
target_link_libraries(warpforge_benchmark_transformer PRIVATE WarpForge::warpforge)
warpforge_enable_warnings(warpforge_benchmark_transformer)

if(BUILD_TESTING)
    find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)

    add_executable(warpforge_validation_test tests/cpu/validation_test.cpp)
    target_link_libraries(warpforge_validation_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_validation_test)

    add_executable(warpforge_benchmark_statistics_test tests/cpu/benchmark_statistics_test.cpp)
    target_link_libraries(warpforge_benchmark_statistics_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_benchmark_statistics_test)

    add_executable(warpforge_cuda_smoke tests/cuda/runtime_smoke_test.cpp)
    set_target_properties(warpforge_cuda_smoke PROPERTIES OUTPUT_NAME "warpforge-cuda-smoke")
    target_link_libraries(warpforge_cuda_smoke PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_cuda_smoke)

    add_executable(warpforge_vector_add_test tests/cuda/vector_add_test.cpp)
    target_link_libraries(warpforge_vector_add_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_vector_add_test)

    add_executable(warpforge_memory_test tests/cuda/memory_test.cpp)
    target_link_libraries(warpforge_memory_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_memory_test)

    add_executable(warpforge_reduction_test tests/cuda/reduction_test.cpp)
    target_link_libraries(warpforge_reduction_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_reduction_test)

    add_executable(warpforge_gemm_test tests/cuda/gemm_test.cu)
    target_link_libraries(warpforge_gemm_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_gemm_test)

    add_executable(warpforge_softmax_test tests/cuda/softmax_test.cpp)
    target_link_libraries(warpforge_softmax_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_softmax_test)

    add_executable(warpforge_rmsnorm_test tests/cuda/rmsnorm_test.cpp)
    target_link_libraries(warpforge_rmsnorm_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_rmsnorm_test)

    add_executable(warpforge_rope_test tests/cuda/rope_test.cpp)
    target_link_libraries(warpforge_rope_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_rope_test)

    add_executable(warpforge_elementwise_test tests/cuda/elementwise_test.cpp)
    target_link_libraries(warpforge_elementwise_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_elementwise_test)

    add_executable(warpforge_causal_mask_test tests/cuda/causal_mask_test.cpp)
    target_link_libraries(warpforge_causal_mask_test PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_causal_mask_test)

    add_test(NAME cpu.validation COMMAND warpforge_validation_test)
    add_test(NAME cpu.benchmark_statistics COMMAND warpforge_benchmark_statistics_test)
    add_test(NAME cuda.runtime_smoke COMMAND warpforge_cuda_smoke)
    add_test(NAME cuda.vector_add COMMAND warpforge_vector_add_test)
    add_test(NAME cuda.memory COMMAND warpforge_memory_test)
    add_test(NAME cuda.reduction COMMAND warpforge_reduction_test)
    add_test(NAME cuda.gemm COMMAND warpforge_gemm_test)
    add_test(NAME cuda.softmax COMMAND warpforge_softmax_test)
    add_test(NAME cuda.rmsnorm COMMAND warpforge_rmsnorm_test)
    add_test(NAME cuda.rope COMMAND warpforge_rope_test)
    add_test(NAME cuda.elementwise COMMAND warpforge_elementwise_test)
    add_test(NAME cuda.causal_mask COMMAND warpforge_causal_mask_test)
    add_test(
        NAME benchmark.vector_add
        COMMAND
            warpforge_benchmark_vector_add
            --size 4097
            --warmups 2
            --iterations 5
            --output ${CMAKE_CURRENT_BINARY_DIR}/test-results/vector_add.json
    )
    add_test(
        NAME benchmark.json_valid
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/validate_benchmark_json.py
            ${CMAKE_CURRENT_BINARY_DIR}/test-results/vector_add.json
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/schema/v1.json
    )
    add_test(
        NAME benchmark.memory
        COMMAND
            warpforge_benchmark_memory
            --size 4097
            --rows 37
            --columns 53
            --warmups 1
            --iterations 3
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/test-results/memory
    )
    add_test(
        NAME benchmark.memory_json_valid
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/validate_memory_results.py
            ${CMAKE_CURRENT_BINARY_DIR}/test-results/memory
    )
    add_test(
        NAME benchmark.reduction
        COMMAND
            warpforge_benchmark_reduction
            --size 65537
            --warmups 1
            --iterations 3
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/test-results/reduction
    )
    add_test(
        NAME benchmark.reduction_json_valid
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/validate_reduction_results.py
            ${CMAKE_CURRENT_BINARY_DIR}/test-results/reduction
    )
    add_test(
        NAME benchmark.gemm
        COMMAND
            warpforge_benchmark_gemm
            --size 64
            --warmups 1
            --iterations 3
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/test-results/gemm
    )
    add_test(
        NAME benchmark.gemm_json_valid
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/validate_gemm_results.py
            ${CMAKE_CURRENT_BINARY_DIR}/test-results/gemm
    )
    add_test(
        NAME benchmark.transformer
        COMMAND
            warpforge_benchmark_transformer
            --rows 37
            --columns 53
            --elements 4097
            --sequence 17
            --heads 2
            --head-dim 16
            --mask-length 17
            --warmups 1
            --iterations 3
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/test-results/transformer
    )
    add_test(
        NAME benchmark.transformer_json_valid
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/validate_transformer_results.py
            ${CMAKE_CURRENT_BINARY_DIR}/test-results/transformer
    )

    set_tests_properties(cpu.validation cpu.benchmark_statistics PROPERTIES LABELS "cpu;unit")
    set_tests_properties(cuda.runtime_smoke PROPERTIES LABELS "cuda;smoke")
    set_tests_properties(
        cuda.vector_add cuda.memory cuda.reduction cuda.gemm cuda.softmax cuda.rmsnorm
        cuda.rope
        cuda.elementwise cuda.causal_mask
        PROPERTIES LABELS "cuda;correctness"
    )
    set_tests_properties(
        benchmark.vector_add benchmark.memory benchmark.reduction benchmark.gemm
        benchmark.transformer
        PROPERTIES LABELS "cuda;benchmark"
    )
    set_tests_properties(
        benchmark.json_valid
        PROPERTIES
            DEPENDS benchmark.vector_add
            LABELS "benchmark;schema"
    )
    set_tests_properties(
        benchmark.memory_json_valid
        PROPERTIES
            DEPENDS benchmark.memory
            LABELS "benchmark;schema"
    )
    set_tests_properties(
        benchmark.reduction_json_valid
        PROPERTIES
            DEPENDS benchmark.reduction
            LABELS "benchmark;schema"
    )
    set_tests_properties(
        benchmark.gemm_json_valid
        PROPERTIES
            DEPENDS benchmark.gemm
            LABELS "benchmark;schema"
    )
    set_tests_properties(
        benchmark.transformer_json_valid
        PROPERTIES
            DEPENDS benchmark.transformer
            LABELS "benchmark;schema"
    )
endif()
