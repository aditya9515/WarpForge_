cmake_minimum_required(VERSION 3.24)

project(
    WarpForge
    VERSION 0.1.0
    DESCRIPTION "CUDA kernel engineering and compact inference runtime"
    LANGUAGES CXX CUDA
)

include(CTest)
find_package(CUDAToolkit 12.6 REQUIRED)

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

add_library(warpforge STATIC src/runtime/device.cu)
add_library(WarpForge::warpforge ALIAS warpforge)

target_compile_features(warpforge PUBLIC cxx_std_17 cuda_std_17)
target_include_directories(
    warpforge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(warpforge PUBLIC CUDA::cudart)
warpforge_enable_warnings(warpforge)

add_executable(warpforge_device_info apps/device_info/main.cpp)
set_target_properties(warpforge_device_info PROPERTIES OUTPUT_NAME "warpforge-device-info")
target_link_libraries(warpforge_device_info PRIVATE WarpForge::warpforge)
warpforge_enable_warnings(warpforge_device_info)

if(BUILD_TESTING)
    add_executable(warpforge_cuda_smoke tests/cuda/runtime_smoke_test.cpp)
    set_target_properties(warpforge_cuda_smoke PROPERTIES OUTPUT_NAME "warpforge-cuda-smoke")
    target_link_libraries(warpforge_cuda_smoke PRIVATE WarpForge::warpforge)
    warpforge_enable_warnings(warpforge_cuda_smoke)

    add_test(NAME cuda.runtime_smoke COMMAND warpforge_cuda_smoke)
    set_tests_properties(cuda.runtime_smoke PROPERTIES LABELS "cuda;smoke")
endif()
