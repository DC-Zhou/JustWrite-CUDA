cmake_minimum_required(VERSION 3.24)

project(grid_level_cg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

set(CUDA_SAMPLE_COMMON_DIR D:/cuda-samples-11.6/Common)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_SAMPLE_COMMON_DIR})

add_executable(grid_level_cg reduction.cpp reduction.h reduction_kernel.cu)

# use the following line to enable C++ cuda compilation
set_target_properties(grid_level_cg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

