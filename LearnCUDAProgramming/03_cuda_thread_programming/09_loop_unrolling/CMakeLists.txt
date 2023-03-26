cmake_minimum_required(VERSION 3.24)

project(loop_unrolling LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(loop_unrolling_cg reduction.cpp reduction.h reduction_cg_kernel.cu)
add_executable(loop_unrolling_wp reduction.cpp reduction.h reduction_wp_kernel.cu)

set_target_properties(loop_unrolling_cg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(loop_unrolling_wp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)
