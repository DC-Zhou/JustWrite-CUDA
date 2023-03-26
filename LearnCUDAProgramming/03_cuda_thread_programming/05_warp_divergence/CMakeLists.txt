cmake_minimum_required(VERSION 3.24)

project(warp_divergence LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

add_executable(warp_divergence_interleaving shared_select.cpp reduction.h reduction_kernel_interleaving.cu)
add_executable(warp_divergence_sequential shared_select.cpp reduction.h reduction_kernel_sequential.cu)

set_target_properties(warp_divergence_interleaving PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(warp_divergence_sequential PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)
include_directories(${CUDA_INCLUDE_DIRS})