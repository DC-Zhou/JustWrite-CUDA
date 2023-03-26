cmake_minimum_required(VERSION 3.24)

project(warp_synchronous_programming LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

add_executable(warp_synchronous_programming reduction.cpp reduction.h reduction_wp_kernel.cu)

set_target_properties(warp_synchronous_programming PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)
include_directories(${CUDA_INCLUDE_DIRS})