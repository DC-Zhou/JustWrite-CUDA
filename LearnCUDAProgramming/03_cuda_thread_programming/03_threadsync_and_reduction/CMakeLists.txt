cmake_minimum_required(VERSION 3.24)

project(threadsync_and_reduction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

add_executable(global reduction_global.cpp reduction.h reduction_global_kernel.cu)
add_executable(shared reduction_shared.cpp reduction.h reduction_shared_kernel.cu)

set_target_properties(global PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)
include_directories(${CUDA_INCLUDE_DIRS})