cmake_minimum_required(VERSION 3.24)

project(atomic_operation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(atomic_operation reduction.cpp reduction.h reduction_kernel.cu)
add_executable(atomic_operation_blk_atmc reduction.cpp reduction.h reduction_blk_atmc_kernel.cu)
add_executable(atomic_operation_wrp_atmc reduction.cpp reduction.h reduction_wrp_atmc_kernel.cu)

set_target_properties(atomic_operation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(atomic_operation_blk_atmc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(atomic_operation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)
