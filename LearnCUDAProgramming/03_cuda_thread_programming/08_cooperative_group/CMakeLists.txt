cmake_minimum_required(VERSION 3.24)

project(cooperative_group LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(cooperative_group reduction.cpp reduction.h reduction_cg_kernel.cu)
add_executable(cooperative_group_shift reduction.cpp reduction.h reduction_cg_shift_kernel.cu)

set_target_properties(cooperative_group PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cooperative_group_shift PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)
