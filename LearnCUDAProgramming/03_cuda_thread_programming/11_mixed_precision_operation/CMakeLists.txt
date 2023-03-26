cmake_minimum_required(VERSION 3.24)

project(mixed_precision LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(mixed_precision mixed_precision.cu util.cuh)
add_executable(mixed_precision_int mixed_precision_int.cu util.cuh)
add_executable(mixed_precision_half mixed_precision_half.cu util.cuh)

set_target_properties(mixed_precision PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mixed_precision_int PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mixed_precision_half PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)
