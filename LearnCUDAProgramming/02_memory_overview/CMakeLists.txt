cmake_minimum_required(VERSION 3.24)

project(chapter02 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

enable_language(CUDA)

add_executable(sgemm sgemm.cu)
add_executable(vec_add vector_addition.cu)
add_executable(aos aos.cu)
add_executable(soa soa.cu)
add_executable(matrix_transpose matrix_transpose.cu)
add_executable(unified_memory unified_memory.cu)
add_executable(unified_memory_initialized unified_memory_initialized.cu)
add_executable(unified_memory_64align unified_memory_64align.cu)
add_executable(unified_memory_prefetch unified_memory_prefetch.cu)
# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)