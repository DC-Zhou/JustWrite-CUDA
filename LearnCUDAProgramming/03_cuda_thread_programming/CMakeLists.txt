cmake_minimum_required(VERSION 3.24)

project(chapter03 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

enable_language(CUDA)

add_executable(cuda_thread_block cuda_thread_block.cu)
add_executable(sgemm sgemm.cu)

add_subdirectory(threadsync_and_reduction)

# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)