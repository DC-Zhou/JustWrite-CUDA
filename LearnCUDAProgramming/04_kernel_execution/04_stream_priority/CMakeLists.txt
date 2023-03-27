cmake_minimum_required(VERSION 3.24)

project(stream_priority LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

set(CUDA_SAMPLE_COMMON_DIR D:/cuda-samples-11.6/Common)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_SAMPLE_COMMON_DIR})

add_executable(stream_priority prioritized_cuda_stream.cu)

# use the following line to enable C++ cuda compilation
#set_target_properties(cuda_default_stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

