cmake_minimum_required(VERSION 3.24)

project(cuda_pipelining LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(cuda_pipelining cuda_pipelining.cu)

# use the following line to enable C++ cuda compilation
#set_target_properties(cuda_default_stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# include the cuda-samples-11.6/Common directory
include_directories(D:/cuda-samples-11.6/Common)
