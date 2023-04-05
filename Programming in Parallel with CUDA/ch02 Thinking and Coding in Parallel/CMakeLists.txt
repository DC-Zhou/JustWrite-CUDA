cmake_minimum_required(VERSION 3.24)

project(ch02 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

set(CUDA_SAMPLE_COMMON_DIR D:/cuda-samples-11.6/Common)

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${CUDA_SAMPLE_COMMON_DIR})

add_executable(grid3D grid3D.cu)
add_executable(grid3D_linear grid3D_linear.cu)

add_executable(reduce0 reduce0.cu common.h)
add_executable(reduce1 reduce1.cu common.h)
add_executable(reduce2 reduce2.cu common.h)
add_executable(reduce3 reduce3.cu common.h)

# use the following line to enable host device mix compilation
#set_target_properties(cuda_sum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

