cmake_minimum_required(VERSION 3.24)

project(scan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

enable_language(CUDA)

set(CUDA_SAMPLE_COMMON_DIR D:/cuda-samples-11.6/Common)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_SAMPLE_COMMON_DIR})

message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")

add_executable(${PROJECT_NAME} scan.cu scan.h scan_v1.cu scan_v2.cu utils.h)

# use the following line to enable host device mix compilation
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

