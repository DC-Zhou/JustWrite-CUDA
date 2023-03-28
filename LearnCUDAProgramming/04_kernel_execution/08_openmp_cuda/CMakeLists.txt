cmake_minimum_required(VERSION 3.24)

project(openmp_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)



enable_language(CUDA)

set(CUDA_SAMPLE_COMMON_DIR D:/cuda-samples-11.6/Common)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_SAMPLE_COMMON_DIR})

add_executable(openmp_check main.cpp)
add_executable(openmp_cuda openmp.cu)
add_executable(openmp_default_stream openmp_default_stream.cu)
add_executable(openmp_gpus openmp_gpus.cu)

if(OpenMP_CXX_FOUND)
    target_link_libraries(openmp_check PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(openmp_cuda PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(openmp_default_stream PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(openmp_gpus PUBLIC OpenMP::OpenMP_CXX)
endif()

# use the following line to enable C++ cuda mix compilation
#set_target_properties(cuda_default_stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

