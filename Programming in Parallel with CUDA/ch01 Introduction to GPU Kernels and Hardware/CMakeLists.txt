cmake_minimum_required(VERSION 3.24)

project(ch01 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

enable_language(CUDA)

set(CUDA_SAMPLE_COMMON_DIR D:/cuda-samples-11.6/Common)

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${CUDA_SAMPLE_COMMON_DIR})

#cpu
add_executable(cpu_sum cpu_sum.cpp)

#openmp
add_executable(openmp_sum openmp_sum.cpp)
target_link_libraries(openmp_sum PUBLIC OpenMP::OpenMP_CXX)

#cuda
add_executable(cuda_sum cuda_sum.cu)

# use the following line to enable host device mix compilation
#set_target_properties(cuda_sum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

