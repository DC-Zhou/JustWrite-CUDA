cmake_minimum_required(VERSION 3.24)

project(chapter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

enable_language(CUDA)

add_executable(hello hello.cu)
add_executable(vector_add vector_addition.cu)