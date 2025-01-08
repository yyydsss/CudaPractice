cmake_minimum_required(VERSION 3.5.0)
project(yangds VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)  # 根据你的 GPU 架构设置
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(1_reduce)
add_subdirectory(include)
find_package(CUDAToolkit REQUIRED)

# include_directories("/home/yds/include")
include_directories(${CMAKE_SOURCE_DIR}/include)
# add_executable(yangds main.cpp)

