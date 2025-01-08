find_package(CUDAToolkit REQUIRED)
add_executable(reduce_0 reduce_0.cu ${CMAKE_SOURCE_DIR}/include/utils.cpp)
target_include_directories(reduce_0 PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(reduce_1 reduce_1.cu)
target_link_libraries(reduce_0 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

