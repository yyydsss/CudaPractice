find_package(CUDAToolkit REQUIRED)

add_executable(elementwise elementwise.cu)
target_link_libraries(elementwise PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
