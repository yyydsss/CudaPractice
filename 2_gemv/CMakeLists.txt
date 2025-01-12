find_package(CUDAToolkit REQUIRED)

add_executable(gemv1 gemv1.cu)
target_link_libraries(gemv1PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
