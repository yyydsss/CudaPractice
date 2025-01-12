find_package(CUDAToolkit REQUIRED)

add_executable(gemm_0 gemm_0.cu)
target_link_libraries(gemm_0 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(gemm_0 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(gemm_0 PRIVATE -lineinfo)

