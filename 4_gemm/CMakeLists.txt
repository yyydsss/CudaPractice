find_package(CUDAToolkit REQUIRED)

add_executable(gemm_0_global gemm_0_global.cu)
target_link_libraries(gemm_0_global PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(gemm_0_global PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(gemm_0_global PRIVATE -lineinfo)

