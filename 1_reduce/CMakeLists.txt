find_package(CUDAToolkit REQUIRED)

add_executable(reduce_global reduce_global.cu)
target_link_libraries(reduce_global PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_global PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_0 reduce_0.cu ${CMAKE_SOURCE_DIR}/include/utils.cpp)
target_include_directories(reduce_0 PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(reduce_1 reduce_1.cu)
target_link_libraries(reduce_0 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

target_link_libraries(reduce_1 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

add_executable(reduce_naive reduce_naive.cu)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_0 PRIVATE $<$<COMPILE_LANGUAGE::CUDA>:-G>)
endif()