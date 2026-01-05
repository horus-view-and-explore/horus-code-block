cmake_minimum_required(VERSION 3.18.0)

IF (WIN32)
  set(TENSORRT_ROOT "C:/Program Files/NVIDIA/TensorRT/10.7.0" CACHE PATH "Path to TensorRT installation")
ELSE()
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-9)
  set(TENSORRT_ROOT "/usr/local/cuda/TensorRT" CACHE PATH "Path to TensorRT installation")
ENDIF()

project(HORUS_CUDA_CODE_BLOCK_TENSOR_RT_CONV VERSION 1.0.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)



# Include directories
include_directories(
    ${TENSORRT_ROOT}/include
)

# Find TensorRT libraries
find_library(NVINFER_LIB nvinfer nvinfer_10 PATHS ${TENSORRT_ROOT}/lib)
find_library(NVONNXPARSER_LIB nvonnxparser nvonnxparser_10 PATHS ${TENSORRT_ROOT}/lib)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin nvinfer_plugin_10 PATHS ${TENSORRT_ROOT}/lib)


add_library(cuda_tensor_rt_scale SHARED
  ../../../Horus_code_block.h
  ../../../Horus_cuda_code_block.h
  ../Helper_functions.hpp
  ../Helper_functions.cpp
  ../Horus_cuda.hpp
  ../Horus_cuda.cpp
  cuda_tensor_rt_scale.cpp
  Kernel.cu
  Network.hpp
  )


target_link_libraries(cuda_tensor_rt_scale CUDA::cudart)
target_link_libraries(cuda_tensor_rt_scale CUDA::cuda_driver)
target_link_libraries(cuda_tensor_rt_scale ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${NVINFER_PLUGIN_LIB})

