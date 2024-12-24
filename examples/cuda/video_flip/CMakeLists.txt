cmake_minimum_required(VERSION 3.18.0)

project(HORUS_CUDA_CODE_BLOCK_VIDEO_FLIP VERSION 1.0.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_video_flip SHARED
  ../../../Horus_code_block.h
  ../../../Horus_cuda_code_block.h
  ../Helper_functions.hpp
  ../Helper_functions.cpp
  video_flip.cpp
  Kernel.cu
  )


target_link_libraries(cuda_video_flip CUDA::cudart)
target_link_libraries(cuda_video_flip CUDA::cuda_driver)
