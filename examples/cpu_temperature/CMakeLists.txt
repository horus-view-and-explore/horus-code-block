cmake_minimum_required(VERSION 3.7.2)

project(HORUS_CODE_BLOCK_CPU_TEMPERATURE VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(horus_code_block_cpu_temperature SHARED
  ../../Horus_code_block.h
  Horus_code_block_cpu_temperature.cpp
  )
