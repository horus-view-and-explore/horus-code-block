cmake_minimum_required(VERSION 3.7.2)

project(HORUS_CODE_BLOCK_CPP_EXAMPLE VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(horus_code_block_cpp_examples
  ../Horus_code_block.h
  Horus_code_block_output_message_base.cpp
  Horus_code_block_output_message_base.hpp
  Horus_code_block_user_context_base.cpp
  Horus_code_block_user_context_base.hpp
  )

add_library(horus_code_block_cpp_examples_debug_source SHARED
  ../Horus_code_block.h
  Horus_code_block_debug_source.cpp
  )
target_link_libraries(horus_code_block_cpp_examples_debug_source
  horus_code_block_cpp_examples
  )

add_library(horus_code_block_cpp_examples_lsb_eraser SHARED
  ../Horus_code_block.h
  Horus_code_block_lsb_eraser.cpp
  )
target_link_libraries(horus_code_block_cpp_examples_lsb_eraser
  horus_code_block_cpp_examples
  )

add_library(horus_code_block_cpp_examples_cpu_temperature SHARED
  ../Horus_code_block.h
  Horus_code_block_cpu_temperature.cpp
  )
