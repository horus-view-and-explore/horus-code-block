cmake_minimum_required(VERSION 3.7.2)

project(HORUS_CODE_BLOCK_NVIDIA_BLOG_NAVINFO_TO_HORUS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

configure_file("${PROJECT_SOURCE_DIR}/lib/json.hpp.in" "${PROJECT_SOURCE_DIR}/lib/json.hpp" COPYONLY)

add_library(Horus_code_block_nvidia_blog_navinfo_to_horus SHARED
  ../../Horus_code_block.h
  Horus_code_block_nvidia_blog_navinfo_to_horus.cpp
  lib/json.hpp
)
