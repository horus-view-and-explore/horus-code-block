cmake_minimum_required(VERSION 3.7.2)

project(HORUS_CODE_BLOCK_C_EXAMPLE VERSION 1.0.0 LANGUAGES C)

add_library(horus_code_block_c_example SHARED
  ../Horus_code_block.h
  Horus_code_block_c_example.c
  )
