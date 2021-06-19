cmake_minimum_required(VERSION 3.7.2)

project(HORUS_CODE_BLOCK_DEBUG_SINK VERSION 1.0.0 LANGUAGES C)

add_library(horus_code_block_debug_sink SHARED
  ../../Horus_code_block.h
  Horus_code_block_debug_sink.c
  )
target_compile_options(horus_code_block_debug_sink
  PRIVATE
  $<$<C_COMPILER_ID:GNU>:
  -Wno-stringop-overflow
  >
  )
