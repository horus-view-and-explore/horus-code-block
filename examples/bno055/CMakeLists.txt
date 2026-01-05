cmake_minimum_required(VERSION 3.11)

project(HORUS_CODE_BLOCK_BNO055 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "This code block is only supported on Linux")
endif()

add_subdirectory(vendor/libi2c)

# include(FetchContent)

# FetchContent_Declare(json
#   GIT_REPOSITORY https://github.com/nlohmann/json.git
#   GIT_TAG v3.7.3)

# FetchContent_GetProperties(json)
# if(NOT json_POPULATED)
#   FetchContent_Populate(json)
#   add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
# endif()

string(TOLOWER ${PROJECT_NAME} TARGET_NAME)
add_library(${TARGET_NAME} SHARED
        ../../Horus_code_block.h
        src/Horus_code_block_bno055.cpp
        src/User_context.cpp
        src/BNO055.cpp
        src/Gpio_trigger.cpp
        )
target_include_directories(${TARGET_NAME} PRIVATE ${LIBI2C_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE ${LIBI2C_LIBRARY}) # nlohmann_json::nlohmann_json)
