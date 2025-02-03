cmake_minimum_required(VERSION 3.21)
project(
  mllm
  VERSION 1.0.1
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MLLM_ENABLE_TEST "Enable mllm test" ON)
option(MLLM_ENABLE_BENCHMARK "Enable mllm benchmark" ON)

message(STATUS "CXX Compiler=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Compiler version=${CMAKE_CXX_COMPILER_VERSION}")

set(MLLM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(third_party/fmt EXCLUDE_FROM_ALL)

add_subdirectory(mllm)

if(MLLM_ENABLE_TEST)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(tests)
endif()

if(MLLM_ENABLE_BENCHMARK)
  add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
  add_subdirectory(benchmarks)
endif()

add_subdirectory(tools/Quantizer)
add_subdirectory(tools/TokenizeChecker)
