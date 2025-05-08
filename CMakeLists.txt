cmake_minimum_required(VERSION 3.21)
project(
  mllm
  VERSION 1.0.1
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MLLM_ENABLE_TEST "Enable mllm test" ON)
option(MLLM_ENABLE_BENCHMARK "Enable mllm benchmark" ON)
option(MLLM_ENABLE_EXAMPLE "Enable mllm example" ON)
option(MLLM_ENABLE_PY_MLLM "Enable mllm python binding" OFF)

message(STATUS "CXX Compiler=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Compiler version=${CMAKE_CXX_COMPILER_VERSION}")

set(MLLM_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/half/include/)
set(MLLM_JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/)

add_subdirectory(third_party/fmt EXCLUDE_FROM_ALL)

add_subdirectory(mllm)

if(MLLM_ENABLE_TEST)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(tests)
endif()

if(MLLM_ENABLE_BENCHMARK)
  set(BENCHMARK_ENABLE_TESTING OFF)
  add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
  add_subdirectory(benchmarks)
endif()

if(MLLM_ENABLE_EXAMPLE)
  add_subdirectory(examples/Models)
endif()

add_subdirectory(tools/Quantizer)
add_subdirectory(tools/TokenizeChecker)
add_subdirectory(tools/PowerCounter)

if(MLLM_ENABLE_PY_MLLM)
  target_compile_options(MllmRT PRIVATE -fPIC)

  find_package(Python3 COMPONENTS Interpreter Development)
  include_directories(${Python3_INCLUDE_DIRS})
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)

  set(_py_dep_libs MllmRT MllmX86Backend)

  # pybind compile options
  set(_py_compile_opts
      # Override depends on RTTI.
      # -frtti
      # -fexceptions
      -fPIC)

  # pybind portable lib _C
  pybind11_add_module(
    _C
    SHARED
    ${PROJECT_SOURCE_DIR}/pymllm/_C/PyWarp.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Nn.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Core.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Engine.cpp)
  target_compile_options(_C PUBLIC ${_py_compile_opts})
  target_link_libraries(_C PRIVATE ${_py_dep_libs})

  install(TARGETS _C MllmRT MllmX86Backend LIBRARY DESTINATION pymllm/)
endif()
