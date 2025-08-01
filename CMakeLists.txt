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
option(MLLM_ENABLE_JIT "Enable mllm jit" OFF)

message(STATUS "CXX Compiler=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Compiler version=${CMAKE_CXX_COMPILER_VERSION}")

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
   AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386|x86_64"))
  add_compile_options(-Wno-nan-infinity-disabled)
endif()

set(MLLM_INCLUDE_DIR
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/half/include>
    $<INSTALL_INTERFACE:include/mllm>
    $<INSTALL_INTERFACE:include/third_party/>)
set(MLLM_JSON_INCLUDE_DIR
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/json>
    $<INSTALL_INTERFACE:include/third_party/>)

add_subdirectory(third_party/fmt)

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
add_subdirectory(tools/MllmRunner)

# == Install mllm main runtime
set(INSTALL_MLLM_VERSION ${PROJECT_VERSION})
set(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE PATH "Installation directory for headers")
install(
  TARGETS MllmRT
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mllm/
  DESTINATION include/mllm
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "Vendors/*" EXCLUDE)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt
        DESTINATION packages)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb/include/
  DESTINATION include/third_party/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/half/include/
  DESTINATION include/third_party/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/
  DESTINATION include/third_party/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")

install(TARGETS mllm-quantizer mllm-tokenize-checker mllm-power-counter
                mllm-runner RUNTIME DESTINATION bin)

install(
  EXPORT MllmTargets
  FILE MllmTargets.cmake
  NAMESPACE mllm::
  DESTINATION lib/cmake/)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mllmConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/mllmConfig.cmake
  INSTALL_DESTINATION lib/cmake/
  PATH_VARS INCLUDE_INSTALL_DIR)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/mllmConfigVersion.cmake
  VERSION ${INSTALL_MLLM_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mllmConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/mllmConfigVersion.cmake
        DESTINATION lib/cmake/)

# == Build python binding library
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
    ${PROJECT_SOURCE_DIR}/pymllm/_C/PyWrap.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Nn.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Core.cpp
    ${PROJECT_SOURCE_DIR}/pymllm/_C/Engine.cpp)
  target_compile_options(_C PUBLIC ${_py_compile_opts})
  target_link_libraries(_C PRIVATE ${_py_dep_libs})

  install(TARGETS _C MllmRT MllmX86Backend LIBRARY DESTINATION pymllm/)
endif()
