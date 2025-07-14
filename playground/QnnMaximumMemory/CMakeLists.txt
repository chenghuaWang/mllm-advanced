cmake_minimum_required(VERSION 3.21)
project(
  mllm-qnn-maximum-memory
  VERSION 1.0.1
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(mllm)

add_executable(mllm-qnn-maximum-memory main.cpp)
target_link_libraries(
  mllm-qnn-maximum-memory PRIVATE mllm::MllmRT mllm::MllmArmBackend
                                  mllm::MllmQnnBackend)
target_include_directories(mllm-qnn-maximum-memory PRIVATE ${MLLM_INCLUDE_DIR})
