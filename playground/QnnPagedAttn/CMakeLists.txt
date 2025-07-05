cmake_minimum_required(VERSION 3.21)
project(
  mllm-qnn-paged-attn
  VERSION 1.0.1
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(mllm)

add_executable(mllm-qnn-paged-attn main.cpp)
target_link_libraries(
  mllm-qnn-paged-attn PRIVATE mllm::MllmRT mllm::MllmArmBackend
                                        mllm::MllmQnnBackend)
target_include_directories(mllm-qnn-paged-attn
                           PRIVATE ${MLLM_INCLUDE_DIR})
