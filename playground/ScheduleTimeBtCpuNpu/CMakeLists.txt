cmake_minimum_required(VERSION 3.21)
project(
  mllm-schedule-time-bt-cpu-npu
  VERSION 1.0.1
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(mllm)

add_executable(mllm-schedule-time-bt-cpu-npu main.cpp)
target_link_libraries(
  mllm-schedule-time-bt-cpu-npu PRIVATE mllm::MllmRT mllm::MllmArmBackend
                                        mllm::MllmQnnBackend)
target_include_directories(mllm-schedule-time-bt-cpu-npu
                           PRIVATE ${MLLM_INCLUDE_DIR})
