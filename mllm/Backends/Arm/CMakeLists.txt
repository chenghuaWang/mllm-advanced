file(GLOB_RECURSE MLLM_ARM_BACKEND_SRC_OP ${CMAKE_CURRENT_LIST_DIR}/Ops/*.cpp)
file(GLOB_RECURSE MLLM_ARM_BACKEND_SRC_KERNELS
     ${CMAKE_CURRENT_LIST_DIR}/Kernels/*.cpp)

add_library(
  MllmArmBackend SHARED
  ${MLLM_ARM_BACKEND_SRC_OP} ${MLLM_ARM_BACKEND_SRC_KERNELS} ArmAllocator.cpp
  ArmBackend.cpp)
target_link_libraries(MllmArmBackend PUBLIC MllmRT -fopenmp)
target_compile_options(MllmArmBackend PUBLIC ${MLLM_ARM_BACKEND_COMPILE_OPTIONS}
                                             -fopenmp)
