enable_testing()
add_subdirectory(Nn)

if(MLLM_BUILD_ARM_BACKEND)
  add_subdirectory(Arm)
endif()

if(MLLM_BUILD_CUDA_BACKEND)
  add_subdirectory(CUDA)
endif()
