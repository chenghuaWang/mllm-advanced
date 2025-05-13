enable_testing()
add_subdirectory(Nn)
add_subdirectory(IR)
add_subdirectory(Utils)

if(MLLM_BUILD_ARM_BACKEND)
  add_subdirectory(Arm)
else()
  add_subdirectory(X86)
endif()

if(MLLM_BUILD_CUDA_BACKEND)
  add_subdirectory(CUDA)
endif()
