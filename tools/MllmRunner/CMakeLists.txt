if(MLLM_BUILD_ARM_BACKEND)
  set(_BACKEND_LIB_FOR_NN_TESTS MllmArmBackend)
else()
  set(_BACKEND_LIB_FOR_NN_TESTS MllmX86Backend)
endif()

add_executable(mllm-runner main.cpp)
target_link_libraries(mllm-runner PUBLIC MllmRT ${_BACKEND_LIB_FOR_NN_TESTS})
