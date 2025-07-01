if(MLLM_BUILD_ARM_BACKEND)
  set(_BACKEND_LIB_FOR_NN_TESTS MllmArmBackend)
else()
  set(_BACKEND_LIB_FOR_NN_TESTS MllmX86Backend)
endif()

add_executable(mllm-quantizer main.cpp FlatModuleBuilder.cpp)
target_link_libraries(mllm-quantizer PUBLIC MllmRT ${_BACKEND_LIB_FOR_NN_TESTS})
target_include_directories(mllm-quantizer PRIVATE ${MLLM_INCLUDE_DIR})
