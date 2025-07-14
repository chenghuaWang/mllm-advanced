if(MLLM_BUILD_ARM_BACKEND)
  set(_BACKEND_LIB_FOR_MODEL_EXAMPLES MllmArmBackend)
else()
  set(_BACKEND_LIB_FOR_MODEL_EXAMPLES MllmX86Backend)
endif()

add_executable(demo_qwen2vl demo_qwen2vl.cpp)
target_link_libraries(demo_qwen2vl PRIVATE ${_BACKEND_LIB_FOR_MODEL_EXAMPLES}
                                           MllmRT)
