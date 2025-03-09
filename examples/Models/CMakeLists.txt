if(MLLM_BUILD_ARM_BACKEND)
  set(_BACKEND_LIB_FOR_MODEL_EXAMPLES MllmArmBackend)
else()
  set(_BACKEND_LIB_FOR_MODEL_EXAMPLES MllmX86Backend)
endif()

add_executable(demo_ds_qwen2 demo_ds_qwen2.cpp)
target_link_libraries(demo_ds_qwen2 PRIVATE ${_BACKEND_LIB_FOR_MODEL_EXAMPLES}
                                            MllmRT)
