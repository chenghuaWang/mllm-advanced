if(MLLM_BUILD_ARM_BACKEND)
  set(_BACKEND_LIB_FOR_NN_TESTS MllmArmBackend)
else()
  set(_BACKEND_LIB_FOR_NN_TESTS MllmX86Backend)
endif()

add_executable(Mllm-IR-LlmIRTest LlmIR.cpp)
target_link_libraries(Mllm-IR-LlmIRTest PRIVATE MllmRT
                                                ${_BACKEND_LIB_FOR_NN_TESTS})
target_include_directories(Mllm-IR-LlmIRTest PRIVATE ${MLLM_INCLUDE_DIR})
