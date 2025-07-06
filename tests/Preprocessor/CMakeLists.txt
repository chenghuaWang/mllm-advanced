if(MLLM_BUILD_ARM_BACKEND)
  set(_BACKEND_LIB_FOR_NN_TESTS MllmArmBackend)
else()
  set(_BACKEND_LIB_FOR_NN_TESTS MllmX86Backend)
endif()

add_executable(Mllm-Processor-ImageLoadAndStoreTest ImageLoadAndStoreTest.cpp)
target_link_libraries(Mllm-Processor-ImageLoadAndStoreTest PRIVATE MllmRT
                                                ${_BACKEND_LIB_FOR_NN_TESTS})
target_include_directories(Mllm-Processor-ImageLoadAndStoreTest PRIVATE ${MLLM_INCLUDE_DIR})
