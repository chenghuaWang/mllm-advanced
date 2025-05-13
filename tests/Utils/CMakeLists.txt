if(MLLM_BUILD_ARM_BACKEND)
  set(_BACKEND_LIB_FOR_NN_TESTS MllmArmBackend)
else()
  set(_BACKEND_LIB_FOR_NN_TESTS MllmX86Backend)
endif()

add_executable(Mllm-Utils-MllmThreadPool MllmThreadPool.cpp)
target_link_libraries(Mllm-Utils-MllmThreadPool PRIVATE gtest_main MllmRT
                                                 ${_BACKEND_LIB_FOR_NN_TESTS})
target_include_directories(Mllm-Utils-MllmThreadPool PRIVATE ${MLLM_INCLUDE_DIR})