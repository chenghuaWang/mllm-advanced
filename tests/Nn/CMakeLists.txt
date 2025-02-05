if(MLLM_BUILD_ARM_BACKEND)
  set(_BACKEND_LIB_FOR_NN_TESTS MllmArmBackend)
else()
  set(_BACKEND_LIB_FOR_NN_TESTS MllmX86Backend)
endif()

add_executable(Mllm-Nn-OpCallTest OpCall.cpp)
target_link_libraries(Mllm-Nn-OpCallTest PRIVATE gtest_main MllmRT
                                                 ${_BACKEND_LIB_FOR_NN_TESTS})
target_include_directories(Mllm-Nn-OpCallTest PRIVATE ${MLLM_INLCUDE_DIR})

add_executable(Mllm-Nn-ModuleCreateTest ModuleCreate.cpp)
target_link_libraries(Mllm-Nn-ModuleCreateTest
                      PRIVATE gtest_main MllmRT ${_BACKEND_LIB_FOR_NN_TESTS})
target_include_directories(Mllm-Nn-ModuleCreateTest PRIVATE ${MLLM_INLCUDE_DIR})

include(GoogleTest)

if(NOT MLLM_CROSS_COMPILE)
  gtest_discover_tests(Mllm-Nn-OpCallTest)
  gtest_discover_tests(Mllm-Nn-ModuleCreateTest)
endif()
