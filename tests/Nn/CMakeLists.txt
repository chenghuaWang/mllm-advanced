add_executable(Mllm-Nn-OpCallTest OpCall.cpp)
target_link_libraries(Mllm-Nn-OpCallTest PRIVATE gtest_main MllmRT
                                                 MllmX86Backend)
target_include_directories(Mllm-Nn-OpCallTest PRIVATE ${MLLM_INLCUDE_DIR})
include(GoogleTest)

if(NOT MLLM_CROSS_COMPILE)
  gtest_discover_tests(Mllm-Nn-OpCallTest)
endif()
