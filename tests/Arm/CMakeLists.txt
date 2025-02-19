add_executable(Mllm-ArmTest Kernels.cpp)
target_link_libraries(Mllm-ArmTest PRIVATE gtest_main MllmRT MllmArmBackend)
target_include_directories(Mllm-ArmTest PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)

if(NOT MLLM_CROSS_COMPILE)
  gtest_discover_tests(Mllm-Nn-OpCallTest)
  gtest_discover_tests(Mllm-Nn-ModuleCreateTest)
endif()
