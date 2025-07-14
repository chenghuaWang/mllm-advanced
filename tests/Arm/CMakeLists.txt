add_executable(Mllm-ArmTest Kernels.cpp)
target_link_libraries(Mllm-ArmTest PRIVATE gtest_main MllmRT MllmArmBackend)
target_include_directories(Mllm-ArmTest PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-ArmTest-SingleConv3DTest SingleConv3DTest.cpp)
target_link_libraries(Mllm-ArmTest-SingleConv3DTest PRIVATE gtest_main MllmRT
                                                            MllmArmBackend)
target_include_directories(Mllm-ArmTest-SingleConv3DTest
                           PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
