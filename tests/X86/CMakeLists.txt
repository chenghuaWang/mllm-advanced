add_executable(Mllm-X86Test Kernels.cpp)
target_link_libraries(Mllm-X86Test PRIVATE gtest_main MllmRT MllmX86Backend)
target_include_directories(Mllm-X86Test PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)
