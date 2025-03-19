enable_language(CUDA)

add_executable(Mllm-CUDA-Context-Test Context.cpp)
target_link_libraries(Mllm-CUDA-Context-Test
                      PRIVATE gtest_main MllmRT MllmCUDABackend MllmX86Backend)
target_include_directories(Mllm-CUDA-Context-Test PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-CUDA-Kernels-Test Kernels.cpp)
target_link_libraries(Mllm-CUDA-Kernels-Test
                      PRIVATE gtest_main MllmRT MllmCUDABackend MllmX86Backend)
target_include_directories(Mllm-CUDA-Kernels-Test PRIVATE ${MLLM_INCLUDE_DIR})

include(GoogleTest)

gtest_discover_tests(Mllm-CUDA-Context-Test)
gtest_discover_tests(Mllm-CUDA-Kernels-Test)
