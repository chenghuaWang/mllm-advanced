add_executable(Mllm-Qnn-DetectTest QnnDetectTest.cpp)
target_link_libraries(Mllm-Qnn-DetectTest PRIVATE MllmRT MllmQnnBackend)
target_include_directories(Mllm-Qnn-DetectTest PRIVATE ${MLLM_INCLUDE_DIR})
