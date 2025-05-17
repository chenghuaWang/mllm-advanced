add_executable(Mllm-OpenCL-DetectTest OpenCLDetectTest.cpp)
target_link_libraries(Mllm-OpenCL-DetectTest PRIVATE MllmRT MllmOpenCLBackend)
target_include_directories(Mllm-OpenCL-DetectTest PRIVATE ${MLLM_INCLUDE_DIR})
