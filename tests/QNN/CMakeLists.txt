add_executable(Mllm-Qnn-DetectTest QnnDetectTest.cpp)
target_link_libraries(Mllm-Qnn-DetectTest PRIVATE MllmRT MllmQnnBackend)
target_include_directories(Mllm-Qnn-DetectTest PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Qnn-IRLoweringTest QnnIRLoweringTest.cpp)
target_link_libraries(Mllm-Qnn-IRLoweringTest PRIVATE MllmRT MllmQnnBackend
                                                      MllmArmBackend)
target_include_directories(Mllm-Qnn-IRLoweringTest PRIVATE ${MLLM_INCLUDE_DIR})
