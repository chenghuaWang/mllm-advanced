file(GLOB_RECURSE MLLM_ARM_BACKEND_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_library(MllmArmBackend SHARED ${MLLM_ARM_BACKEND_SRC})
target_link_libraries(MllmArmBackend PUBLIC MllmRT -fopenmp)
target_compile_options(MllmArmBackend PUBLIC ${MLLM_ARM_BACKEND_COMPILE_OPTIONS}
                                             -fopenmp)
