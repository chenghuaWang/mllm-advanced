file(GLOB_RECURSE MLLM_X86_BACKEND_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_library(MllmX86Backend SHARED ${MLLM_X86_BACKEND_SRC})
target_link_libraries(MllmX86Backend PUBLIC MllmRT)
target_compile_options(MllmX86Backend
                       PUBLIC ${MLLM_X86_BACKEND_COMPILE_OPTIONS})
