# Find Halide. Make sure Halide is installed from pip channel first.
find_package(Halide REQUIRED)

file(GLOB_RECURSE MLLM_JIT_EXT_SCHEMA ${CMAKE_CURRENT_LIST_DIR}/Schema/*.cpp)

add_library(MllmJITExt SHARED Core.cpp ${MLLM_JIT_EXT_SCHEMA})
target_link_libraries(MllmJITExt PUBLIC Halide::Halide)
target_include_directories(MllmJITExt PUBLIC ${MLLM_INCLUDE_DIR}
                                             ${MLLM_JSON_INCLUDE_DIR})
