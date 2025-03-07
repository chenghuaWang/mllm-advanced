set(HWY_ENABLE_TESTS OFF)
set(HWY_ENABLE_EXAMPLES OFF)
set(HWY_ENABLE_CONTRIB OFF)
set(BUILD_TESTING OFF)
add_subdirectory(Vendors/highway)

file(GLOB_RECURSE MLLM_X86_BACKEND_OPS_SRC ${CMAKE_CURRENT_LIST_DIR}/Ops/*.cpp)
file(GLOB_RECURSE MLLM_X86_BACKEND_KERNELS_SRC
     ${CMAKE_CURRENT_LIST_DIR}/Kernels/*.cpp)

add_library(
  MllmX86Backend SHARED
  ${MLLM_X86_BACKEND_OPS_SRC} ${MLLM_X86_BACKEND_KERNELS_SRC} X86Allocator.cpp
  X86Backend.cpp)
target_link_libraries(MllmX86Backend PUBLIC MllmRT hwy)
target_compile_options(MllmX86Backend
                       PUBLIC ${MLLM_X86_BACKEND_COMPILE_OPTIONS})
