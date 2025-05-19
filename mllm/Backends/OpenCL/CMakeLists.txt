file(GLOB_RECURSE MLLM_OPENCL_BACKEND_SRC_RT
     ${CMAKE_CURRENT_LIST_DIR}/Runtime/*.cpp)

add_library(MllmOpenCLBackend SHARED ${MLLM_OPENCL_BACKEND_SRC_RT}
                                     OpenCLBackend.cpp)
target_link_libraries(MllmOpenCLBackend PUBLIC MllmRT)
target_include_directories(
  MllmOpenCLBackend PUBLIC ${MLLM_INCLUDE_DIR}
                           ${CMAKE_CURRENT_LIST_DIR}/Vendors/OpenCL-Headers)