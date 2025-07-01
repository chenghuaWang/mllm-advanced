file(GLOB_RECURSE MLLM_OPENCL_BACKEND_SRC_RT
     ${CMAKE_CURRENT_LIST_DIR}/Runtime/*.cpp)

add_library(MllmOpenCLBackend SHARED ${MLLM_OPENCL_BACKEND_SRC_RT}
                                     OpenCLBackend.cpp)
target_link_libraries(MllmOpenCLBackend PUBLIC MllmRT)
target_include_directories(
  MllmOpenCLBackend
  PUBLIC
    ${MLLM_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/Vendors/OpenCL-Headers>
    $<INSTALL_INTERFACE:include/mllm/Backends/OpenCL/Vendors/OpenCL-Headers/>)

add_library(MllmOpenCLBackend_ExtHeaders INTERFACE)
target_include_directories(
  MllmOpenCLBackend_ExtHeaders
  INTERFACE
    ${MLLM_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/Vendors/OpenCL-Headers>
    $<INSTALL_INTERFACE:include/mllm/Backends/OpenCL/Vendors/OpenCL-Headers/>)

target_link_libraries(MllmOpenCLBackend INTERFACE MllmOpenCLBackend_ExtHeaders)

install(
  TARGETS MllmOpenCLBackend MllmOpenCLBackend_ExtHeaders
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Vendors/OpenCL-Headers
  DESTINATION include/mllm/Backends/OpenCL/Vendors
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")
