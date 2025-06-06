if(DEFINED ENV{QNN_SDK_ROOT})
  message(STATUS "QNN_SDK_ROOT is set to: $ENV{QNN_SDK_ROOT}")
else()
  message(ERROR "QNN_SDK_ROOT is not set.")
endif()

if(DEFINED ENV{HEXAGON_SDK_ROOT})
  message(STATUS "HEXAGON_SDK_ROOT is set to: $ENV{HEXAGON_SDK_ROOT}")
else()
  message(ERROR "HEXAGON_SDK_ROOT is not set.")
endif()

file(GLOB_RECURSE MLLM_QNN_BACKEND_SRC_RUNTIME
     ${CMAKE_CURRENT_LIST_DIR}/Runtime/*.cpp)
file(GLOB_RECURSE MLLM_QNN_BACKEND_SRC_PASSES
     ${CMAKE_CURRENT_LIST_DIR}/Passes/*.cpp)
file(GLOB_RECURSE MLLM_QNN_BACKEND_SRC_OPS
     ${CMAKE_CURRENT_LIST_DIR}/Ops/*.cpp)

add_library(
  MllmQnnBackend SHARED
  QnnAllocator.cpp QnnBackend.cpp QnnDispatcher.cpp
  ${MLLM_QNN_BACKEND_SRC_RUNTIME} ${MLLM_QNN_BACKEND_SRC_PASSES} ${MLLM_QNN_BACKEND_SRC_OPS})
target_include_directories(
  MllmQnnBackend
  PUBLIC $ENV{QNN_SDK_ROOT}/include $ENV{QNN_SDK_ROOT}/include/QNN
         $ENV{HEXAGON_SDK_ROOT}/incs)
target_link_libraries(MllmQnnBackend PUBLIC MllmRT)
