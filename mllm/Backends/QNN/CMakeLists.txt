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

add_library(MllmQnnBackend SHARED QnnAllocator.cpp QnnBackend.cpp
                                  QnnDispatcher.cpp)
target_include_directories(
  MllmQnnBackend
  PRIVATE $ENV{QNN_SDK_ROOT}/include $ENV{QNN_SDK_ROOT}/include/QNN
          $ENV{HEXAGON_SDK_ROOT}/incs)
target_link_libraries(MllmQnnBackend PUBLIC MllmRT)
