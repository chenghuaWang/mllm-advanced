file(GLOB_RECURSE MLLM_RT_CORE_SRC ${CMAKE_CURRENT_LIST_DIR}/Core/*.cpp)
file(GLOB_RECURSE MLLM_RT_UTILS_SRC ${CMAKE_CURRENT_LIST_DIR}/Utils/*.cpp)
file(GLOB_RECURSE MLLM_RT_ENGINE_SRC ${CMAKE_CURRENT_LIST_DIR}/Engine/*.cpp)
file(GLOB_RECURSE MLLM_RT_NN_SRC ${CMAKE_CURRENT_LIST_DIR}/Nn/*.cpp)
file(GLOB_RECURSE MLLM_RT_MODELS_SRC ${CMAKE_CURRENT_LIST_DIR}/Models/*.cpp)
file(GLOB_RECURSE MLLM_RT_IR_SRC ${CMAKE_CURRENT_LIST_DIR}/IR/*.cpp)
file(GLOB_RECURSE MLLM_RT_PREPROCESSOR_SRC
     ${CMAKE_CURRENT_LIST_DIR}/Preprocessor/*.cpp)
file(GLOB_RECURSE MLLM_RT_SERVER_SRC ${CMAKE_CURRENT_LIST_DIR}/Server/*.cpp)
file(GLOB_RECURSE MLLM_RT_MOBI_LMCACHE_SRC
     ${CMAKE_CURRENT_LIST_DIR}/MobiLMCache/*.cpp)

add_library(
  MllmRT SHARED
  ${MLLM_RT_CORE_SRC}
  ${MLLM_RT_UTILS_SRC}
  ${MLLM_RT_ENGINE_SRC}
  ${MLLM_RT_NN_SRC}
  ${MLLM_RT_PREPROCESSOR_SRC}
  ${MLLM_RT_MODELS_SRC}
  ${MLLM_RT_IR_SRC}
  ${MLLM_RT_SERVER_SRC}
  ${MLLM_RT_MOBI_LMCACHE_SRC})

target_include_directories(MllmRT PUBLIC ${MLLM_INCLUDE_DIR}
                                         ${MLLM_JSON_INCLUDE_DIR})
target_link_libraries(MllmRT PUBLIC fmt::fmt-header-only)

if(MLLM_BUILD_ARM_BACKEND)
  add_subdirectory(Backends/Arm)
  add_subdirectory(Backends/OpenCL)
endif()

if(MLLM_BUILD_QNN_BACKEND)
  add_subdirectory(Backends/QNN)
endif()

if(MLLM_BUILD_CUDA_BACKEND)
  add_subdirectory(Backends/CUDA)
endif()

if(NOT MLLM_CROSS_COMPILE)
  add_subdirectory(Backends/X86)
endif()

if(MLLM_ENABLE_JIT)
  add_subdirectory(JIT)
endif()
