enable_language(CUDA)

file(GLOB_RECURSE MLLM_CUDA_BACKEND_SRC_OP ${CMAKE_CURRENT_LIST_DIR}/Ops/*.cpp)
file(GLOB_RECURSE MLLM_CUDA_BACKEND_SRC_KERNELS
     ${CMAKE_CURRENT_LIST_DIR}/Kernels/*.cpp)

find_package(CUDAToolkit REQUIRED)

add_library(
  MllmCUDABackend SHARED
  ${MLLM_CUDA_BACKEND_SRC_OP} ${MLLM_CUDA_BACKEND_SRC_KERNELS} CUDAAllocator.cpp CUDABackend.cpp)
target_link_libraries(MllmCUDABackend PUBLIC MllmRT CUDA::cudart)
