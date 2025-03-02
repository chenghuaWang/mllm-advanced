include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  message(STATUS "nvcc path : ${CMAKE_CUDA_COMPILER}")
else()
  message(WARNING "nvcc not found. Please check CUDA is installed correctly!")
endif()

add_executable(MllmCudaKernels-ElewiseOps ElewiseOps.cu)
target_link_libraries(MllmCudaKernels-ElewiseOps MllmCUDABackend)
