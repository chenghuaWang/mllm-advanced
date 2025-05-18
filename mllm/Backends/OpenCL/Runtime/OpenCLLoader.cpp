/**
 * @file OpenCLLoader.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-17
 *
 * @copyright Copyright (c) 2025
 *
 */
// NOTE:
// This file is highly inspired by MNN's impl.
// see:
// https://github.com/alibaba/MNN/blob/master/source/backend/opencl/core/runtime/OpenCLWrapper.cpp

#include <vector>
#include <string>

// This head only works on linux
#include <dlfcn.h>

#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/OpenCL/Runtime/OpenCLLoader.hpp"

namespace mllm::opencl {

bool OpenCLLoader::loadOpenCLDynLib() {
  if (opencl_dynlib_handle_) {
    MLLM_WARN("OpenCL dyn lib is already loaded.");
    return true;
  }

  // This path is only for android platform.
  // Support GPU:
  // Adreno and Mali
  static const std::vector<std::string> possible_opencl_dyn_lib_paths{
      /// -- Android sys path?
      "libOpenCL.so",
      "libGLES_mali.so",
      "libmali.so",
      "libOpenCL-pixel.so",

      /// -- __aarch64__ path?
      // Qualcomm Adreno
      "/system/vendor/lib64/libOpenCL.so",
      "/system/lib64/libOpenCL.so",
      // Mali
      "/system/vendor/lib64/egl/libGLES_mali.so",
      "/system/lib64/egl/libGLES_mali.so",
  };

  for (const auto& lib_path : possible_opencl_dyn_lib_paths) {
    if (tryingToLoadOpenCLDynLibAndParseSymbols(lib_path)) { return true; }
  }

  return false;
}

bool OpenCLLoader::tryingToLoadOpenCLDynLibAndParseSymbols(const std::string& lib_path) {
  opencl_dynlib_handle_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (opencl_dynlib_handle_ == nullptr) { return false; }

  MLLM_INFO("Load opencl dyn lib: {}", lib_path);

  // Load opencl symbols
  using enable_opencl_f_t = void (*)();
  using load_opencl_ptr_f_t = void* (*)(const char*);

  load_opencl_ptr_f_t load_opencl_ptr_f = nullptr;
  enable_opencl_f_t enable_opencl_f =
      reinterpret_cast<enable_opencl_f_t>(dlsym(opencl_dynlib_handle_, "enableOpenCL"));
  if (enable_opencl_f != nullptr) {
    enable_opencl_f();
    load_opencl_ptr_f =
        reinterpret_cast<load_opencl_ptr_f_t>(dlsym(opencl_dynlib_handle_, "loadOpenCLPointer"));
  }

#define LOAD_FUNCTION_PTR(func_name)                                                       \
  func_name = reinterpret_cast<func_name##_f_t>(dlsym(opencl_dynlib_handle_, #func_name)); \
  if (func_name == nullptr && load_opencl_ptr_f != nullptr) {                              \
    func_name = reinterpret_cast<func_name##_f_t>(load_opencl_ptr_f(#func_name));          \
  }                                                                                        \
  if (func_name == nullptr) {                                                              \
    MLLM_ERROR_EXIT(kError, "Failed to load OpenCL function: {}", #func_name);             \
  }

  LOAD_FUNCTION_PTR(clGetPlatformIDs);
  LOAD_FUNCTION_PTR(clGetPlatformInfo);
  LOAD_FUNCTION_PTR(clGetDeviceIDs);
  LOAD_FUNCTION_PTR(clGetDeviceInfo);

#define LOAD_SVM_FUNCTION_PTR(func_name)                                                   \
  func_name = reinterpret_cast<func_name##_f_t>(dlsym(opencl_dynlib_handle_, #func_name)); \
  if (func_name == nullptr && load_opencl_ptr_f != nullptr) {                              \
    func_name = reinterpret_cast<func_name##_f_t>(load_opencl_ptr_f(#func_name));          \
  }                                                                                        \
  if (func_name == nullptr) { svm_load_error_ = true; }

  LOAD_SVM_FUNCTION_PTR(clSVMAlloc);
  LOAD_SVM_FUNCTION_PTR(clSVMFree);
  LOAD_SVM_FUNCTION_PTR(clEnqueueSVMMap);
  LOAD_SVM_FUNCTION_PTR(clEnqueueSVMUnmap);
  LOAD_SVM_FUNCTION_PTR(clSetKernelArgSVMPointer);

  // TODO More functions and special features to load

#undef LOAD_FUNCTION_PTR
#undef LOAD_SVM_FUNCTION_PTR

  return true;
}

}  // namespace mllm::opencl
