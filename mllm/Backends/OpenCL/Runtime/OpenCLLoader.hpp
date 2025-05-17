/**
 * @file OpenCLLoader.hpp
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
// https://github.com/alibaba/MNN/blob/master/source/backend/opencl/core/runtime/OpenCLWrapper.hpp

#pragma once

// Only support android platform.
// The gpu arch we support is :
//  - Adreno(Fully Tested)
//  - Mali(Partially Tested)

#if !defined(__ANDROID__)
#error "Only support android platform with GPU Arch: Adreno and Mali."
#endif

// The CL_TARGET_OPENCL_VERSION is set for devices support OpenCL 3.0.
#define CL_TARGET_OPENCL_VERSION 300

#include "mllm/Utils/Common.hpp"

// The cl.hpp is deprecated. Pls use opencl.hpp instead
#include <CL/opencl.h>

// Special features from hardware vendors.
#include <CL/cl_ext.h>
#ifdef MLLM_OPENCL_GPU_ADRENO
// TODO include and process some adreno based features.
#endif

// Special features from sys side.
#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#endif

#define MLLM_CHECK_OPENCL_SUCCESS(error, info)                                                 \
  if (error != CL_SUCCESS) {                                                                   \
    MLLM_ASSERT_EXIT(kError, "OpenCL device side error. Error code: {}, info: {}", (int)error, \
                     info);                                                                    \
  }

namespace mllm::opencl {

class OpenCLLoader {
 public:
  static OpenCLLoader& instance() {
    static OpenCLLoader instance;
    return instance;
  }

  OpenCLLoader() = default;

  OpenCLLoader(const OpenCLLoader&) = delete;

  OpenCLLoader& operator=(const OpenCLLoader&) = delete;

  bool loadOpenCLDynLib();

  using clGetPlatformIDs_f_t = cl_int(CL_API_CALL*)(cl_uint, cl_platform_id*, cl_uint*);
  using clGetPlatformInfo_f_t = cl_int(CL_API_CALL*)(cl_platform_id, cl_platform_info, size_t,
                                                     void*, size_t*);
  using clGetDeviceIDs_f_t = cl_int(CL_API_CALL*)(cl_platform_id, cl_device_type, cl_uint,
                                                  cl_device_id*, cl_uint*);
  using clGetDeviceInfo_f_t = cl_int(CL_API_CALL*)(cl_device_id, cl_device_info, size_t, void*,
                                                   size_t*);

#define DEFINE_FUNC_PTR_MEMBER(func) func##_f_t func = nullptr
  DEFINE_FUNC_PTR_MEMBER(clGetPlatformIDs);
  DEFINE_FUNC_PTR_MEMBER(clGetPlatformInfo);
  DEFINE_FUNC_PTR_MEMBER(clGetDeviceIDs);
  DEFINE_FUNC_PTR_MEMBER(clGetDeviceInfo);
#undef DEFINE_FUNC_PTR_MEMBER

 private:
  bool tryingToLoadOpenCLDynLibAndParseSymbols(const std::string& lib_path);

  void* opencl_dynlib_handle_ = nullptr;
};

}  // namespace mllm::opencl
