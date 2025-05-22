/**
 * @file OpenCLRuntime.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <string>
#include <memory>
#include <cstdint>
#include "mllm/Backends/OpenCL/Runtime/OpenCLKernel.hpp"

namespace mllm::opencl {

// TODO define macros for specific arch separately.

struct OpenCLDeviceInfo {
  enum class GpuArch {
    kUnknown,
    kMali,
    kAdreno,
  };

  enum class GpuLevel {
    kUndefined = 0,
    kTop = 1,
    kMedium = 2,
    kLow = 3,
  };

  enum class SVMSupport {
    kNotSupported,
    kFineBuffer,
    kCoarseBuffer,
  };

  // OpenCL Based info
  float opencl_version_ = 3.f;

  // Platform and devices info
  int platform_size_;
  int platform_id_;
  int device_id_;
  std::string platform_name_;
  std::string platform_vendor_;
  std::string platform_version_;
  std::string device_name_;
  std::string device_vendor_;
  std::string device_version_;
  void* context_ptr_ = nullptr;
  GpuArch gpu_arch_ = GpuArch::kUnknown;
  GpuLevel gpu_level_ = GpuLevel::kUndefined;

  // Kernel info
  uint64_t gpu_global_mem_cache_size_;
  uint32_t gpu_compute_units_num_;
  uint32_t max_freq_;
  uint64_t max_mem_alloc_size_;
  uint64_t max_local_mem_size_;
  uint32_t max_threads_per_device_;
  uint32_t max_work_group_size_;
  uint32_t use_recordable_queue_size_ = 0;

  // Features enabled
  bool fe_support_fp16_ = false;
  bool fe_support_recordable_queue_ = false;
  bool fe_support_dot_int8_ = false;
  bool fe_support_dot_acc_int8_ = false;
  bool fe_support_low_power_ = false;
  bool fe_support_android_hardware_buffer_ = false;
  bool fe_set_workgroup_attr_ = false;

  // Shared Virtual Memory support
  cl_device_svm_capabilities fe_support_svm_capabilities_;
  SVMSupport fe_support_svm_type_ = SVMSupport::kNotSupported;
};

class OpenCLKernelPool {
  // TODO
};

class MllmOpenCLRuntime {
 public:
  MllmOpenCLRuntime(void* ctx_ptr, int platform_size, int platform_id, int device_id);

  bool compileKernel(OpenCLKernel::ptr_t kernel);

 private:
  std::shared_ptr<::cl::Device> gpu_device_ptr_;
  std::shared_ptr<::cl::Context> ctx_ptr_;
  std::shared_ptr<::cl::CommandQueue> command_queue_ptr_;
  std::shared_ptr<::cl::CommandQueue> command_queue_tuning_ptr_;
  ::cl::CommandQueue* cur_command_queue_ptr_;

  std::pair<size_t, size_t> max_image_size_;
  std::vector<uint32_t> max_work_items_;

  OpenCLDeviceInfo device_info_;
  OpenCLKernelPool kernel_pool_;
};

}  // namespace mllm::opencl
