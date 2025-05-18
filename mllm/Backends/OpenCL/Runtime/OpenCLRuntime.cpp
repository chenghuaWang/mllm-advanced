/**
 * @file OpenCLRuntime.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <sstream>
#include <CL/opencl.hpp>
#include "mllm/Backends/OpenCL/Runtime/OpenCLRuntime.hpp"
#include "mllm/Backends/OpenCL/Runtime/OpenCLLoader.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::opencl {

MllmOpenCLRuntime::MllmOpenCLRuntime(void* ctx_ptr, int platform_size, int platform_id,
                                     int device_id) {
  device_info_.context_ptr_ = ctx_ptr;
  device_info_.platform_size_ = platform_size;
  device_info_.platform_id_ = platform_id;
  device_info_.device_id_ = device_id;

  // Check platform
  std::vector<cl::Platform> platforms;
  MLLM_CHECK_OPENCL_SUCCESS(cl::Platform::get(&platforms), "");
  MLLM_RT_ASSERT(platforms.size() > 0 && platforms.size() > platform_id);
  cl::Platform::setDefault(platforms[platform_id]);
  device_info_.platform_name_ = platforms[platform_id].getInfo<CL_PLATFORM_NAME>();
  device_info_.platform_vendor_ = platforms[platform_id].getInfo<CL_PLATFORM_VENDOR>();
  device_info_.platform_version_ = platforms[platform_id].getInfo<CL_PLATFORM_VERSION>();

  // Check device
  std::vector<cl::Device> gpu_devices;
  MLLM_CHECK_OPENCL_SUCCESS(platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &gpu_devices),
                            "");
  MLLM_RT_ASSERT(gpu_devices.size() > 0 && gpu_devices.size() > device_id);
  gpu_device_ptr_ = std::make_shared<cl::Device>(gpu_devices[device_id]);
  MLLM_RT_ASSERT(gpu_device_ptr_ != nullptr);
  device_info_.device_name_ = gpu_device_ptr_->getInfo<CL_DEVICE_NAME>();
  device_info_.device_vendor_ = gpu_device_ptr_->getInfo<CL_DEVICE_VENDOR>();
  device_info_.device_version_ = gpu_device_ptr_->getInfo<CL_DEVICE_VERSION>();

  // We only support OpenCL version > 3.0
  {
    std::stringstream ss(device_info_.device_version_);
    std::string _;
    ss >> _ >> device_info_.opencl_version_ >> _;
  }
  MLLM_RT_ASSERT(device_info_.opencl_version_ >= 3.0f);

  auto& ocl_loader = OpenCLLoader::instance();

  // Enable SVM if SVM is supported.
  if (ocl_loader.isSVMSymbolLoaded()) {
    MLLM_CHECK_OPENCL_SUCCESS(gpu_device_ptr_->getInfo(CL_DEVICE_SVM_CAPABILITIES,
                                                       &device_info_.fe_support_svm_capabilities_),
                              "")
    if (device_info_.fe_support_svm_capabilities_ & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
      device_info_.fe_support_svm_type_ = OpenCLDeviceInfo::SVMSupport::kFineBuffer;
    } else if (device_info_.fe_support_svm_capabilities_ & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
      device_info_.fe_support_svm_type_ = OpenCLDeviceInfo::SVMSupport::kCoarseBuffer;
    }
  } else {
    device_info_.fe_support_svm_type_ = OpenCLDeviceInfo::SVMSupport::kNotSupported;
  }

  // Handle specific GPU Arch
  // ADRENO
  if (device_info_.device_name_.find("Qualcomm") != std::string::npos
      || device_info_.device_name_.find("QUALCOMM Adreno") != std::string::npos) {
    device_info_.gpu_arch_ = OpenCLDeviceInfo::GpuArch::kAdreno;

    std::string adreno_version =
        device_info_.device_version_.substr(device_info_.device_version_.size() - 3);

    if (adreno_version >= "730") {
      device_info_.fe_set_workgroup_attr = true;
      device_info_.gpu_level_ = OpenCLDeviceInfo::GpuLevel::kTop;
    } else {
      NYI("Unsupported Adreno Version");
    }
  } else {
    NYI("Unsupported GPU Arch");
  }

  // Set properties for specific GPU arch
  auto extensions = platforms[platform_id].getInfo<CL_PLATFORM_EXTENSIONS>();
  bool ext_has_property_hints = (extensions.find("cl_khr_priority_hints") != std::string::npos);
  std::vector<cl_context_properties> device_ext_properties;

  if (ext_has_property_hints) {
    switch (device_info_.gpu_arch_) {
      case OpenCLDeviceInfo::GpuArch::kAdreno:
        // TODO
        break;
      default: break;
    }
  }

  // TODO check if has other features. Such as fp16, int8, low power...
}

bool MllmOpenCLRuntime::compileKernel(OpenCLKernel::ptr_t kernel) { return true; }

}  // namespace mllm::opencl
