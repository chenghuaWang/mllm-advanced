/**
 * @file OpenCLRuntime.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include <sstream>
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
      device_info_.fe_set_workgroup_attr_ = true;
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
#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#define CL_CONTEXT_PRIORITY_HINT_QCOM 0x40C9
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

        device_ext_properties.push_back(CL_CONTEXT_PERF_HINT_QCOM);
        device_ext_properties.push_back(CL_PERF_HINT_HIGH_QCOM);
        device_ext_properties.push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
        device_ext_properties.push_back(CL_PRIORITY_HINT_LOW_QCOM);
        device_info_.fe_support_low_power_ = true;

#undef CL_CONTEXT_PERF_HINT_QCOM
#undef CL_PERF_HINT_HIGH_QCOM
#undef CL_CONTEXT_PRIORITY_HINT_QCOM
#undef CL_PRIORITY_HINT_LOW_QCOM
        break;
      default: break;
    }
  }

  auto device_exts = gpu_device_ptr_.get()->getInfo<CL_DEVICE_EXTENSIONS>();

  auto check_device_support_fe = [](const cl::Device& device, const std::string& feat_str) -> bool {
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    auto pos = extensions.find(feat_str);
    return (pos != std::string::npos);
  };

  device_info_.fe_support_android_hardware_buffer_ =
      (device_info_.gpu_arch_ == OpenCLDeviceInfo::GpuArch::kAdreno)
      && check_device_support_fe(*(gpu_device_ptr_.get()),
                                 "cl_qcom_android_ahardwarebuffer_host_ptr");

  if (nullptr != device_info_.context_ptr_) {
    ctx_ptr_ =
        std::shared_ptr<cl::Context>((cl::Context*)device_info_.context_ptr_, [](void* ptr) {});
  } else {
    cl_int res;
    if (device_ext_properties.size() > 0) {
      device_ext_properties.push_back(0);
      ctx_ptr_ =
          std::make_shared<cl::Context>(std::vector<cl::Device>({*gpu_device_ptr_}),
                                        device_ext_properties.data(), nullptr, nullptr, &res);
    } else {
      ctx_ptr_ = std::make_shared<cl::Context>(std::vector<cl::Device>({*gpu_device_ptr_}), nullptr,
                                               nullptr, nullptr, &res);
    }
    MLLM_CHECK_OPENCL_SUCCESS(res, "context create error.")
  }

  cl_int res;
  if (ext_has_property_hints) {
    cl_queue_properties prop[] = {CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_LOW_KHR, 0};

    command_queue_ptr_ = std::make_shared<cl::CommandQueue>(clCreateCommandQueueWithProperties(
        (*ctx_ptr_).get(), (*gpu_device_ptr_).get(), prop, &res));
  } else {
    cl_command_queue_properties properties = 0;
    command_queue_ptr_ =
        std::make_shared<cl::CommandQueue>(*ctx_ptr_, *gpu_device_ptr_, properties, &res);
  }
  MLLM_CHECK_OPENCL_SUCCESS(res, "command queue create error.")

  command_queue_tuning_ptr_ = std::make_shared<cl::CommandQueue>(*ctx_ptr_, *gpu_device_ptr_,
                                                                 CL_QUEUE_PROFILING_ENABLE, &res);

  cur_command_queue_ptr_ = command_queue_ptr_.get();
  MLLM_CHECK_OPENCL_SUCCESS(gpu_device_ptr_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                                     &device_info_.gpu_global_mem_cache_size_),
                            "");
  MLLM_CHECK_OPENCL_SUCCESS(
      gpu_device_ptr_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_info_.gpu_compute_units_num_),
      "");
  MLLM_CHECK_OPENCL_SUCCESS(
      gpu_device_ptr_->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &device_info_.max_freq_), "");
  MLLM_CHECK_OPENCL_SUCCESS(
      gpu_device_ptr_->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &device_info_.max_mem_alloc_size_),
      "");
  MLLM_CHECK_OPENCL_SUCCESS(
      gpu_device_ptr_->getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &device_info_.max_local_mem_size_), "");
  device_info_.max_work_group_size_ = gpu_device_ptr_->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

  {
    cl_device_fp_config fp_cfg;
    auto success = gpu_device_ptr_->getInfo(CL_DEVICE_HALF_FP_CONFIG, &fp_cfg);
    device_info_.fe_support_fp16_ = CL_SUCCESS == success && fp_cfg > 0;
    bool has_fp16_ext = check_device_support_fe(*(gpu_device_ptr_.get()), "cl_khr_fp16");
    device_info_.fe_support_fp16_ = (device_info_.fe_support_fp16_ && has_fp16_ext);
  }

  if (check_device_support_fe(*(gpu_device_ptr_.get()), "cl_arm_integer_dot_product_int8")) {
    device_info_.fe_support_dot_int8_ = true;
  }
  if (check_device_support_fe(*(gpu_device_ptr_.get()),
                              "cl_arm_integer_dot_product_accumulate_int8")) {
    device_info_.fe_support_dot_acc_int8_ = true;
  }

  // recordable queue
  // TODO, for specific gpu arch, need to include diff ext provided by vendors.

  {
    // Init info
    size_t max_height, max_width;
    MLLM_CHECK_OPENCL_SUCCESS(gpu_device_ptr_->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &max_height),
                              "");
    MLLM_CHECK_OPENCL_SUCCESS(gpu_device_ptr_->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &max_width),
                              "");
    max_image_size_ = {max_height, max_width};
  }

  do {
    int dims = 3;
    MLLM_CHECK_OPENCL_SUCCESS(gpu_device_ptr_->getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &dims),
                              "");

    if (dims < 3) {
      std::vector<uint32_t> work_item(3, 8);
      max_work_items_ = work_item;
      break;
    }
    cl::vector<cl::size_type> _work_items(dims, 1);
    MLLM_CHECK_OPENCL_SUCCESS(gpu_device_ptr_->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &_work_items),
                              "");

    std::vector<uint32_t> work_items(dims, 1);
    for (int i = 0; i < dims; ++i) { work_items[i] = _work_items[i]; }
    max_work_items_ = work_items;
  } while (false);
}

bool MllmOpenCLRuntime::compileKernel(OpenCLKernel::ptr_t kernel) { return true; }

}  // namespace mllm::opencl
