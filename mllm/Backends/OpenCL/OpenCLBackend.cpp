/**
 * @file OpenCLBackend.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-17
 *
 * @copyright Copyright (c) 2025
 *
 */
/// The OpenCLBackend.hpp head should be imported first. Due to its defined some opencl based macros
/// that will be used in the opencl related infrastructure below.
#include "mllm/Backends/OpenCL/OpenCLBackend.hpp"
#include "mllm/Backends/OpenCL/Runtime/OpenCLLoader.hpp"

namespace mllm::opencl {
OpenCLBackend::OpenCLBackend() : BackendBase(kOpenCL) {
  allocator_ = nullptr;
  auto& opencl_handle = OpenCLLoader::instance();
  opencl_handle.loadOpenCLDynLib();

  // clang-format off
  // Show platform info
  cl_uint num_platforms;
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetPlatformIDs(0, nullptr, &num_platforms), "");
  MLLM_RT_ASSERT(num_platforms == 1);

  std::vector<cl_platform_id> platforms(num_platforms);
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "");
  auto platform = platforms[0];

  char platform_name[128], platform_vendor[128], platform_version[128];
  size_t ext_size;
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr), "");
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr), "");
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr), "");
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, nullptr, &ext_size), "");
  char* extensions = new char[ext_size];
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, ext_size, extensions, nullptr), "");
  delete[] extensions;
  MLLM_INFO("Platform: name={}, vendor={}, version={}", platform_name, platform_vendor, platform_version);
  MLLM_INFO("     -> extensions={}", extensions);

  // Show device info
  // 1. get device count
  cl_uint num_devices;
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices), "");
  
  std::vector<cl_device_id> devices(num_devices);
  MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr), "");

  for (cl_uint i = 0; i < num_devices; ++i) {
    cl_device_id device = devices[i];
    
    char device_name[128], device_vendor[128];
    cl_uint compute_units;
    size_t max_work_group_size;
    
    MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr), "");
    MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, nullptr), "");
    MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr), "");
    MLLM_CHECK_OPENCL_SUCCESS(opencl_handle.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr), "");

    MLLM_INFO("Device: id={}, name={}, vendor={}", i, device_name, device_vendor);
    MLLM_INFO("     -> compute_units={}, max_work_group_size={}", compute_units, max_work_group_size);
  }
  // clang-format on
}

std::shared_ptr<OpenCLBackend> createOpenCLBackend() { return std::make_shared<OpenCLBackend>(); }

}  // namespace mllm::opencl
