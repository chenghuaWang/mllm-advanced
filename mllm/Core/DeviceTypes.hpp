/**
 * @file DeviceTypes.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

namespace mllm {

enum DeviceTypes {
  kDeviceTypes_Start = 0,

  kCPU,
  kCUDA,
  kOpenCL,

  kDeviceTypes_End,
};

inline const char* deviceTypes2Str(DeviceTypes type) {
  switch (type) {
    case DeviceTypes::kDeviceTypes_Start: return "kDeviceTypes_Start";
    case DeviceTypes::kCPU: return "kCPU";
    case DeviceTypes::kCUDA: return "kCUDA";
    case DeviceTypes::kOpenCL: return "kOpenCL";
    case DeviceTypes::kDeviceTypes_End: return "kDeviceTypes_End";
    default: return "Unknown";
  }
}

}  // namespace mllm
