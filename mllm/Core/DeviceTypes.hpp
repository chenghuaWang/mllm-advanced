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

#include <tuple>
#include <cstdint>
#include <string>

namespace mllm {

enum DeviceTypes {  // NOLINT
  kDeviceTypes_Start = 0,

  kCPU,
  kCUDA,
  kOpenCL,
  kQNN,

  kDeviceTypes_End,
};

inline const char* deviceTypes2Str(DeviceTypes type) {
  switch (type) {
    case DeviceTypes::kDeviceTypes_Start: return "DeviceTypes_Start";
    case DeviceTypes::kCPU: return "CPU";
    case DeviceTypes::kCUDA: return "CUDA";
    case DeviceTypes::kOpenCL: return "OpenCL";
    case DeviceTypes::kQNN: return "QNN";
    case DeviceTypes::kDeviceTypes_End: return "DeviceTypes_End";
    default: return "Unknown";
  }
}

inline DeviceTypes str2DeviceType(const std::string& type_str) {
  if (type_str == "CPU") {
    return DeviceTypes::kCPU;
  } else if (type_str == "CUDA") {
    return DeviceTypes::kCUDA;
  } else if (type_str == "OpenCL") {
    return DeviceTypes::kOpenCL;
  } else if (type_str == "QNN") {
    return DeviceTypes::kQNN;
  } else {
    return DeviceTypes::kDeviceTypes_End;
  }
}

// FIXME: Impl this struct. compatible with dlpack's definition
struct DeviceType {
  explicit DeviceType(const std::string& device_binary) {
    auto [type_tt, id_tt] = parse(device_binary);
    type_ = type_tt;
    id_ = id_tt;
  }

  DeviceType(DeviceTypes device_type, int32_t id = 0)  // NOLINT: google-explicit-constructor
      : type_(device_type), id_(id) {}

  bool operator=(const DeviceType& d) { return d.type_ == type_ && d.id_ == id_; }

  inline std::string str() {
    return std::string(deviceTypes2Str(type_)) + ":" + std::to_string(id_);
  }

  std::tuple<DeviceTypes, int32_t> parse(const std::string& device_binary) {
    size_t colon_pos = device_binary.find(':');
    std::string type_str;
    int32_t id = 0;

    if (colon_pos != std::string::npos) {
      type_str = device_binary.substr(0, colon_pos);
      std::string id_str = device_binary.substr(colon_pos + 1);
      id = std::stoi(id_str);
    } else {
      type_str = device_binary;
    }

    type_ = str2DeviceType(type_str);
    id_ = id;

    return std::make_tuple(type_, id_);
  }

  DeviceTypes type_ = DeviceTypes::kCPU;
  int32_t id_ = 0;
};

}  // namespace mllm

namespace std {
template<>
struct hash<::mllm::DeviceType> {
  std::size_t operator()(const ::mllm::DeviceType& d) const {
    using std::hash;
    using std::size_t;

    // Combine the hash of type_ and id_
    size_t h1 = hash<int>()(static_cast<int>(d.type_));
    size_t h2 = hash<int>()(d.id_);
    return h1 ^ (h2 << 1);  // Simple combination of hashes
  }
};
}  // namespace std
