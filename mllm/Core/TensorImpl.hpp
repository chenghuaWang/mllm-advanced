/**
 * @file TensorImpl.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/DataTypes.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include <cstdint>
#include <vector>
#include <memory>
#include <string>

#define MLLM_TENSOR_SHAPE_MAX_LEN 8

namespace mllm {

enum TensorMemTypes : int32_t {
  kTensorMemTypes_Start = 0,
  kNormal,
  kExtraInput,
  kExtraOutput,
  kManual,
  kGlobal,
  kParams,
  kTensorMemTypes_End,
};

class PTTensorImpl;
class PCTensorImpl;

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
 public:
  explicit TensorImpl(const std::vector<size_t>& shape, DataTypes dtype = kFp32,
                      DeviceTypes device = kCPU);

  ~TensorImpl();

  std::shared_ptr<PTTensorImpl> toPerTensorImpl();

  std::shared_ptr<PCTensorImpl> toPerChannelImpl();

  DataTypes dtype() const;

  DeviceTypes device() const;

  [[nodiscard]] std::string name() const;

  [[nodiscard]] TensorMemTypes memType() const;

  void setName(const std::string& name);

  void setMemType(TensorMemTypes mem_type);

  [[nodiscard]] uint32_t uuid() const;

  void setUUID(uint32_t uuid);

  [[nodiscard]] void* rptr() const;

  template<typename T>
  T* ptr() {
    return static_cast<T*>(data_);
  }

  void _setRawPtr(void* ptr);

  size_t size() const;

  [[nodiscard]] std::vector<size_t> shape() const;

 private:
  void* data_ = nullptr;
  TensorMemTypes mem_type_ = kNormal;
  DataTypes dtype_ = kFp32;
  DeviceTypes device_type_ = kCPU;
  int32_t shape_len_ = 0;
  int32_t shape_[MLLM_TENSOR_SHAPE_MAX_LEN];
  int32_t stride_[MLLM_TENSOR_SHAPE_MAX_LEN];
  std::string name_;
  uint32_t custom_32bit_uuid_;
};

class PTTensorImpl : public TensorImpl {
 public:
 private:
  float bias_ = 0.f;
  float scale_ = 1.f;
};

class PCTensorImpl : public TensorImpl {
 public:
 private:
  void* bias_ = nullptr;
  void* scale_ = nullptr;
};

}  // namespace mllm
