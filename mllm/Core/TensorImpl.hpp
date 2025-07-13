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
#include "mllm/Utils/Common.hpp"
#include "mllm/Core/Storage.hpp"
#include <cstdint>
#include <vector>
#include <memory>
#include <string>

#define MLLM_TENSOR_SHAPE_MAX_LEN 16

namespace mllm {

enum TensorMemTypes : int32_t {  // NOLINT
  kTensorMemTypes_Start = 0,
  kNormal,
  kExtraInput,
  kExtraOutput,
  kManual,
  kGlobal,
  kParams,
  kReference,

  kQnnAppReadWrite,

  kTensorMemTypes_End,
};

// Tensor = (TensorViewImpl) * N
//
//// FIXME: TensorStorage should be made thread safe.
class TensorStorage final : public Storage {
 public:
  ~TensorStorage() override;

  static std::shared_ptr<TensorStorage> create(const std::vector<int32_t>& shape, DataTypes dtype,
                                               DeviceTypes device);

  std::string name_;
  DataTypes dtype_ = kFp32;
  TensorMemTypes mem_type_ = kNormal;
};

// FIXME: TensorView should be made thread safe.
class TensorViewImpl : public std::enable_shared_from_this<TensorViewImpl> {
 public:
  using storage_t = std::shared_ptr<TensorStorage>;
  using shape_t = std::vector<int32_t>;
  using dtype_t = DataTypes;
  using device_t = DeviceTypes;

  TensorViewImpl() = default;

  TensorViewImpl(TensorViewImpl&) = delete;
  TensorViewImpl(const TensorViewImpl&) = delete;
  TensorViewImpl(const TensorViewImpl&&) = delete;

  DataTypes dtype() const;

  DeviceTypes device() const;

  storage_t storage() const;

  [[nodiscard]] std::string name() const;

  [[nodiscard]] TensorMemTypes memType() const;

  [[nodiscard]] uint32_t uuid() const;

  uint64_t address() const;

  [[nodiscard]] void* rptr() const;

  char* offsettedRawPtr(const std::vector<int32_t>& offsets);

  // How many bytes. Not Aligned
  size_t size() const;

  size_t numel() const;

  [[nodiscard]] std::vector<int32_t> shape() const;

  [[nodiscard]] std::vector<int32_t> stride() const;

  bool isContiguous() const;

  std::shared_ptr<TensorViewImpl> clone() const;

  int32_t storageOffset() const;

  // create empty TensorViewImpl
  static std::shared_ptr<TensorViewImpl> create();

  // Will automatic calculate stride for you.
  static std::shared_ptr<TensorViewImpl> create(const std::vector<int32_t>& shape,
                                                const std::shared_ptr<TensorStorage>& storage);

  static std::shared_ptr<TensorViewImpl> create(int32_t storage_offset,
                                                const std::vector<int32_t>& shape,
                                                const std::shared_ptr<TensorStorage>& storage);

  static std::shared_ptr<TensorViewImpl> create(int32_t storage_offset,
                                                const std::vector<int32_t>& shape,
                                                const std::vector<int32_t>& stride,
                                                const std::shared_ptr<TensorStorage>& storage);

  template<typename T>
  T* ptr() {
    return (T*)(((char*)(storage_->ptr_))
                + (size_t)(storage_offset_ * dataTypeSize(storage_->dtype_)));
  }

  template<typename T>
  T* offsettedPtr(const std::vector<int32_t>& offsets) {
    MLLM_RT_ASSERT_EQ(offsets.size(), shape_len_);

    int32_t _offset = 0;
    for (int i = 0; i < shape_len_; ++i) { _offset += offsets[i] * stride_[i]; }

    return ptr<T>() + _offset;
  }

  inline void dropStorage() { storage_ = nullptr; }

 private:
  int32_t shape_len_ = 0;
  int32_t storage_offset_ = 0;
  int32_t shape_[MLLM_TENSOR_SHAPE_MAX_LEN];
  int32_t stride_[MLLM_TENSOR_SHAPE_MAX_LEN];
  std::shared_ptr<TensorStorage> storage_ = nullptr;
};

}  // namespace mllm
