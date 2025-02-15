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
  kReference,
  kTensorMemTypes_End,
};

struct QuantizePayload {};

struct PTQuantizePayload : public QuantizePayload {
  float bias_ = 0.f;
  float scale_ = 0.f;
};

struct PCQuantizePayload : public QuantizePayload {
  int channel_dim_ = 0;
  float* bias_ = nullptr;
  float* scale_ = nullptr;
};

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
 public:
  explicit TensorImpl(const std::vector<size_t>& shape, DataTypes dtype = kFp32,
                      DeviceTypes device = kCPU);

  TensorImpl(const std::vector<size_t>& shape, const std::vector<size_t>& stride,
             const std::vector<size_t>& stride_offsets, DataTypes dtype = kFp32,
             DeviceTypes device = kCPU);

  ~TensorImpl();

  DataTypes dtype() const;

  DeviceTypes device() const;

  void setDtype(DataTypes dtype);

  [[nodiscard]] std::string name() const;

  [[nodiscard]] TensorMemTypes memType() const;

  void setName(const std::string& name);

  void setMemType(TensorMemTypes mem_type);

  [[nodiscard]] uint32_t uuid() const;

  void setUUID(uint32_t uuid);

  [[nodiscard]] std::shared_ptr<QuantizePayload> quantizePayload();

  void setQuantizePayload(const std::shared_ptr<QuantizePayload>& payload);

  [[nodiscard]] void* rptr() const;

  template<typename T>
  T* ptr() {
    return static_cast<T*>(data_);
  }

  // E.g.:
  // Tensor t = Tensor::ones({1, 1024, 128, 12});
  // t's strides is {1024*128*12, 128*12, 12, 1}.
  // Tensor a = t.refFrom({{}, {}, {100: Auto}, {}});
  // a's shape is {1, 1024, 28, 12}.
  // a's stride is still {1024*128*12, 128*12, 12, 1}.
  // a's stride_offsets_ is {0, 0, 100, 0}.
  // When using:
  // a.offsettedPtr<T>({0, x, 3, 0});
  // The offsets is 128*12*x + 12*(3+100).
  //
  // Use case in mllm: KVCache.
  // The KVCache in mllm is [B, H, S, D]. KVCache will not copy cached data into a new output tensor
  // but reuse the cached tensor. Which means `output = cache.refFrom({0, 0, cur_seq_len_, 0})`.
  // When doing matmul Q @ K^T, we need to iterate the head dim and give gemm kernel with correct M,
  // K, N.
  // for (int i = 0; i < h; ++i) {
  //    float* k_ptr = k.offsettedPtr<float>({0, i, 0, 0}); // shape is [1, 1, cur_seq_len_, D]
  //    ...
  // }
  template<typename T>
  T* offsettedPtr(const std::vector<size_t>& offsets) {
    MLLM_RT_ASSERT_EQ(offsets.size(), shape_len_);

    size_t _offset = 0;
    for (int i = 0; i < shape_len_; ++i) {
      _offset += (offsets[i] + stride_offsets_[i]) * stride_[i];
    }

    return ptr<T>() + _offset;
  }

  char* offsettedRawPtr(const std::vector<size_t>& offsets);

  void _setRawPtr(void* ptr);

  // How many bytes. Not Aligned
  size_t size() const;

  size_t numel() const;

  [[nodiscard]] std::vector<size_t> shape() const;

  [[nodiscard]] std::vector<size_t> stride() const;

  void setShape(const std::vector<int32_t>& shape);

  bool isContiguous() const;

 private:
  void* data_ = nullptr;
  TensorMemTypes mem_type_ = kNormal;
  DataTypes dtype_ = kFp32;
  DeviceTypes device_type_ = kCPU;
  int32_t shape_len_ = 0;
  int32_t shape_[MLLM_TENSOR_SHAPE_MAX_LEN];
  int32_t stride_[MLLM_TENSOR_SHAPE_MAX_LEN];
  int32_t stride_offsets_[MLLM_TENSOR_SHAPE_MAX_LEN];
  std::string name_;
  uint32_t custom_32bit_uuid_;
  std::shared_ptr<QuantizePayload> quantize_payload_ = nullptr;
};

}  // namespace mllm
