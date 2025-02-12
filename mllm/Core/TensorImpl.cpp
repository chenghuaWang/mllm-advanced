/**
 * @file TensorImpl.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include "mllm/Core/TensorImpl.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Engine/Context.hpp"

namespace mllm {

TensorImpl::TensorImpl(const std::vector<size_t>& shape, DataTypes dtype, DeviceTypes device)
    : dtype_(dtype), device_type_(device) {
  shape_len_ = (int32_t)shape.size();
  int _cnt = 0;
  for (unsigned int it : shape) {
    shape_[_cnt] = (int32_t)it;
    stride_offsets_[_cnt++] = 0;
  }
  int _acc = 1;
  stride_[shape_len_ - 1] = 1;
  for (int i = shape_len_ - 1; i > 0; i--) {
    stride_[i - 1] = _acc * shape_[i];
    _acc *= shape_[i];
  }

  custom_32bit_uuid_ = MllmEngineCtx::instance().getUUID();
}

TensorImpl::TensorImpl(const std::vector<size_t>& shape, const std::vector<size_t>& stride,
                       const std::vector<size_t>& stride_offsets, DataTypes dtype,
                       DeviceTypes device)
    : dtype_(dtype), device_type_(device) {
  shape_len_ = (int32_t)shape.size();
  for (int i = 0; i < shape_len_; ++i) {
    shape_[i] = (int32_t)shape[i];
    stride_[i] = (int32_t)stride[i];
    stride_offsets_[i] = (int32_t)stride_offsets[i];
  }

  custom_32bit_uuid_ = MllmEngineCtx::instance().getUUID();
}

TensorImpl::~TensorImpl() {
  switch (mem_type_) {
    case kNormal:
    case kGlobal: MllmEngineCtx::instance().mem()->free(this); break;
    case kExtraInput:
    case kExtraOutput:
    case kParams:
    case kManual:
    case kReference: break;
    default:
      MLLM_WARN("When trying to free TensorImpl, found invalid mem_type_. Mllm will still trying "
                "to free this tensor, but may lead to memory error.");
      MllmEngineCtx::instance().mem()->free(this);
      break;
  };
}

DataTypes TensorImpl::dtype() const { return dtype_; }

DeviceTypes TensorImpl::device() const { return device_type_; }

void TensorImpl::setDtype(DataTypes dtype) { dtype_ = dtype; }

std::string TensorImpl::name() const { return name_; }

[[nodiscard]] TensorMemTypes TensorImpl::memType() const { return mem_type_; }

void TensorImpl::setName(const std::string& name) { name_ = name; }

void TensorImpl::setMemType(TensorMemTypes mem_type) { mem_type_ = mem_type; }

uint32_t TensorImpl::uuid() const { return custom_32bit_uuid_; }

void TensorImpl::setUUID(uint32_t uuid) { custom_32bit_uuid_ = uuid; }

std::shared_ptr<QuantizePayload> TensorImpl::quantizePayload() { return quantize_payload_; }

void TensorImpl::setQuantizePayload(const std::shared_ptr<QuantizePayload>& payload) {
  quantize_payload_ = payload;
}

void* TensorImpl::rptr() const { return data_; }

char* TensorImpl::offsettedRawPtr(const std::vector<size_t>& offsets) {
  MLLM_RT_ASSERT_EQ(offsets.size(), shape_len_);

  size_t _offset = 0;
  for (int i = 0; i < shape_len_; ++i) {
    _offset += (offsets[i] + stride_offsets_[i]) * stride_[i];
  }

  return ptr<char>() + (size_t)((float)_offset * dataTypeSize(dtype_));
}

void TensorImpl::_setRawPtr(void* ptr) { data_ = ptr; }

size_t TensorImpl::size() const {
  size_t acc = 1;
  for (int i = 0; i < shape_len_; i++) { acc *= shape_[i]; }

  acc = (size_t)((float)acc * dataTypeSize(dtype_));

  return acc;
}

size_t TensorImpl::numel() const {
  size_t acc = 1;
  for (int i = 0; i < shape_len_; i++) { acc *= shape_[i]; }
  return acc;
}

std::vector<size_t> TensorImpl::shape() const { return {shape_, shape_ + shape_len_}; }

std::vector<size_t> TensorImpl::stride() const { return {stride_, stride_ + shape_len_}; }

bool TensorImpl::isContiguous() const {
  if (shape_len_ == 0) return true;

  for (int i = 0; i < shape_len_; ++i) {
    if (stride_offsets_[i] != 0) { return false; }
  }

  int expected_stride = 1;
  for (int i = shape_len_ - 1; i >= 0; --i) {
    if (stride_[i] != expected_stride) { return false; }
    expected_stride *= shape_[i];
  }

  return true;
}

}  // namespace mllm
