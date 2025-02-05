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
  for (unsigned int it : shape) { shape_[_cnt++] = (int32_t)it; }
  int _acc = 1;
  stride_[shape_len_ - 1] = 1;
  for (int i = shape_len_ - 1; i > 0; i--) {
    stride_[i - 1] = _acc * shape_[i];
    _acc *= shape_[i];
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
    case kManual: break;
    default:
      MLLM_WARN("When trying to free TensorImpl, found invalid mem_type_. Mllm will still trying "
                "to free this tensor, but may lead to memory error.");
      MllmEngineCtx::instance().mem()->free(this);
      break;
  };
}

std::shared_ptr<PTTensorImpl> TensorImpl::toPerTensorImpl() {
  return std::static_pointer_cast<PTTensorImpl>(shared_from_this());
}

std::shared_ptr<PCTensorImpl> TensorImpl::toPerChannelImpl() {
  return std::static_pointer_cast<PCTensorImpl>(shared_from_this());
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

void* TensorImpl::rptr() const { return data_; }

void TensorImpl::_setRawPtr(void* ptr) { data_ = ptr; }

size_t TensorImpl::size() const {
  size_t acc = 1;
  for (int i = 0; i < shape_len_; i++) { acc *= shape_[i]; }

  acc = (size_t)((float)acc * dataTypeSize(dtype_));

  return acc;
}

size_t TensorImpl::elementSize() const {
  size_t acc = 1;
  for (int i = 0; i < shape_len_; i++) { acc *= shape_[i]; }
  return acc;
}

std::vector<size_t> TensorImpl::shape() const { return {shape_, shape_ + shape_len_}; }

}  // namespace mllm
