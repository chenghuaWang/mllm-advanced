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

TensorStorage::~TensorStorage() {
  switch (mem_type_) {
    case kNormal:
    case kGlobal: MllmEngineCtx::instance().mem()->free(this); break;
    case kExtraInput:
    case kExtraOutput:
    case kParams:
    case kManual: break;
    case kReference: MLLM_WARN("mem_type_ kReference is not used anymore."); break;
    default:
      MLLM_WARN(
          "When trying to free TensorStorage, found invalid mem_type_. Mllm will still trying "
          "to free this TensorStorage, but may lead to memory error.");
      MllmEngineCtx::instance().mem()->free(this);
      break;
  };
}

std::shared_ptr<TensorStorage> TensorStorage::create(const std::vector<int32_t>& shape,
                                                     DataTypes dtype, DeviceTypes device) {
  auto ret = std::make_shared<TensorStorage>();

  ret->dtype_ = dtype;
  ret->device_ = device;

  size_t cnt = 1;
  for (auto i : shape) { cnt *= i; }

  ret->size_ = cnt * dataTypeSize(dtype);
  ret->type_ = Storage::kTensor;
  ret->custom_32bit_uuid_ = MllmEngineCtx::instance().getUUID();

  return ret;
}

DataTypes TensorViewImpl::dtype() const { return storage_->dtype_; }

DeviceTypes TensorViewImpl::device() const { return storage_->device_; }

TensorViewImpl::storage_t TensorViewImpl::storage() const { return storage_; }

std::string TensorViewImpl::name() const { return storage_->name_; }

TensorMemTypes TensorViewImpl::memType() const { return storage_->mem_type_; }

uint32_t TensorViewImpl::uuid() const { return storage_->custom_32bit_uuid_; }

uint64_t TensorViewImpl::address() const { return (uint64_t)(storage_->ptr_); }

void* TensorViewImpl::rptr() const {
  return (void*)(((char*)storage_->ptr_)
                 + (size_t)(storage_offset_ * dataTypeSize(storage_->dtype_)));
}

char* TensorViewImpl::offsettedRawPtr(const std::vector<int32_t>& offsets) {
  MLLM_RT_ASSERT_EQ(offsets.size(), shape_len_);

  size_t _offset = 0;
  for (int i = 0; i < shape_len_; ++i) { _offset += offsets[i] * stride_[i]; }

  return ptr<char>() + (size_t)(_offset * dataTypeSize(storage_->dtype_));
}

size_t TensorViewImpl::size() const { return storage_->size_; }

size_t TensorViewImpl::numel() const {
  size_t cnt = 1;
  for (int i = 0; i < shape_len_; ++i) cnt *= shape_[i];
  return cnt;
}

std::vector<int32_t> TensorViewImpl::shape() const { return {shape_, shape_ + shape_len_}; }

std::vector<int32_t> TensorViewImpl::stride() const { return {stride_, stride_ + shape_len_}; }

bool TensorViewImpl::isContiguous() const {
  if (shape_len_ == 0) return true;

  // check stride
  int expected_stride = 1;
  for (int i = shape_len_ - 1; i >= 0; --i) {
    if (stride_[i] != expected_stride) { return false; }
    expected_stride *= shape_[i];
  }

  return true;
}

std::shared_ptr<TensorViewImpl> TensorViewImpl::clone() const {
  auto ret = TensorViewImpl::create();

  ret->shape_len_ = shape_len_;
  for (int i = 0; i < shape_len_; ++i) {
    ret->shape_[i] = shape_[i];
    ret->stride_[i] = stride_[i];
  }
  ret->storage_offset_ = storage_offset_;
  ret->storage_ = storage_;

  return ret;
}

int32_t TensorViewImpl::storageOffset() const { return storage_offset_; }

// create empty TensorViewImpl
std::shared_ptr<TensorViewImpl> TensorViewImpl::create() {
  return std::make_shared<TensorViewImpl>();
}

// Will automatic calculate stride for you.
std::shared_ptr<TensorViewImpl> TensorViewImpl::create(
    const std::vector<int32_t>& shape, const std::shared_ptr<TensorStorage>& storage) {
  auto ret = std::make_shared<TensorViewImpl>();
  ret->shape_len_ = shape.size();
  ret->storage_offset_ = 0;

  int _cnt = 0;
  for (unsigned int it : shape) { ret->shape_[_cnt++] = (int32_t)it; }
  int _acc = 1;
  ret->stride_[ret->shape_len_ - 1] = 1;
  for (int i = ret->shape_len_ - 1; i > 0; i--) {
    ret->stride_[i - 1] = _acc * ret->shape_[i];
    _acc *= ret->shape_[i];
  }
  ret->storage_ = storage;

  return ret;
}

std::shared_ptr<TensorViewImpl> TensorViewImpl::create(
    int32_t storage_offset, const std::vector<int32_t>& shape,
    const std::shared_ptr<TensorStorage>& storage) {
  auto ret = std::make_shared<TensorViewImpl>();
  ret->shape_len_ = shape.size();
  ret->storage_offset_ = storage_offset;

  int _cnt = 0;
  for (unsigned int it : shape) { ret->shape_[_cnt++] = (int32_t)it; }
  int _acc = 1;
  ret->stride_[ret->shape_len_ - 1] = 1;
  for (int i = ret->shape_len_ - 1; i > 0; i--) {
    ret->stride_[i - 1] = _acc * ret->shape_[i];
    _acc *= ret->shape_[i];
  }
  ret->storage_ = storage;

  return ret;
}

std::shared_ptr<TensorViewImpl> TensorViewImpl::create(
    int32_t storage_offset, const std::vector<int32_t>& shape, const std::vector<int32_t>& stride,
    const std::shared_ptr<TensorStorage>& storage) {
  auto ret = std::make_shared<TensorViewImpl>();
  ret->shape_len_ = shape.size();
  ret->storage_offset_ = storage_offset;

  int _cnt = 0;
  for (unsigned int it : shape) { ret->shape_[_cnt++] = (int32_t)it; }
  _cnt = 0;
  for (unsigned int it : stride) { ret->stride_[_cnt++] = (int32_t)it; }
  ret->storage_ = storage;

  return ret;
}

}  // namespace mllm
