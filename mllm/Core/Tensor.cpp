/**
 * @file Tensor.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/Tensor.hpp"
#include "mllm/Core/AOps/CastTypeOp.hpp"
#include "mllm/Core/AOps/D2HOp.hpp"
#include "mllm/Core/AOps/ElewiseOp.hpp"
#include "mllm/Core/AOps/FillOp.hpp"
#include "mllm/Core/AOps/TransposeOp.hpp"
#include "mllm/Core/AOps/ViewOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/TensorImpl.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Common.hpp"
#include <half/half.hpp>

namespace mllm {

SliceIndicesPair::SliceIndicesPair(int32_t v) : start_(v), end_(v + 1) {
  if (v == kAll) {
    start_ = kAll;
    end_ = kAll;
  }
}

SliceIndicesPair::SliceIndicesPair(int32_t start, int32_t end, int32_t step)
    : start_(start), end_(end), step_(step) {}

Tensor::Tensor(const std::shared_ptr<TensorViewImpl>& impl) : impl_(impl) {}

Tensor Tensor::empty(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(shape, storage);
  return Tensor(impl);
}

Tensor Tensor::zeros(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(shape, storage);
  MllmEngineCtx::instance().mem()->alloc(storage);
  return MllmEngineCtx::instance().dispatch(OpType::kFill, FillOpCargo{.type = 0},
                                            {Tensor(impl)})[0];
}

Tensor Tensor::ones(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(shape, storage);
  MllmEngineCtx::instance().mem()->alloc(storage);
  return MllmEngineCtx::instance().dispatch(OpType::kFill, FillOpCargo{.type = 1},
                                            {Tensor(impl)})[0];
}

Tensor& Tensor::alloc() {
  if (impl_->storage()->ptr_) {
    MLLM_WARN("Tensor already allocated. Tensor uuid is <{}> name is <{}>", impl_->uuid(),
              impl_->name());
    return *this;
  }
  MllmEngineCtx::instance().mem()->alloc(impl_->storage());
  return *this;
}

Tensor& Tensor::allocExtraTensorView(const std::string& extra_tensor_name,
                                     const std::vector<int32_t>& shape, DataTypes dtype,
                                     DeviceTypes device) {
  MLLM_RT_ASSERT_EQ(extra_tensor_view_.count(extra_tensor_name), 0);
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(shape, storage);
  extra_tensor_view_.insert({extra_tensor_name, impl});
  return *this;
}

Tensor Tensor::getExtraTensorViewInTensor(const std::string& extra_tensor_name) {
  MLLM_RT_ASSERT_EQ(extra_tensor_view_.count(extra_tensor_name), 1);
  return Tensor(extra_tensor_view_.at(extra_tensor_name));
}

Tensor Tensor::to(DeviceTypes device) {
  if (device == impl_->device()) { return *this; }
  // TODO create a function like op
  // H2D, D2H Ops
  // TODO reset device
  return *this;
}

Tensor Tensor::to(DataTypes dtype) {
  if (dtype == impl_->dtype()) { return *this; }
  return MllmEngineCtx::instance().dispatch(OpType::kCastType, CastTypeOpCargo{.to_dtype = dtype},
                                            {*this})[0];
}

Tensor Tensor::cpu() {
  if (kCPU == impl_->device()) { return *this; }
  return MllmEngineCtx::instance().dispatch(
      OpType::kD2H, D2HOpCargo{.from_device_type = impl_->device(), .to_device_type = kCPU},
      {*this})[0];
}

Tensor Tensor::cuda() {
  if (kCUDA == impl_->device()) { return *this; }
  // TODO Host 2 Device || Device to Device
  return Tensor(nullptr);
}

Tensor Tensor::operator[](const SliceIndices& slice_index) {
  if (!impl_) { return Tensor(nullptr); }

  MLLM_RT_ASSERT_EQ(slice_index.size(), shape().size());

  auto old_impl = impl_;
  auto old_storage = old_impl->storage();
  int32_t old_rank = shape().size();
  std::vector<int32_t> new_shape;
  int32_t new_storage_offset = old_impl->storageOffset();
  std::vector<int32_t> new_stride;

  for (int i = 0; i < old_rank; ++i) {
    const auto& pair = slice_index[i];
    int32_t start = pair.start_;
    int32_t end = pair.end_;
    int32_t step = pair.step_;

    if (start == kAll) { start = 0; }
    if (end == kAll) { end = shape()[i]; }

    if (start < 0) { start = start + shape()[i]; }
    if (end < 0) { end = end + shape()[i]; }

    if (step < 1) { NYI("Mllm only support step >= 1 in operator[] right now"); }

    int32_t num_elements = 0;
    if (end > start) { num_elements = (end - start + step - 1) / step; }

    new_storage_offset += start * old_impl->stride()[i];
    new_stride.push_back(old_impl->stride()[i] * step);
    new_shape.push_back(num_elements);
  }

  auto new_impl = TensorViewImpl::create(new_storage_offset, new_shape, new_stride, old_storage);

  return Tensor(new_impl);
}

Tensor Tensor::operator+(const Tensor& rhs) {
  return MllmEngineCtx::instance().dispatch(OpType::kAdd, AddOpCargo{}, {*this, rhs})[0];
}

Tensor Tensor::operator-(const Tensor& rhs) {
  return MllmEngineCtx::instance().dispatch(OpType::kSub, SubOpCargo{}, {*this, rhs})[0];
}

Tensor Tensor::operator*(const Tensor& rhs) {
  return MllmEngineCtx::instance().dispatch(OpType::kMul, MulOpCargo{}, {*this, rhs})[0];
}

Tensor Tensor::operator/(const Tensor& rhs) {
  return MllmEngineCtx::instance().dispatch(OpType::kDiv, DivOpCargo{}, {*this, rhs})[0];
}

Tensor Tensor::operator+(float rhs) {
  auto st = Tensor::empty({1}, dtype(), device()).alloc();
  switch (dtype()) {
    case kFp32: *(st.ptr<float>()) = rhs; break;
    case kFp16: *(st.ptr<half_float::half>()) = half_float::half(rhs); break;
    default: NYI("Type is not supported"); break;
  }
  return *this + st;
}

Tensor Tensor::operator-(float rhs) {
  auto st = Tensor::empty({1}, dtype(), device()).alloc();
  switch (dtype()) {
    case kFp32: *(st.ptr<float>()) = rhs; break;
    case kFp16: *(st.ptr<half_float::half>()) = half_float::half(rhs); break;
    default: NYI("Type is not supported"); break;
  }
  return *this - st;
}

Tensor Tensor::operator*(float rhs) {
  auto st = Tensor::empty({1}, dtype(), device()).alloc();
  switch (dtype()) {
    case kFp32: *(st.ptr<float>()) = rhs; break;
    case kFp16: *(st.ptr<half_float::half>()) = half_float::half(rhs); break;
    default: NYI("Type is not supported"); break;
  }
  return *this * st;
}

Tensor Tensor::operator/(float rhs) {
  auto st = Tensor::empty({1}, dtype(), device()).alloc();
  switch (dtype()) {
    case kFp32: *(st.ptr<float>()) = rhs; break;
    case kFp16: *(st.ptr<half_float::half>()) = half_float::half(rhs); break;
    default: NYI("Type is not supported"); break;
  }
  return *this / st;
}

Tensor Tensor::transpose(int dim0, int dim1) {
  auto shape_size = impl_->shape().size();
  MLLM_RT_ASSERT(dim0 < shape_size);
  MLLM_RT_ASSERT(dim1 < shape_size);

  return MllmEngineCtx::instance().dispatch(
      OpType::kTranspose,
      TransposeOpCargo{.transpose_dim_x = (size_t)dim0, .transpose_dim_y = (size_t)dim1},
      {*this})[0];
}

std::string Tensor::name() const { return impl_->name(); }

TensorMemTypes Tensor::memType() const { return impl_->memType(); }

Tensor& Tensor::setName(const std::string& name) {
  impl_->storage()->name_ = name;
  return *this;
}

Tensor& Tensor::setMemType(TensorMemTypes mem_type) {
  if (impl_->storage()->mem_type_ != kNormal) {
    MLLM_WARN("You are trying to change a tensor storage whose memory type is not normal. Which "
              "may lead to memory error. Mllm will still change its memory type, but not guarantee "
              "the correctness");
  }
  impl_->storage()->mem_type_ = mem_type;
  return *this;
}

DataTypes Tensor::dtype() const { return impl_->dtype(); }

DeviceTypes Tensor::device() const { return impl_->device(); }

std::vector<int32_t> Tensor::shape() const { return impl_->shape(); }

size_t Tensor::numel() const { return impl_->numel(); }

uint32_t Tensor::uuid() const { return impl_->uuid(); }

bool Tensor::isContiguous() const { return impl_->isContiguous(); }

Tensor Tensor::contiguous() {
  if (isContiguous()) { return *this; }
  return MllmEngineCtx::instance().dispatch(OpType::kFill, FillOpCargo{.type = 5}, {*this})[0];
}

Tensor Tensor::reshape(const std::vector<int>& shape) {
  // TODO
  return Tensor(nullptr);
}

Tensor Tensor::view(const std::vector<int>& indicies) {
  return MllmEngineCtx::instance().dispatch(OpType::kView, ViewOpCargo{.to_shape_ = indicies},
                                            {*this})[0];
}

char* Tensor::offsettedRawPtr(const std::vector<int32_t>& offsets) {
  return impl_->offsettedRawPtr(offsets);
}

Affine::Affine(const std::string& sym_exp_str, std::unordered_map<std::string, float>& co)
    : expr_(sym_exp_str), co_(co) {}

Affine AffinePrimitives::create(const std::string& sym_exp_str) { return {sym_exp_str, co_}; }

// extern template class TiledTensor to reduce binary size and compile time.
EXTERN_TEMPLATE_TILED_TENSOR_IMPL(float)
EXTERN_TEMPLATE_TILED_TENSOR_IMPL(half_float::half)
EXTERN_TEMPLATE_TILED_TENSOR_IMPL(int8_t)
EXTERN_TEMPLATE_TILED_TENSOR_IMPL(int16_t)

}  // namespace mllm