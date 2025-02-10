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
#include "mllm/Core/AOps/ElewiseOp.hpp"
#include "mllm/Core/AOps/FillOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Log.hpp"

namespace mllm {

Tensor::Tensor(const std::shared_ptr<TensorImpl>& impl) : impl_(impl) {}

Tensor Tensor::empty(const std::vector<size_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto impl = std::make_shared<TensorImpl>(shape, dtype, device);
  return Tensor(impl);
}

Tensor Tensor::ones(const std::vector<size_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto impl = std::make_shared<TensorImpl>(shape, dtype, device);
  MllmEngineCtx::instance().mem()->alloc(impl);
  return MllmEngineCtx::instance().dispatch(OpType::kFill, FillOpCargo{.type = 1},
                                            {Tensor(impl)})[0];
}

Tensor& Tensor::alloc() {
  if (impl_->rptr()) {
    MLLM_WARN("Tensor already allocated. Tensor uuid is <{}> name is <{}>", impl_->uuid(),
              impl_->name());
    return *this;
  }
  MllmEngineCtx::instance().mem()->alloc(impl_);
  return *this;
}

Tensor& Tensor::to(DeviceTypes device) {
  if (device == impl_->device()) { return *this; }
  // TODO create a function like op
  // H2D, D2H Ops
  // TODO reset device
  return *this;
}

Tensor& Tensor::to(DataTypes dtype) {
  if (dtype == impl_->dtype()) { return *this; }
  // TODO create a function like op
  // data casting ops
  // TODO reset dtype
  return *this;
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

std::string Tensor::name() const { return impl_->name(); }

TensorMemTypes Tensor::memType() const { return impl_->memType(); }

Tensor& Tensor::setName(const std::string& name) {
  impl_->setName(name);
  return *this;
}

Tensor& Tensor::setMemType(TensorMemTypes mem_type) {
  impl_->setMemType(mem_type);
  return *this;
}

DataTypes Tensor::dtype() const { return impl_->dtype(); }

DeviceTypes Tensor::device() const { return impl_->device(); }

std::vector<size_t> Tensor::shape() const { return impl_->shape(); }

size_t Tensor::elementSize() const { return impl_->elementSize(); }

uint32_t Tensor::uuid() const { return impl_->uuid(); }

}  // namespace mllm