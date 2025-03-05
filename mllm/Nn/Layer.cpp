/**
 * @file Layer.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Layer.hpp"
#include "mllm/Engine/Context.hpp"

namespace mllm::nn {

void LayerImpl::dump(DumpPrinter& printer) {
  printer.print("{}, device={}", absoluteName(), deviceTypes2Str(device()));
}

std::unordered_map<std::string, std::shared_ptr<TensorImpl>>& LayerImpl::refParams() {
  return parameter_loader_->params();
}

void LayerImpl::load(const std::shared_ptr<ParameterLoader>& ploader) {
  parameter_loader_ = ploader;
  MllmEngineCtx::instance().thisThread()->layer_ops_table[absoluteName()]->load(ploader);
}

void LayerImpl::to(DeviceTypes device_type) {
  auto& ctx = MllmEngineCtx::instance();
  ctx.thisThread()->layer_ops_table.remove(absoluteName());
  ctx.thisThread()->layer_ops_table.reg(absoluteName(),
                                        ctx.getBackend(device_type)->createOp(op_type_, cargo_));
  ctx.thisThread()->layer_ops_table[absoluteName()]->setName(absoluteName());
  if (parameter_loader_) {
    // reload param to current op
    ctx.thisThread()->layer_ops_table[absoluteName()]->load(parameter_loader_);
  }
  device_type_ = device_type;
}

OpType LayerImpl::opType() const { return op_type_; }

BaseOpCargoBase& LayerImpl::refCargo() { return cargo_; }

std::shared_ptr<LayerImpl> Layer::impl() const { return impl_; }

void Layer::print() {
  auto p = DumpPrinter();
  impl_->dump(p);
}

OpType Layer::opType() const { return impl()->opType(); }

BaseOpCargoBase& Layer::refCargo() { return impl()->refCargo(); }

Layer& Layer::to(DeviceTypes device_type) {
  impl_->to(device_type);
  return *this;
}

}  // namespace mllm::nn
