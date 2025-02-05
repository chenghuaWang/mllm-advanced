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

LayerImpl::LayerImpl() : HierarchyBase(HierarchyTypes::kLayer) {}

void LayerImpl::dump(DumpPrinter& printer) {
  printer.print("{}, device={}", absoluteName(), deviceTypes2Str(device()));
}

std::unordered_map<std::string, std::shared_ptr<TensorImpl>>& LayerImpl::refParams() {
  return parameter_loader_->params();
}

void LayerImpl::load(std::shared_ptr<ParameterLoader>& ploader) {
  parameter_loader_ = ploader;
  MllmEngineCtx::instance().thisThread()->layer_ops_table[absoluteName()]->load(ploader);
}

std::shared_ptr<LayerImpl> Layer::impl() const { return impl_; }

void Layer::print() {
  auto p = DumpPrinter();
  impl_->dump(p);
}

OpType Layer::opType() const { return op_type_; }

BaseOpCargoBase& Layer::refCargo() { return cargo_; }

}  // namespace mllm::nn
