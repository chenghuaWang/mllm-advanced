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

namespace mllm::nn {

LayerImpl::LayerImpl() : HierarchyBase(HierarchyTypes::kLayer) {}

void LayerImpl::dump(DumpPrinter& printer) {
  printer.print("{}, device={}", absoluteName(), deviceTypes2Str(device()));
}

std::unordered_map<std::string, std::shared_ptr<TensorImpl>>& LayerImpl::refParams() {
  return params_;
}

Layer::Layer() { impl_ = std::make_shared<LayerImpl>(); }

std::shared_ptr<LayerImpl> Layer::impl() const { return impl_; }

void Layer::print() {
  auto p = DumpPrinter();
  impl_->dump(p);
}

}  // namespace mllm::nn
