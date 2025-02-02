/**
 * @file Layer.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "mllm/Core/TensorImpl.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Nn/HierarchyBase.hpp"
#include "mllm/Utils/DumpPrinter.hpp"

namespace mllm::nn {

class LayerImpl : public HierarchyBase {
 public:
  LayerImpl();

  void dump(DumpPrinter& printer);

  std::unordered_map<std::string, std::shared_ptr<TensorImpl>>& refParams();

 private:
  std::unordered_map<std::string, std::shared_ptr<TensorImpl>> params_;
};

class Layer {
 public:
  Layer();

  [[nodiscard]] std::shared_ptr<LayerImpl> impl() const;

  template<typename... Args>
  Tensor operator()(Args&&... args) {
    MLLM_RT_ASSERT((std::is_base_of_v<Tensor, Args> && ...));
    MLLM_RT_ASSERT((!args.name().empty() && ...));

    if (MllmEngineCtx::instance().traceMode()) {
      // TODO if pre-planing(Trace flag is set in context).
      // 1. reshape all layers first
      // 2. setup all layers.
      // return nullptr here. we create tensorimpl in reshape phase
      return Tensor(nullptr).setName(impl_->absoluteName() + ".out-0");
    }

    // TODO if fully eager mode.
    // 1. call reshape for this layer immediately.
    // 2. execute forward for this layer immediately.
    // return final tensor
  }

  void print();

  // TODO
  virtual bool load() { return true; };

 private:
  std::shared_ptr<LayerImpl> impl_;
};

}  // namespace mllm::nn