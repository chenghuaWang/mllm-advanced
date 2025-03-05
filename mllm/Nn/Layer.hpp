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
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/TensorImpl.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Engine/ParameterReader.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Nn/HierarchyBase.hpp"
#include "mllm/Utils/DumpPrinter.hpp"

namespace mllm::nn {

class LayerImpl : public HierarchyBase {
 public:
  template<typename T>
  LayerImpl(OpType op_type, const T& cargo)
      : HierarchyBase(HierarchyTypes::kLayer), op_type_(op_type), cargo_(cargo) {}

  void dump(DumpPrinter& printer);

  std::unordered_map<std::string, std::shared_ptr<TensorImpl>>& refParams();

  void load(std::shared_ptr<ParameterLoader>& ploader);

  void to(DeviceTypes device_type);

  [[nodiscard]] OpType opType() const;

  BaseOpCargoBase& refCargo();

 private:
  BaseOpCargoBase cargo_;
  OpType op_type_;
  std::shared_ptr<ParameterLoader> parameter_loader_;
};

class Layer {
 public:
  template<typename T>
  Layer(OpType op_type, const T& cargo) {
    impl_ = std::make_shared<LayerImpl>(op_type, cargo);
  }

  [[nodiscard]] std::shared_ptr<LayerImpl> impl() const;

  template<typename... Args>
  Tensor operator()(Args&&... args) {
    // MLLM_RT_ASSERT((std::is_same_v<Tensor, Args> && ...));

    if (MllmEngineCtx::instance().traceMode()) {
      MLLM_RT_ASSERT((!args.name().empty() && ...));
      // TODO if pre-planing(Trace flag is set in context).
      // 1. reshape all layers first
      // 2. setup all layers.
      // return nullptr here. we create tensor impl in reshape phase
      return Tensor(nullptr).setName(impl_->absoluteName() + ".out-0");
    }

    // eager mode
    auto inputs = std::vector<Tensor>{std::forward<decltype(args)>(args)...};
    return MllmEngineCtx::instance().dispatch(impl_->absoluteName(), inputs)[0];
  }

  void print();

  [[nodiscard]] OpType opType() const;

  BaseOpCargoBase& refCargo();

  Layer& to(DeviceTypes device_type);

 private:
  std::shared_ptr<LayerImpl> impl_;
};

}  // namespace mllm::nn