/**
 * @file Module.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/Tensor.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Engine/ParameterReader.hpp"
#include "mllm/Nn/HierarchyBase.hpp"
#include "mllm/Nn/Layer.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Utils/DumpPrinter.hpp"
#include <memory>
#include <vector>
#include <string>

namespace mllm::nn {

class ModuleImpl : public HierarchyBase {
 public:
  ModuleImpl();

  void regHierarchy(std::shared_ptr<HierarchyBase> hb);

  void dump(DumpPrinter& printer);

  std::vector<std::shared_ptr<HierarchyBase>>& hierarchies();

  void load(std::shared_ptr<ParameterLoader>& ploader);

  std::shared_ptr<ParameterLoader> params() const;

 private:
  std::shared_ptr<ParameterLoader> param_loader_;
  std::vector<std::shared_ptr<HierarchyBase>> reg_hierarchies_;
};

template<typename T>
class ModuleLists;

class Module {
 public:
  Module();

  void selfAssignName(const std::string& name);

  std::shared_ptr<ModuleImpl> impl();

  template<typename T, typename... Args>
  auto reg(const std::string& name, Args&&... args) {
    if constexpr (std::is_base_of_v<Module, T>) {
      auto ret = T(impl_->absoluteName() + "." + name, std::forward<Args>(args)...);
      impl_->regHierarchy(ret.impl());

      return ret;
    }

    // register to thisThread table.
    if constexpr (std::is_base_of_v<Layer, T>) {
      auto ret = T(std::forward<Args>(args)...);
      impl_->regHierarchy(ret.impl());
      ret.impl()->setAbsoluteName(impl_->absoluteName() + "." + name);

      auto& ctx = MllmEngineCtx::instance();
      auto _op = ctx.getBackend(ret.impl()->device())->createOp(ret.opType(), ret.refCargo());
      _op->setName(ret.impl()->absoluteName());
      ctx.thisThread()->layer_ops_table.reg(ret.impl()->absoluteName(), _op);

      return ret;
    }
  }

  template<typename... Args>
  std::vector<Tensor> operator()(Args&&... args) {
    std::vector<Tensor> inputs = {std::forward<Args>(args)...};

    // check inputs first
    for (auto& t : inputs) {
      if (t.memType() == TensorMemTypes::kExtraInput && t.name().empty()) {
        MLLM_ERROR_EXIT(kError,
                        "The inputs to nn::module is `ExtraInput` but its name is empty. You may "
                        "need to use t.setName().setMemType() when initializing the tensor.");
      }
    }

    return forward(inputs);
  }

  void print();

  Module& load(std::shared_ptr<ParameterLoader>& ploader);

  virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;

  Tensor params(const std::string& name) { return Tensor(impl_->params()->operator[](name)); }

 private:
  std::shared_ptr<ModuleImpl> impl_ = nullptr;
};

template<typename T>
class ModuleList final : public Module {
  std::vector<T> layers_;

 public:
  ModuleList() = default;

  template<typename... Args>
  ModuleList(const std::string& name, int nums, Args&&... args) {
    selfAssignName(name);
    for (int i = 0; i < nums; ++i) {
      layers_.emplace_back(
          reg<T>(/*name*/ std::to_string(i), /*args*/ std::forward<Args>(args)...));
    }
  };

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    std::vector<Tensor> o = inputs;
    for (auto& layer : layers_) { o = layer.forward(o); }
    return o;
  }

  std::vector<T>& getList() { return layers_; }
};

}  // namespace mllm::nn
