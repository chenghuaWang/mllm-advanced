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

#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Engine/ParameterReader.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/Nn/HierarchyBase.hpp"
#include "mllm/Nn/Layer.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Utils/DumpPrinter.hpp"
#include "mllm/IR/Node.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/CF/Op.hpp"
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

  void to(DeviceTypes device_type);

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

  Module& to(DeviceTypes device_type);

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

  template<typename... Args>
  std::vector<Tensor> trace(std::shared_ptr<ir::IRContext> ir_ctx, Args&&... args) {
    // create call graph.
    auto call_op = ir_ctx->create<ir::graph::CallGraphOp>(
        ir_ctx->create<ir::SymbolAttr>(impl_->absoluteName()));

    // create subgraph under ModuleOp
    std::shared_ptr<ir::graph::SubGraphOp> this_graph_ir = nullptr;
    {
      auto guard =
          ir::IRWriterGuard(ir_ctx, ir_ctx->topLevelOp()->cast_<ir::ModuleOp>()->getTopRegion());
      this_graph_ir = ir_ctx->create<ir::graph::SubGraphOp>(
          ir_ctx->create<ir::SymbolAttr>(impl_->absoluteName()));
    }

    // to tensor vector
    std::vector<Tensor> inputs = {std::forward<Args>(args)...};

    // wrap the inputs to tensor ir.
    std::vector<std::shared_ptr<ir::tensor::TensorValue>> inputs_ir;
    for (auto& t : inputs) { inputs_ir.emplace_back(ir_ctx->create<ir::tensor::TensorValue>(t)); }

    // link inputs to subgraph.
    for (size_t i = 0; i < inputs_ir.size(); ++i) {
      auto input_ir = inputs_ir[i];
      (*input_ir)-- > this_graph_ir;
    }

    // forward
    std::vector<Tensor> outputs;
    {
      auto guard = ir::IRWriterGuard(ir_ctx, this_graph_ir->getTopRegion());
      outputs = forward(inputs);
    }

    // wrap the outputs to tensor ir.
    std::vector<std::shared_ptr<ir::tensor::TensorValue>> outputs_ir;
    for (auto& t : outputs) { outputs_ir.emplace_back(ir_ctx->create<ir::tensor::TensorValue>(t)); }

    // link outputs to subgraph.
    for (size_t i = 0; i < outputs_ir.size(); ++i) {
      auto output_ir = outputs_ir[i];
      (*this_graph_ir)-- > output_ir;
    }

    // create return op
    {
      auto guard = ir::IRWriterGuard(ir_ctx, this_graph_ir->getTopRegion());
      std::vector<ir::val_ptr_t> vals;
      for (auto& o : outputs_ir) vals.push_back(o);
      ir_ctx->create<ir::cf::ReturnOp>(vals);
    }

    // return the outputs.
    return outputs;
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
