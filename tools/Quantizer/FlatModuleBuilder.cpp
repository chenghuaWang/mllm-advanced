/**
 * @file FlatModuleBuilder.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/AOps/LinearOp.hpp"
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/IR/Builtin/Op.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/IR/Tensor/Op.hpp"
#include "mllm/Nn/HierarchyBase.hpp"
#include "mllm/Utils/Common.hpp"
#include "tools/Quantizer/FlatModuleBuilder.hpp"

namespace mllm::tools {
std::shared_ptr<ir::IRContext> createFlatModule(
    std::vector<std::shared_ptr<BaseOp>>& mllm_quantized_ops,
    std::shared_ptr<ParameterLoader>& param_loader, MllmModelCfg& cfg) {
  auto& mllm_ctx = MllmEngineCtx::instance();

  auto ir_ctx = std::make_shared<ir::IRContext>();
  auto ir_module =
      ir_ctx->createAndSetModuleOp<ir::ModuleOp>(ir_ctx->create<ir::SymbolAttr>("main"));

  // create call op
  auto call_op = ir_ctx->create<ir::graph::CallGraphOp>(
      ir_ctx->create<ir::SymbolAttr>("<QUANT ANONYMOUS FLAT MODULE>"));

  // create subgraph under ModuleOp
  std::shared_ptr<ir::graph::SubGraphOp> this_graph_ir = nullptr;
  {
    auto guard =
        ir::IRWriterGuard(ir_ctx, ir_ctx->topLevelOp()->cast_<ir::ModuleOp>()->getTopRegion());
    this_graph_ir = ir_ctx->create<ir::graph::SubGraphOp>(
        ir_ctx->create<ir::SymbolAttr>("<QUANT ANONYMOUS FLAT MODULE>"),
        std::make_shared<HierarchyBase>(HierarchyTypes::kModule));
  }
  this_graph_ir->setDevice(kCPU);

  {
    auto guard = ir::IRWriterGuard(ir_ctx, this_graph_ir->getTopRegion());

    // create ops and ir
    for (auto& op_name : cfg.opNames()) {
      auto op_type = cfg.opType(op_name);
      if (op_type == "Linear") {
        auto weight_tensor = param_loader->params()[op_name + ".weight"];

        // FIXME:
        // We suppose:
        // weight_tensor.shape()[0] is out_channel
        // weight_tensor.shape()[1] is in_channel
        //
        // We may need to make this more robust
        auto mllm_op = mllm_ctx.getBackend(kCPU)->createOp(
            OpType::kLinear,
            LinearOpCargo{.in_channels = weight_tensor->shape()[1],
                          .out_channels = weight_tensor->shape()[0],
                          .bias = param_loader->params().count(op_name + ".bias") ? true : false,
                          .transpose = false});
        mllm_op->setName(op_name);
        mllm_quantized_ops.emplace_back(mllm_op);
        ir_ctx->create<ir::linalg::LinearOp>(mllm_op,
                                             std::vector<ir::tensor::TensorValue::self_ptr_t>{},
                                             std::vector<ir::tensor::TensorValue::self_ptr_t>{});
      } else {
        MLLM_WARN("Unsupported op for quant: {}", op_type);
      }
    }
  }

  // Create other params in "init_params" graph.
  std::shared_ptr<ir::graph::SubGraphOp> params_graph_ir = nullptr;
  {
    auto guard =
        ir::IRWriterGuard(ir_ctx, ir_ctx->topLevelOp()->cast_<ir::ModuleOp>()->getTopRegion());
    params_graph_ir = ir_ctx->create<ir::graph::SubGraphOp>(
        ir_ctx->create<ir::SymbolAttr>("init_params"),
        std::make_shared<HierarchyBase>(HierarchyTypes::kModule));
  }
  params_graph_ir->setDevice(kCPU);

  {
    auto guard = ir::IRWriterGuard(ir_ctx, params_graph_ir->getTopRegion());
    std::vector<std::string> param_initted_by_ops = cfg.opNames();
    std::vector<std::string> param_need_to_be_initted;
    for (auto& [name, tensor] : param_loader->params()) {
      bool flag = true;
      for (auto& op_name : param_initted_by_ops) {
        if (name.find(op_name) != std::string::npos) {
          flag = false;
          break;
        }
      }
      if (flag) { param_need_to_be_initted.emplace_back(name); }
    }

    for (auto& name : param_need_to_be_initted) {
      ir_ctx->create<ir::tensor::AllocGlobalOp>(
          ir_ctx->create<ir::tensor::TensorValue>(Tensor(param_loader->params()[name])));
    }
  }

  return ir_ctx;
}
}  // namespace mllm::tools