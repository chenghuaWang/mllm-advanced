/**
 * @file ElewiseOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"

#define __MLLM_ELEWISE_OP_DEFINE(name)                                                      \
  class name : public BaseOp {                                                              \
   public:                                                                                  \
    name();                                                                                 \
    void load(const std::shared_ptr<ParameterLoader>& ploader) override;                    \
    void trace(void* trace_context, const std::vector<Tensor>& inputs,                      \
               std::vector<Tensor>& outputs) override;                                      \
    void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override; \
    void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override; \
    void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;   \
  };

// TODO I have not impl the broad cast yet.
#define __MLLM_ELEWISE_OP_IMPL(name)                                                           \
  name::name() : BaseOp(OpType::kAdd) {}                                                       \
  void name::load(const std::shared_ptr<ParameterLoader>& ploader) {}                          \
  void name::trace(void* trace_context, const std::vector<Tensor>& inputs,                     \
                   std::vector<Tensor>& outputs) {                                             \
    auto ctx = (ir::IRContext*)trace_context;                                                  \
    auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);                                \
    auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);                               \
    ctx->create<ir::linalg::name>(shared_from_this(), i_irs, o_irs);                           \
  }                                                                                            \
  void name::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {        \
    MLLM_WARN(#name "::forward is not implemented");                                           \
  }                                                                                            \
  void name::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {        \
    Tensor output_0 = Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device()); \
    outputs.emplace_back(output_0);                                                            \
  }                                                                                            \
  void name::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {          \
    for (auto& t : outputs) t.alloc();                                                         \
  }

namespace mllm {

struct AddOpCargo : public BaseOpCargo<AddOpCargo> {};

struct SubOpCargo : public BaseOpCargo<SubOpCargo> {};

struct MulOpCargo : public BaseOpCargo<MulOpCargo> {};

struct DivOpCargo : public BaseOpCargo<DivOpCargo> {};

__MLLM_ELEWISE_OP_DEFINE(AddOp);
__MLLM_ELEWISE_OP_DEFINE(SubOp);
__MLLM_ELEWISE_OP_DEFINE(MulOp);
__MLLM_ELEWISE_OP_DEFINE(DivOp);

}  // namespace mllm