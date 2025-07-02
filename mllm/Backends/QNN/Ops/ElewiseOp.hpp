/**
 * @file ElewiseOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/ElewiseOp.hpp"
#include "mllm/Backends/QNN/Ops/QnnBaseOp.hpp"

// QNN Documents
//
// Qnn will handle Broadcast automatically

#define __MLLM_QNN_ELEWISE_OP_PATTERN_DEFINE(op_pattern_name)                               \
  class op_pattern_name final : public QnnBaseOpPattern {                                   \
   public:                                                                                  \
    bool match(const ir::op_ptr_t& op) override;                                            \
    bool addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,                                 \
                 const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,            \
                 const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) override; \
    static std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> create();                   \
  };

#define __MLLM_QNN_ELEWISE_OP_PATTERN_IMPL(op_pattern_name, qnn_op_name, base_op_name,             \
                                           qnn_package_op_name, op_type)                           \
  bool op_pattern_name::match(const ir::op_ptr_t& op) {                                            \
    return op->isa_<ir::linalg::base_op_name>();                                                   \
  }                                                                                                \
  bool op_pattern_name::addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,                         \
                                const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,    \
                                const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) { \
    std::vector<std::string> input_names;                                                          \
    input_names.reserve(inputs.size());                                                            \
    for (auto& input_tensor_ir : inputs) { input_names.emplace_back(input_tensor_ir->name()); }    \
    std::vector<Qnn_Tensor_t> output_tensors;                                                      \
    output_tensors.reserve(outputs.size());                                                        \
    for (auto& out : op->outputs()) {                                                              \
      output_tensors.emplace_back(QnnTensorTransform::instance().transform(                        \
          out->cast_<ir::tensor::TensorValue>(), QNN_TENSOR_VERSION_2));                           \
    }                                                                                              \
    auto mllm_elewise_op = (qnn_op_name*)(op->cast_<ir::linalg::base_op_name>()->getAOp());        \
    graph.addOp(QNN_OPCONFIG_VERSION_1, mllm_elewise_op->name(), QnnIRGraph::QTI_AISW_OP_PACKAGE,  \
                qnn_package_op_name, {}, input_names, output_tensors);                             \
    return true;                                                                                   \
  }                                                                                                \
  std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> op_pattern_name::create() {                 \
    return {OpType::op_type, std::make_shared<op_pattern_name>()};                                 \
  }

#define __MLLM_QNN_ELEWISE_OP_DEFINE(op_pattern_name, base_op_name, op_type, base_op_cargo_name) \
  class op_pattern_name final : public base_op_name {                                            \
   public:                                                                                       \
    explicit op_pattern_name(const base_op_cargo_name& cargo);                                   \
    void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;      \
    void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;        \
  };                                                                                             \
  class op_pattern_name##Factory : public TypedOpFactory<OpType::op_type, base_op_cargo_name> {  \
   public:                                                                                       \
    std::shared_ptr<BaseOp> createOpImpl(const base_op_cargo_name& cargo) override {             \
      return std::make_shared<op_pattern_name>(cargo);                                           \
    }                                                                                            \
  };

#define __MLLM_QNN_ELEWISE_OP_IMPL(qnn_op_name, base_op_name)                                  \
  qnn_op_name::qnn_op_name(const base_op_name##Cargo& cargo) : base_op_name() {}               \
  void qnn_op_name::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { \
    MLLM_EMPTY_SCOPE;                                                                          \
  }                                                                                            \
  void qnn_op_name::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {   \
    MLLM_EMPTY_SCOPE;                                                                          \
  }

namespace mllm::qnn {

__MLLM_QNN_ELEWISE_OP_PATTERN_DEFINE(QnnAddOpPattern);
__MLLM_QNN_ELEWISE_OP_PATTERN_DEFINE(QnnSubOpPattern);
__MLLM_QNN_ELEWISE_OP_PATTERN_DEFINE(QnnMulOpPattern);
__MLLM_QNN_ELEWISE_OP_PATTERN_DEFINE(QnnDivOpPattern);

__MLLM_QNN_ELEWISE_OP_DEFINE(QnnAddOp, AddOp, kAdd, AddOpCargo);
__MLLM_QNN_ELEWISE_OP_DEFINE(QnnSubOp, SubOp, kSub, SubOpCargo);
__MLLM_QNN_ELEWISE_OP_DEFINE(QnnMulOp, MulOp, kMul, MulOpCargo);
__MLLM_QNN_ELEWISE_OP_DEFINE(QnnDivOp, DivOp, kDiv, DivOpCargo);

}  // namespace mllm::qnn
