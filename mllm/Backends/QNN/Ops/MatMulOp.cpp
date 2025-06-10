/**
 * @file MatMulOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Backends/QNN/Ops/MatMulOp.hpp"

namespace mllm::qnn {

bool QnnMatMulOpPattern::match(const ir::op_ptr_t& op) { return op->isa_<ir::linalg::MatMulOp>(); }

bool QnnMatMulOpPattern::addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
                                 const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
                                 const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) {
  std::vector<std::string> input_names;
  input_names.reserve(inputs.size());
  for (auto& input_tensor_ir : inputs) {
    // Use ir value name instead of tensor's name !!!
    input_names.emplace_back(input_tensor_ir->name());
  }

  auto mllm_matmul_op = (QnnMatMulOp*)(op->cast_<ir::linalg::MatMulOp>()->getAOp());
  if (!mllm_matmul_op->transposeA() && !mllm_matmul_op->transposeB()) {
    std::vector<Qnn_Tensor_t> output_tensors;

    // Transform output tensor ir to qnn tensor
    for (auto& out : op->outputs()) {
      output_tensors.emplace_back(QnnTensorTransform::instance().transform(
          out->cast_<ir::tensor::TensorValue>(), QNN_TENSOR_VERSION_2));
    }

    // Add Node to Qnn Graph.
    graph.addOp(QNN_OPCONFIG_VERSION_1, mllm_matmul_op->name(), QnnIRGraph::QTI_AISW_OP_PACKAGE,
                "MatMul", {}, input_names, output_tensors);
  } else {
    NYI("Not support transpose LHS or RHS in QnnMatMulOp");
  }

  return true;
}

std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> QnnMatMulOpPattern::create() {
  return {OpType::kMatMul, std::make_shared<QnnMatMulOpPattern>()};
}

QnnMatMulOp::QnnMatMulOp(const MatMulOpCargo& cargo) : MatMulOp(cargo) {}

void QnnMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // left it empty
}

void QnnMatMulOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // left it empty
}
}  // namespace mllm::qnn
