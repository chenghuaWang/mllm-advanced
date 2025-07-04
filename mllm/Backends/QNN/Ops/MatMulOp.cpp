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
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/QNN/Ops/MatMulOp.hpp"

namespace mllm::qnn {

bool QnnMatMulOpPattern::match(const ir::op_ptr_t& op) { return op->isa_<ir::linalg::MatMulOp>(); }

bool QnnMatMulOpPattern::addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
                                 const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
                                 const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) {
  // 1. Prepare inputs
  std::vector<std::string> input_names;
  input_names.reserve(inputs.size());
  for (auto& input_tensor_ir : inputs) {
    // Use ir value name instead of tensor's name !!!
    input_names.emplace_back(input_tensor_ir->name());
  }

  // 2. Prepare outputs
  // Transform output tensor ir to qnn tensor
  std::vector<Qnn_Tensor_t> output_tensors;
  for (auto& out : op->outputs()) {
    output_tensors.emplace_back(QnnTensorTransform::instance().transform(
        out->cast_<ir::tensor::TensorValue>(), QNN_TENSOR_VERSION_2));
  }

  // 3. Get mllm's qnn matmul op
  auto mllm_matmul_op = (QnnMatMulOp*)(op->cast_<ir::linalg::MatMulOp>()->getAOp());

  // 4. Make Param_t for transpose
  std::vector<Qnn_Param_t*> params;
  if (mllm_matmul_op->transposeA()) {
    auto param_transpose_in0 =
        new Qnn_Param_t{.paramType = QNN_PARAMTYPE_SCALAR,
                        .name = "transpose_in0",
                        .scalarParam = Qnn_Scalar_t{QNN_DATATYPE_BOOL_8, {.bool8Value = true}}};
    params.emplace_back(param_transpose_in0);
  }
  if (mllm_matmul_op->transposeB()) {
    auto param_transpose_in1 =
        new Qnn_Param_t{.paramType = QNN_PARAMTYPE_SCALAR,
                        .name = "transpose_in1",
                        .scalarParam = Qnn_Scalar_t{QNN_DATATYPE_BOOL_8, {.bool8Value = true}}};
    params.emplace_back(param_transpose_in1);
  }

  // 5. Add Node to Qnn Graph.
  graph.addOp(QNN_OPCONFIG_VERSION_1, mllm_matmul_op->name(), QnnIRGraph::QTI_AISW_OP_PACKAGE,
              "MatMul", params, input_names, output_tensors);

  // 6. Delete template values
  for (auto p : params) { delete p; }

  return true;
}

std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> QnnMatMulOpPattern::create() {
  return {OpType::kMatMul, std::make_shared<QnnMatMulOpPattern>()};
}

QnnMatMulOp::QnnMatMulOp(const MatMulOpCargo& cargo) : MatMulOp(cargo) {}

void QnnMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void QnnMatMulOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}
}  // namespace mllm::qnn
