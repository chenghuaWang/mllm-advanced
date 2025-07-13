/**
 * @file SiLUOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/QNN/Ops/SiLUOp.hpp"

namespace mllm::qnn {

bool QnnSiLUOpPattern::match(const ir::op_ptr_t& op) { return op->isa_<ir::linalg::SiLUOp>(); }

bool QnnSiLUOpPattern::addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
                               const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
                               const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) {
  std::vector<std::string> input_names;
  input_names.reserve(inputs.size());
  for (auto& input_tensor_ir : inputs) {
    // Use ir value name instead of tensor's name !!!
    input_names.emplace_back(graph.checkTensorName(input_tensor_ir));
  }

  // Transform output tensor ir to qnn tensor
  std::vector<Qnn_Tensor_t> output_tensors;
  output_tensors.reserve(outputs.size());
  for (auto& out : op->outputs()) {
    output_tensors.emplace_back(QnnTensorTransform::instance().transform(
        out->cast_<ir::tensor::TensorValue>(), QNN_TENSOR_VERSION_2));
  }

  // Get mllm's qnn op and transform it to qnn's op
  auto mllm_silu_op = (QnnSiLUOp*)(op->cast_<ir::linalg::SiLUOp>()->getAOp());

  // NOTE: Approximate impl with LUT.
  graph.addOp(QNN_OPCONFIG_VERSION_1, mllm_silu_op->name(), QnnIRGraph::MLLM_QNN_OP_PACKAGE, "SiLU",
              {}, input_names, output_tensors);

  return true;
}

std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> QnnSiLUOpPattern::create() {
  return {OpType::kSiLU, std::make_shared<QnnSiLUOpPattern>()};
}

QnnSiLUOp::QnnSiLUOp(const SiLUOpCargo& cargo) : SiLUOp(cargo) {}

void QnnSiLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void QnnSiLUOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::qnn
