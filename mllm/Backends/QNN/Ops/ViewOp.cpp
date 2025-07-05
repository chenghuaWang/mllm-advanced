/**
 * @file ViewOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/QNN/Ops/ViewOp.hpp"

namespace mllm::qnn {

bool QnnViewOpPattern::match(const ir::op_ptr_t& op) { return op->isa_<ir::linalg::ViewOp>(); }

bool QnnViewOpPattern::addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
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
  auto mllm_view_op = (ViewOp*)(op->cast_<ir::linalg::ViewOp>()->getAOp());

  // Reshape Op.
  graph.addOp(QNN_OPCONFIG_VERSION_1, mllm_view_op->name(), QnnIRGraph::QTI_AISW_OP_PACKAGE,
              "Reshape", {}, input_names, output_tensors);

  return true;
}

std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> QnnViewOpPattern::create() {
  return {OpType::kView, std::make_shared<QnnViewOpPattern>()};
}

QnnViewOp::QnnViewOp(const ViewOpCargo& cargo) : ViewOp(cargo) {}

void QnnViewOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void QnnViewOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Qnn can not handle shallow copy, so we create a new node for output, just like SSA does.
  const auto& it = inputs[0];
  auto const& new_shape = cargo_.to_shape_;

  outputs.emplace_back(Tensor::empty(new_shape, it.dtype(), it.device()));
  MLLM_RT_ASSERT_EQ(it.numel(), outputs[0].numel());
  outputs[0].setMemType(it.memType());
}

void QnnViewOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::qnn