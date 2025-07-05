/**
 * @file LinearOp.cpp
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
#include "mllm/Backends/QNN/Ops/LinearOp.hpp"

namespace mllm::qnn {

bool QnnLinearOpPattern::match(const ir::op_ptr_t& op) { return op->isa_<ir::linalg::LinearOp>(); }

bool QnnLinearOpPattern::addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
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
  auto mllm_linear_op = (QnnLinearOp*)(op->cast_<ir::linalg::LinearOp>()->getAOp());

  // Before add weight and bias, the input size is always 1.
  // pos 0: activation
  // pos 1: weight
  // pos 2: bias (optional)
  MLLM_RT_ASSERT_EQ(input_names.size(), 1);

  // Verify the activation's last dim is in_channel
  MLLM_RT_ASSERT_EQ(inputs[0]->tensor_.shape().back(), mllm_linear_op->cargo().in_channels);

  // Verify the weight is [out_channels, in_channels]
  MLLM_RT_ASSERT_EQ(mllm_linear_op->weight().shape()[0], mllm_linear_op->cargo().out_channels);
  MLLM_RT_ASSERT_EQ(mllm_linear_op->weight().shape()[1], mllm_linear_op->cargo().in_channels);

  // Verify the bias is [out_channels]
  if (mllm_linear_op->cargo().bias) {
    MLLM_RT_ASSERT_EQ(mllm_linear_op->bias().shape().size(), 1);
    MLLM_RT_ASSERT_EQ(mllm_linear_op->bias().shape()[0], mllm_linear_op->cargo().out_channels);
  }

  // Add weight and bias to graph's tensor list
  auto qnn_weight_tensor_ir =
      ctx_->lookupSymbolTable(mllm_linear_op->weight().name())->cast_<ir::tensor::TensorValue>();
  auto qnn_weight_tensor =
      QnnTensorTransform::instance().transform(qnn_weight_tensor_ir, QNN_TENSOR_VERSION_2);
  graph.addTensor(qnn_weight_tensor_ir->name(), qnn_weight_tensor);
  input_names.emplace_back(mllm_linear_op->weight().name());

  // Add weight and bias to input_names list.
  if (mllm_linear_op->cargo().bias) {
    auto qnn_bias_tensor_ir =
        ctx_->lookupSymbolTable(mllm_linear_op->bias().name())->cast_<ir::tensor::TensorValue>();
    auto qnn_bias_tensor =
        QnnTensorTransform::instance().transform(qnn_bias_tensor_ir, QNN_TENSOR_VERSION_2);
    graph.addTensor(qnn_bias_tensor_ir->name(), qnn_bias_tensor);
    input_names.emplace_back(mllm_linear_op->bias().name());
  }

  graph.addOp(QNN_OPCONFIG_VERSION_1, mllm_linear_op->name(), QnnIRGraph::QTI_AISW_OP_PACKAGE,
              "FullyConnected", {}, input_names, output_tensors);

  return true;
}

std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> QnnLinearOpPattern::create() {
  return {OpType::kLinear, std::make_shared<QnnLinearOpPattern>()};
}

QnnLinearOp::QnnLinearOp(const LinearOpCargo& cargo) : LinearOp(cargo) {}

void QnnLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void QnnLinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}
}  // namespace mllm::qnn
