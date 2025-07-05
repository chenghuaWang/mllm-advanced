/**
 * @file RepeatOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/QNN/Ops/RepeatOp.hpp"
#include <cstdint>

namespace mllm::qnn {

bool QnnRepeatOpPattern::match(const ir::op_ptr_t& op) { return op->isa_<ir::linalg::RepeatOp>(); }

bool QnnRepeatOpPattern::addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
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
  auto mllm_repeat_op = (QnnRepeatOp*)(op->cast_<ir::linalg::RepeatOp>()->getAOp());

  // Create multiplier tensor for param
  uint32_t multiplier_dims[5] = {static_cast<uint32_t>(inputs[0]->tensor_.shape().size())};
  auto multiplier_tensor_name = mllm_repeat_op->name() + ".multiples";
  std::vector<uint32_t> multiplier_dims_vec(multiplier_dims[0], 1);
  multiplier_dims_vec[mllm_repeat_op->dim()] *= mllm_repeat_op->multiplier();
  auto multiplier_tensor = Qnn_Tensor_t{.version = QNN_TENSOR_VERSION_2, .v2 = QNN_TENSOR_V2_INIT};
  multiplier_tensor.v2.id = 0;
  multiplier_tensor.v2.name = multiplier_tensor_name.c_str();
  multiplier_tensor.v2.type = QNN_TENSOR_TYPE_STATIC;
  multiplier_tensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  multiplier_tensor.v2.dataType = QNN_DATATYPE_UINT_32;
  multiplier_tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  multiplier_tensor.v2.rank = 1;
  multiplier_tensor.v2.dimensions = multiplier_dims;
  multiplier_tensor.v2.clientBuf = {
      .data = multiplier_dims_vec.data(),
      .dataSize = static_cast<uint32_t>(multiplier_dims[0] * sizeof(uint32_t))};

  // Create parameter
  Qnn_Param_t multiplier_param = QNN_PARAM_INIT;
  multiplier_param.paramType = QNN_PARAMTYPE_TENSOR;
  multiplier_param.name = "multiples";
  multiplier_param.tensorParam = multiplier_tensor;

  // Add Op Node to Graph.
  graph.addOp(QNN_OPCONFIG_VERSION_1, mllm_repeat_op->name(), QnnIRGraph::QTI_AISW_OP_PACKAGE,
              "Tile", {&multiplier_param}, input_names, output_tensors);

  return true;
}

std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> QnnRepeatOpPattern::create() {
  return {OpType::kRepeat, std::make_shared<QnnRepeatOpPattern>()};
}

QnnRepeatOp::QnnRepeatOp(const RepeatOpCargo& cargo) : RepeatOp(cargo) {}

void QnnRepeatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void QnnRepeatOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::qnn
