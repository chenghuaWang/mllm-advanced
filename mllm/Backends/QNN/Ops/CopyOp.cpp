/**
 * @file CopyOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/QNN/Ops/CopyOp.hpp"

namespace mllm::qnn {

bool QnnCopyOpPattern::match(const ir::op_ptr_t& op) { return op->isa_<ir::linalg::CopyOp>(); }

bool QnnCopyOpPattern::addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
                               const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
                               const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) {
  MLLM_WARN("We didn't handle this copyop pattern yet. You should check if you really need to copy "
            "a tensor, or why IR optimization pass not erases this op");

  return false;
}

std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> QnnCopyOpPattern::create() {
  return {OpType::kSiLU, std::make_shared<QnnCopyOpPattern>()};
}

QnnCopyOp::QnnCopyOp(const CopyOpCargo& cargo) : CopyOp(cargo) {}

void QnnCopyOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void QnnCopyOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::qnn