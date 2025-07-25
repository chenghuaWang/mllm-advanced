/**
 * @file MatMulOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/MatMulOp.hpp"
#include "mllm/Backends/QNN/Ops/QnnBaseOp.hpp"

namespace mllm::qnn {

class QnnMatMulOpPattern final : public QnnBaseOpPattern {
 public:
  bool match(const ir::op_ptr_t& op) override;

  bool addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
               const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) override;

  static std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> create();
};

class QnnMatMulOp final : public MatMulOp {
 public:
  explicit QnnMatMulOp(const MatMulOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  // NOTE: There is no need to override `reshape` function.

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QnnMatMulOpFactory : public TypedOpFactory<OpType::kMatMul, MatMulOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const MatMulOpCargo& cargo) override {
    return std::make_shared<QnnMatMulOp>(cargo);
  }
};

}  // namespace mllm::qnn
