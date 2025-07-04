/**
 * @file ViewOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/ViewOp.hpp"
#include "mllm/Backends/QNN/Ops/QnnBaseOp.hpp"

namespace mllm::qnn {

class QnnViewOpPattern final : public QnnBaseOpPattern {
 public:
  bool match(const ir::op_ptr_t& op) override;

  bool addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
               const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) override;

  static std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> create();
};

class QnnViewOp final : public ViewOp {
 public:
  explicit QnnViewOp(const ViewOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QnnViewOpFactory : public TypedOpFactory<OpType::kView, ViewOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const ViewOpCargo& cargo) override {
    return std::make_shared<QnnViewOp>(cargo);
  }
};

}  // namespace mllm::qnn
