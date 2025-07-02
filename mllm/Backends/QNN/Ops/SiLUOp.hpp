/**
 * @file SiLUOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/SiLUOp.hpp"
#include "mllm/Backends/QNN/Ops/QnnBaseOp.hpp"

// QNN Documents:
// HardSwish
// The hard swish operation computes:

// out[0] = in[0] * max(0, min(6, (x + 3))) / 6

namespace mllm::qnn {

class QnnSiLUOpPattern final : public QnnBaseOpPattern {
 public:
  bool match(const ir::op_ptr_t& op) override;

  bool addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
               const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) override;

  static std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> create();
};

class QnnSiLUOp final : public SiLUOp {
 public:
  explicit QnnSiLUOp(const SiLUOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  // NOTE: There is no need to override `reshape` function.

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QnnSiLUOpFactory : public TypedOpFactory<OpType::kSiLU, SiLUOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const SiLUOpCargo& cargo) override {
    return std::make_shared<QnnSiLUOp>(cargo);
  }
};

}  // namespace mllm::qnn
