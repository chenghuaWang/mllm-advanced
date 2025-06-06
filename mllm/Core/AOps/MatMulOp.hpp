/**
 * @file MatMulOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

struct MatMulOpCargo : public BaseOpCargo<MatMulOpCargo> {
  bool transpose_a = false;
  bool transpose_b = false;
};

class MatMulOp : public BaseOp {
 public:
  explicit MatMulOp(const MatMulOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline bool transposeA() { return cargo_.transpose_a; }

  inline bool transposeB() { return cargo_.transpose_b; }

 protected:
  MatMulOpCargo cargo_;
};

}  // namespace mllm
