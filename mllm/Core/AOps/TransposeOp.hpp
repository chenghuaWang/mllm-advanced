/**
 * @file TransposeOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstddef>
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

struct TransposeOpCargo : public BaseOpCargo<TransposeOpCargo> {
  size_t transpose_dim_x;
  size_t transpose_dim_y;
};

class TransposeOp : public BaseOp {
 public:
  explicit TransposeOp(const TransposeOpCargo& cargo);

  void load(std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  TransposeOpCargo cargo_;
};

}  // namespace mllm
