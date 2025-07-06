/**
 * @file PermuteOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {
struct PermuteOpCargo : public BaseOpCargo<PermuteOpCargo> {
  std::vector<int32_t> permute_dims;
};

class PermuteOp : public BaseOp {
 public:
  explicit PermuteOp(const PermuteOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline std::vector<int32_t> permuteDims() const { return cargo_.permute_dims; }

 protected:
  PermuteOpCargo cargo_;
};

}  // namespace mllm