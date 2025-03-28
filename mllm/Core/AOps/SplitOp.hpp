/**
 * @file SplitOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

struct SplitOpCargo : public BaseOpCargo<SplitOpCargo> {
  int32_t dim_;

  // if split_size_or_sections_.size() is 1, use split size. else split to sections.
  std::vector<int32_t> split_size_or_sections_;
};

class SplitOp : public BaseOp {
 public:
  explicit SplitOp(const SplitOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  SplitOpCargo cargo_;
};

}  // namespace mllm
