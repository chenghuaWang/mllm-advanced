/**
 * @file SiLUOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {
struct SiLUOpCargo : public BaseOpCargo<SiLUOpCargo> {};

class SiLUOp : public BaseOp {
 public:
  explicit SiLUOp(const SiLUOpCargo& cargo);

  void load(std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  SiLUOpCargo cargo_;
};

}  // namespace mllm
