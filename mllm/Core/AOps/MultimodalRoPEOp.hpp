/**
 * @file MultimodalRoPEOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

enum class MultimodalRoPEOpCargoType : uint8_t {
  kStart = 0,
  kDefault,
  kQwen2VL,
  kEnd,
};

struct MultimodalRoPEOpCargo : public BaseOpCargo<MultimodalRoPEOpCargo> {
  MultimodalRoPEOpCargoType type;
  std::vector<int32_t> mrope_section;
};

class MultimodalRoPEOp : public BaseOp {
 public:
  explicit MultimodalRoPEOp(const MultimodalRoPEOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  MultimodalRoPEOpCargo cargo_;
};

}  // namespace mllm