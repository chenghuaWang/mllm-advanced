/**
 * @file VisionRoPEOp.hpp
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

enum class VisionRoPEOpCargoType : uint8_t {
  kStart = 0,
  kQwen2VL,
  kEnd,
};

struct Qwen2VLRoPEOpCargo {
  int32_t dims;
  int32_t spatial_merge_size = 2;
  float theta;
};

struct VisionRoPEOpCargo : public BaseOpCargo<VisionRoPEOpCargo> {
  VisionRoPEOpCargoType type;
  union {
    Qwen2VLRoPEOpCargo qwen2vl_rope_op_cargo;
  };
};

class VisionRoPEOp : public BaseOp {
 public:
  explicit VisionRoPEOp(const VisionRoPEOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  VisionRoPEOpCargo cargo_;
};

}  // namespace mllm
