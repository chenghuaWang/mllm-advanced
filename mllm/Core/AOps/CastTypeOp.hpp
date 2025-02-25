/**
 * @file CastTypeOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-25
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <cstddef>
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/DataTypes.hpp"

namespace mllm {

struct CastTypeOpCargo : public BaseOpCargo<CastTypeOpCargo> {
  DataTypes to_dtype;
};

class CastTypeOp : public BaseOp {
 public:
  explicit CastTypeOp(const CastTypeOpCargo& cargo);

  void load(std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  CastTypeOpCargo cargo_;
};

}  // namespace mllm