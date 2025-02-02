/**
 * @file ElewiseOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

struct AddOpCargo : public BaseOpCargo<AddOpCargo> {};

struct SubOpCargo : public BaseOpCargo<SubOpCargo> {};

struct MulOpCargo : public BaseOpCargo<MulOpCargo> {};

struct DivOpCargo : public BaseOpCargo<DivOpCargo> {};

class AddOp : public BaseOp {
 public:
  AddOp();

  void load(void* params) override;

  void trace(void* trace_contex, std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class SubOp : public BaseOp {};

class MulOp : public BaseOp {};

class DivOp : public BaseOp {};

}  // namespace mllm