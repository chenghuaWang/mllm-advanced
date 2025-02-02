/**
 * @file LinearOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstddef>
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {

// When creating layers/ops. It's recommend to use OpCargo if you want to specific the
// inputs/outputs' dtype.
//
// auto x =
//     LinearOpCargo{
//         .in_channels = 1024,
//         .out_channels = 1024,
//         .bias = true,
//         .transpose = false,
//     }
//         .setInputsDtype(LinearOpCargo::InputsPos::kInput, kInt8)
//         .setInputsDtype(LinearOpCargo::InputsPos::kWeight, kInt8)
//         .setOutputsDtype(LinearOpCargo::OutputsPos::kOutput, kInt8);
struct LinearOpCargo : public BaseOpCargo<LinearOpCargo> {
  enum InputsPos : size_t {
    kInput = 0,
    kWeight = 1,
    kBias = 2,
  };

  enum OutputsPos : size_t {
    kOutput = 0,
  };

  int32_t in_channels = 0;
  int32_t out_channels = 0;
  bool bias = true;
  bool transpose = false;
};

class LinearOp : public BaseOp {
 public:
 private:
  LinearOpCargo cargo_;
};

}  // namespace mllm
