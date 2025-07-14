/**
 * @file Planning.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Planning.hpp"
#include "mllm/Core/AOps/CopyOp.hpp"

namespace mllm::nn::planning {

void copy(Tensor& dst, Tensor& src, bool side_effect) {
  (void)MllmEngineCtx::instance().dispatch(OpType::kCopy, CopyOpCargo{.side_effect_ = side_effect},
                                           {dst, src});
}

}  // namespace mllm::nn::planning
