/**
 * @file QnnBaseOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Ops/QnnBaseOp.hpp"

namespace mllm::qnn {

void QnnBaseOpPattern::setIRCtx(const std::shared_ptr<ir::IRContext>& ctx) { ctx_ = ctx; }

}  // namespace mllm::qnn
