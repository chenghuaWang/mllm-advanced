/**
 * @file ElewiseOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/ElewiseOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Utils/Log.hpp"

namespace mllm {

__MLLM_ELEWISE_OP_IMPL(AddOp);
__MLLM_ELEWISE_OP_IMPL(SubOp);
__MLLM_ELEWISE_OP_IMPL(MulOp);
__MLLM_ELEWISE_OP_IMPL(DivOp);

}  // namespace mllm
