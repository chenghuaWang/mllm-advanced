/**
 * @file ElewiseOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/QNN/Ops/ElewiseOp.hpp"

namespace mllm::qnn {

__MLLM_QNN_ELEWISE_OP_PATTERN_IMPL(QnnAddOpPattern, QnnAddOp, AddOp, "ElementWiseAdd", kAdd);
__MLLM_QNN_ELEWISE_OP_PATTERN_IMPL(QnnSubOpPattern, QnnSubOp, SubOp, "ElementWiseSubtract", kSub);
__MLLM_QNN_ELEWISE_OP_PATTERN_IMPL(QnnMulOpPattern, QnnMulOp, MulOp, "ElementWiseMultiply", kMul);
__MLLM_QNN_ELEWISE_OP_PATTERN_IMPL(QnnDivOpPattern, QnnDivOp, DivOp, "ElementWiseDivide", kDiv);

__MLLM_QNN_ELEWISE_OP_IMPL(QnnAddOp, AddOp);
__MLLM_QNN_ELEWISE_OP_IMPL(QnnSubOp, SubOp);
__MLLM_QNN_ELEWISE_OP_IMPL(QnnMulOp, MulOp);
__MLLM_QNN_ELEWISE_OP_IMPL(QnnDivOp, DivOp);

}  // namespace mllm::qnn
