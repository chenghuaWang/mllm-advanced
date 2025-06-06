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

void QnnBaseOpPattern::transformTensorIR2QnnTensor(
    const ir::tensor::TensorValue::self_ptr_t& tensor_ir, Qnn_Tensor_t& qnn_tensor_ptr) {
  // TODO
}

}  // namespace mllm::qnn
