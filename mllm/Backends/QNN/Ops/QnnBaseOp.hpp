/**
 * @file QnnBaseOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <vector>
#include "mllm/IR/Node.hpp"
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/Backends/QNN/Runtime/QnnIRGraph.hpp"

#include <QNN/QnnTypes.h>

namespace mllm::qnn {

class QnnBaseOpPattern {
 public:
  virtual bool match(const ir::op_ptr_t& op) = 0;

  virtual bool addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
                       const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
                       const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) = 0;

  void transformTensorIR2QnnTensor(const ir::tensor::TensorValue::self_ptr_t& tensor_ir,
                                   Qnn_Tensor_t& qnn_tensor_ptr);

 private:
};

}  // namespace mllm::qnn
