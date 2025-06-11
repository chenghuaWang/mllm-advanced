/**
 * @file QnnCompiledObj.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <fstream>
#include <memory>

#include "mllm/Core/Tensor.hpp"
#include "mllm/Backends/QNN/QnnAllocator.hpp"
#include "mllm/Backends/QNN/Runtime/QnnIRGraph.hpp"

namespace mllm::qnn {

class QnnCompiledObj {
 public:
  explicit QnnCompiledObj(const std::shared_ptr<QnnIRGraph>& qnn_ir_graph,
                          const std::shared_ptr<QnnAllocator>& allocator);

  bool allocRuntime();

  bool freeRuntime();

  // There is no need to give in any inputs and outputs tensors. You should call allocRuntime to
  // alloc shared buffer inputs and outputs runtime for both mllm arm backends and qnn backends.
  bool forward();

  inline std::vector<Tensor>& getInputsTensor() { return buffered_mllm_tensor_inputs_; }

  inline std::vector<Tensor>& getOutputsTensor() { return buffered_mllm_tensor_outputs_; }

 private:
  Tensor qnnTensorDescriptorToMllmTensor(const Qnn_Tensor_t& qnn_tensor);

  std::shared_ptr<QnnAllocator> allocator_ = nullptr;
  std::vector<Tensor> buffered_mllm_tensor_inputs_;
  std::vector<Tensor> buffered_mllm_tensor_outputs_;
  std::shared_ptr<QnnIRGraph> qnn_ir_graph_ = nullptr;
};

}  // namespace mllm::qnn
