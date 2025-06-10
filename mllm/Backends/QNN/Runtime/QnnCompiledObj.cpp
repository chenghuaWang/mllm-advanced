/**
 * @file QnnCompiledObj.cpp
 * @brief
 * @version 0.1
 * @date 2025-06-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cstdint>
#include "mllm/Backends/QNN/Runtime/QnnCompiledObj.hpp"

namespace mllm::qnn {

QnnCompiledObj::QnnCompiledObj(QnnIRGraph* const qnn_ir_graph,
                               const std::shared_ptr<QnnAllocator>& allocator)
    : qnn_ir_graph_(qnn_ir_graph), allocator_(allocator) {}

bool QnnCompiledObj::allocRuntime() {
  auto& qnn_inputs_tensor_descriptor = qnn_ir_graph_->getInputs();
  for (auto& qnn_i : qnn_inputs_tensor_descriptor) {
    auto t = qnnTensorDescriptorToMllmTensor(qnn_i);
    buffered_mllm_tensor_inputs_.emplace_back(t);

    // Will let QnnAllocator to alloc ION shared memory
    t.alloc();
    // Register this alloced ION memory to Qnn context.
    allocator_->registerQnnTensorToSharedBuffer(t.ptr<uint8_t>(), qnn_i);
  }

  auto& qnn_outputs_tensor_descriptor = qnn_ir_graph_->getOutputs();
  for (auto& qnn_o : qnn_outputs_tensor_descriptor) {
    auto t = qnnTensorDescriptorToMllmTensor(qnn_o);
    buffered_mllm_tensor_outputs_.emplace_back(t);

    // Will let QnnAllocator to alloc ION shared memory
    t.alloc();
    // Register this alloced ION memory to Qnn context.
    allocator_->registerQnnTensorToSharedBuffer(t.ptr<uint8_t>(), qnn_o);
  }

  return true;
}

bool QnnCompiledObj::freeRuntime() {
  for (auto& mllm_tensor_i : buffered_mllm_tensor_inputs_) {
    allocator_->deRegisterQnnTensorFromSharedBuffer(mllm_tensor_i.ptr<uint8_t>());
  }
  // After the buffer list freed. The memory will be freed automatically. The counter of this qnn
  // buffer should decreased to zero.
  buffered_mllm_tensor_inputs_.clear();

  for (auto& mllm_tensor_o : buffered_mllm_tensor_outputs_) {
    allocator_->deRegisterQnnTensorFromSharedBuffer(mllm_tensor_o.ptr<uint8_t>());
  }
  // After the buffer list freed. The memory will be freed automatically. The counter of this qnn
  // buffer should decreased to zero.
  buffered_mllm_tensor_outputs_.clear();
  return true;
}

bool QnnCompiledObj::forward() {
  // TODO
  return true;
}

Tensor QnnCompiledObj::qnnTensorDescriptorToMllmTensor(const Qnn_Tensor_t& qnn_tensor) {
  // TODO
  return {};
}

}  // namespace mllm::qnn
