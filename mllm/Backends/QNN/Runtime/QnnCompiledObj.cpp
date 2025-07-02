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
#include "mllm/Backends/QNN/QnnTensorHelpMacros.hpp"
#include "mllm/Backends/QNN/Runtime/QnnCompiledObj.hpp"
#include "mllm/Utils/Dbg.hpp"

namespace mllm::qnn {

QnnCompiledObj::QnnCompiledObj(const std::shared_ptr<QnnIRGraph>& qnn_ir_graph,
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
  auto status = qnn_ir_graph_->qnnFuncSymbols().qnn_interface_.graphExecute(
      qnn_ir_graph_->qnnGraphHandle(), qnn_ir_graph_->getInputs().data(),
      qnn_ir_graph_->getInputs().size(), qnn_ir_graph_->getOutputs().data(),
      qnn_ir_graph_->getOutputs().size(), qnn_ir_graph_->qnnBackendDevice().profile_bk_handle_,
      nullptr);

  MLLM_RT_ASSERT_EQ(QNN_GRAPH_NO_ERROR, status);

  return true;
}

Tensor QnnCompiledObj::qnnTensorDescriptorToMllmTensor(const Qnn_Tensor_t& qnn_tensor) {
  DataTypes mllm_tensor_dtype = DataTypes::kDataTypes_End;
  std::vector<int32_t> mllm_tensor_shape;

  auto rank = HELP_QNN_TENSOR_GET_RANK(qnn_tensor);
  mllm_tensor_shape.resize(rank);

  auto dim_ptr = HELP_QNN_TENSOR_GET_DIMENSIONS(qnn_tensor);
  for (int i = 0; i < rank; i++) { mllm_tensor_shape[i] = dim_ptr[i]; }

  switch (HELP_QNN_TENSOR_GET_DATA_TYPE(qnn_tensor)) {
    case QNN_DATATYPE_FLOAT_32: mllm_tensor_dtype = DataTypes::kFp32; break;
    case QNN_DATATYPE_FLOAT_16: mllm_tensor_dtype = DataTypes::kFp16; break;
    default: NYI("QNN data type not supported") break;
  }

  return Tensor::empty(mllm_tensor_shape, mllm_tensor_dtype, kQNN);
}

}  // namespace mllm::qnn
