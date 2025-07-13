/**
 * @file QnnTensorTransform.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-06-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// This file impl a helper class for create Qnn_Tensor_t from mllm::ir::tensor::TensorValue
// This class also help mllm context to handle memory's from Qnn_Tensor_t, such as:
// name: char ptr
// dimension: uint32_t ptr
// isDynamic: uint32_t ptr
// etc...
//
// Also, this class handle all version specific details of Qnn_Tensor_t. Such as v1, v2...

#include <vector>
#include <unordered_map>

#include "mllm/IR/Tensor/Value.hpp"

#include <QNN/QnnTypes.h>
#include <QNN/QnnTensor.h>

namespace mllm::qnn {

class QnnTensorTransform {
  struct QnnTensorTransformMetaInfo {
    Qnn_Tensor_t qnn_tensor_;
    std::vector<void*> anonymous_trash_;
  };

 public:
  static QnnTensorTransform& instance() {
    static QnnTensorTransform instance;
    return instance;
  }

  ~QnnTensorTransform();

  QnnTensorTransform() = default;

  QnnTensorTransform(const QnnTensorTransform&) = delete;

  QnnTensorTransform& operator=(const QnnTensorTransform&) = delete;

  Qnn_Tensor_t transform(const ir::tensor::TensorValue::self_ptr_t& tensor_ir,
                         Qnn_TensorVersion_t version);

  Qnn_Tensor_t transform(Tensor& mllm_tensor, Qnn_TensorVersion_t version);

  Qnn_Tensor_t deepCopy(Qnn_Tensor_t* src_tensor);

 private:
  Qnn_Tensor_t transformV1(const ir::tensor::TensorValue::self_ptr_t& tensor_ir);

  Qnn_Tensor_t transformV2(const ir::tensor::TensorValue::self_ptr_t& tensor_ir);

  Qnn_Tensor_t transformV1(Tensor& mllm_tensor);

  Qnn_Tensor_t transformV2(Tensor& mllm_tensor);

  Qnn_TensorType_t autoQnnTensorType(const ir::tensor::TensorValue::self_ptr_t& tensor_ir);

  Qnn_TensorDataFormat_t autoQnnTensorDataFormat(
      const ir::tensor::TensorValue::self_ptr_t& tensor_ir);

  Qnn_DataType_t autoQnnTensorDataType(const ir::tensor::TensorValue::self_ptr_t& tensor_ir);

  Qnn_QuantizeParams_t autoQnnTensorQuantParams(
      const ir::tensor::TensorValue::self_ptr_t& tensor_ir);

  Qnn_TensorMemType_t autoQnnTensorMemType(const ir::tensor::TensorValue::self_ptr_t& tensor_ir);

  Qnn_TensorType_t autoQnnTensorType(Tensor& mllm_tensor);

  Qnn_TensorDataFormat_t autoQnnTensorDataFormat(Tensor& mllm_tensor);

  Qnn_DataType_t autoQnnTensorDataType(Tensor& mllm_tensor);

  Qnn_QuantizeParams_t autoQnnTensorQuantParams(Tensor& mllm_tensor);

  Qnn_TensorMemType_t autoQnnTensorMemType(Tensor& mllm_tensor);

  std::vector<QnnTensorTransformMetaInfo> qnn_tensors_;
  static const std::unordered_map<Qnn_DataType_t, size_t> dtype_to_size_;
};

}  // namespace mllm::qnn