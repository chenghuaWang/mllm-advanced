/**
 * @file QnnTensorHelpMacros.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

#include <QNN/QnnTypes.h>
#include "mllm/Utils/Common.hpp"

namespace mllm::qnn {

inline bool validate_tensor_version(Qnn_Tensor_t tensor) {
  if (tensor.version != QNN_TENSOR_VERSION_1 || tensor.version != QNN_TENSOR_VERSION_2) {
    MLLM_ERROR_EXIT(kError, "The mllm-advanced lib only support QNN_TENSOR_VERSION_1 and "
                            "QNN_TENSOR_VERSION_2 right now");
  }
  return true;
}

inline uint32_t get_qnn_tensor_id(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.id;
    case QNN_TENSOR_VERSION_2: return tensor->v2.id;
    default: NYI("tensor->version is not supported");
  }

  return 0u;
}

inline uint32_t get_qnn_tensor_id(const Qnn_Tensor_t& tensor) { return get_qnn_tensor_id(&tensor); }

inline const char* get_qnn_tensor_name(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.name;
    case QNN_TENSOR_VERSION_2: return tensor->v2.name;
    default: NYI("tensor->version is not supported");
  }

  return nullptr;
}

inline const char* get_qnn_tensor_name(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_name(&tensor);
}

// TODO

#define HELP_QNN_TENSOR_VALIDATE_VERSION(__t) MLLM_RT_ASSERT_EQ(validate_tensor_version(__t), true);
#define HELP_QNN_TENSOR_GET_ID(__t) get_qnn_tensor_id(__t)
#define HELP_QNN_TENSOR_GET_NAME(__t) get_qnn_tensor_name(__t)

// TODO
}  // namespace mllm::qnn
