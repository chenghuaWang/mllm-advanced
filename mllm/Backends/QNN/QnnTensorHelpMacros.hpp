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

inline bool validate_tensor_version(Qnn_Tensor_t& tensor) {
  if (tensor.version != QNN_TENSOR_VERSION_1 && tensor.version != QNN_TENSOR_VERSION_2) {
    MLLM_ERROR_EXIT(kError, "The mllm-advanced lib only support QNN_TENSOR_VERSION_1 and "
                            "QNN_TENSOR_VERSION_2 right now");
  }
  return true;
}

inline bool validate_tensor_version(Qnn_Tensor_t* tensor) {
  if (tensor->version != QNN_TENSOR_VERSION_1 && tensor->version != QNN_TENSOR_VERSION_2) {
    MLLM_ERROR_EXIT(kError, "The mllm-advanced lib only support QNN_TENSOR_VERSION_1 and "
                            "QNN_TENSOR_VERSION_2 right now!");
  }
  return true;
}

inline uint32_t get_qnn_tensor_id(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.id;
    case QNN_TENSOR_VERSION_2: return tensor->v2.id;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }

  return 0u;
}

inline uint32_t get_qnn_tensor_id(const Qnn_Tensor_t& tensor) { return get_qnn_tensor_id(&tensor); }

inline const char* get_qnn_tensor_name(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.name;
    case QNN_TENSOR_VERSION_2: return tensor->v2.name;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }

  return nullptr;
}

inline const char* get_qnn_tensor_name(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_name(&tensor);
}

inline Qnn_TensorType_t get_qnn_tensor_type(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.type;
    case QNN_TENSOR_VERSION_2: return tensor->v2.type;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return QNN_TENSOR_TYPE_UNDEFINED;
}

inline Qnn_TensorType_t get_qnn_tensor_type(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_type(&tensor);
}

inline Qnn_TensorDataFormat_t get_qnn_tensor_data_format(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.dataFormat;
    case QNN_TENSOR_VERSION_2: return tensor->v2.dataFormat;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

inline Qnn_TensorDataFormat_t get_qnn_tensor_data_format(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_data_format(&tensor);
}

inline Qnn_DataType_t get_qnn_tensor_data_type(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.dataType;
    case QNN_TENSOR_VERSION_2: return tensor->v2.dataType;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return QNN_DATATYPE_UNDEFINED;
}

inline Qnn_DataType_t get_qnn_tensor_data_type(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_data_type(&tensor);
}

inline Qnn_QuantizeParams_t get_qnn_tensor_quant_params(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.quantizeParams;
    case QNN_TENSOR_VERSION_2: return tensor->v2.quantizeParams;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return QNN_QUANTIZE_PARAMS_INIT;
}

inline Qnn_QuantizeParams_t get_qnn_tensor_quant_params(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_quant_params(&tensor);
}

inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.rank;
    case QNN_TENSOR_VERSION_2: return tensor->v2.rank;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return 0u;
}

inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_rank(&tensor);
}

inline uint32_t* get_qnn_tensor_dimensions(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.dimensions;
    case QNN_TENSOR_VERSION_2: return tensor->v2.dimensions;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return nullptr;
}

inline uint32_t* get_qnn_tensor_dimensions(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_dimensions(&tensor);
}

inline Qnn_TensorMemType_t get_qnn_tensor_mem_type(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.memType;
    case QNN_TENSOR_VERSION_2: return tensor->v2.memType;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return QNN_TENSORMEMTYPE_UNDEFINED;
}

inline Qnn_TensorMemType_t get_qnn_tensor_mem_type(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_mem_type(&tensor);
}

inline Qnn_ClientBuffer_t get_qnn_tensor_client_buf(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.clientBuf;
    case QNN_TENSOR_VERSION_2: return tensor->v2.clientBuf;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return QNN_CLIENT_BUFFER_INIT;
}

inline Qnn_ClientBuffer_t get_qnn_tensor_client_buf(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_client_buf(&tensor);
}

inline Qnn_MemHandle_t get_qnn_tensor_mem_handle(const Qnn_Tensor_t* tensor) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: return tensor->v1.memHandle;
    case QNN_TENSOR_VERSION_2: return tensor->v2.memHandle;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
  return nullptr;
}

inline Qnn_MemHandle_t get_qnn_tensor_mem_handle(const Qnn_Tensor_t& tensor) {
  return get_qnn_tensor_mem_handle(&tensor);
}

inline void set_qnn_tensor_id(Qnn_Tensor_t* tensor, uint32_t id) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.id = id; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.id = id; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_id(Qnn_Tensor_t& tensor, uint32_t id) { set_qnn_tensor_id(&tensor, id); }

inline void set_qnn_tensor_name(Qnn_Tensor_t* tensor, const char* name) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.name = name; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.name = name; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_name(Qnn_Tensor_t& tensor, const char* name) {
  set_qnn_tensor_name(&tensor, name);
}

inline void set_qnn_tensor_type(Qnn_Tensor_t* tensor, Qnn_TensorType_t type) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.type = type; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.type = type; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_type(Qnn_Tensor_t& tensor, Qnn_TensorType_t type) {
  set_qnn_tensor_type(&tensor, type);
}

inline void set_qnn_tensor_data_format(Qnn_Tensor_t* tensor, Qnn_TensorDataFormat_t format) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.dataFormat = format; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.dataFormat = format; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_data_format(Qnn_Tensor_t& tensor, Qnn_TensorDataFormat_t format) {
  set_qnn_tensor_data_format(&tensor, format);
}

inline void set_qnn_tensor_data_type(Qnn_Tensor_t* tensor, Qnn_DataType_t dataType) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.dataType = dataType; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.dataType = dataType; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_data_type(Qnn_Tensor_t& tensor, Qnn_DataType_t dataType) {
  set_qnn_tensor_data_type(&tensor, dataType);
}

inline void set_qnn_tensor_quant_params(Qnn_Tensor_t* tensor, Qnn_QuantizeParams_t params) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.quantizeParams = params; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.quantizeParams = params; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_quant_params(Qnn_Tensor_t& tensor, Qnn_QuantizeParams_t params) {
  set_qnn_tensor_quant_params(&tensor, params);
}

inline void set_qnn_tensor_rank(Qnn_Tensor_t* tensor, uint32_t rank) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.rank = rank; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.rank = rank; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_rank(Qnn_Tensor_t& tensor, uint32_t rank) {
  set_qnn_tensor_rank(&tensor, rank);
}

inline void set_qnn_tensor_dimensions(Qnn_Tensor_t* tensor, uint32_t* dims) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.dimensions = dims; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.dimensions = dims; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_dimensions(Qnn_Tensor_t& tensor, uint32_t* dims) {
  set_qnn_tensor_dimensions(&tensor, dims);
}

inline void set_qnn_tensor_mem_type(Qnn_Tensor_t* tensor, Qnn_TensorMemType_t memType) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.memType = memType; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.memType = memType; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_mem_type(Qnn_Tensor_t& tensor, Qnn_TensorMemType_t memType) {
  set_qnn_tensor_mem_type(&tensor, memType);
}

inline void set_qnn_tensor_client_buf(Qnn_Tensor_t* tensor, Qnn_ClientBuffer_t clientBuf) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.clientBuf = clientBuf; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.clientBuf = clientBuf; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_client_buf(Qnn_Tensor_t& tensor, Qnn_ClientBuffer_t clientBuf) {
  set_qnn_tensor_client_buf(&tensor, clientBuf);
}

inline void set_qnn_tensor_mem_handle(Qnn_Tensor_t* tensor, Qnn_MemHandle_t handle) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1: tensor->v1.memHandle = handle; break;
    case QNN_TENSOR_VERSION_2: tensor->v2.memHandle = handle; break;
    default: NYI("Unsupported tensor version: {}", (int)tensor->version);
  }
}

inline void set_qnn_tensor_mem_handle(Qnn_Tensor_t& tensor, Qnn_MemHandle_t handle) {
  set_qnn_tensor_mem_handle(&tensor, handle);
}

#define HELP_QNN_TENSOR_VALIDATE_VERSION(__t) MLLM_RT_ASSERT_EQ(validate_tensor_version(__t), true);

#define HELP_QNN_TENSOR_GET_ID(__t) get_qnn_tensor_id(__t)
#define HELP_QNN_TENSOR_GET_NAME(__t) get_qnn_tensor_name(__t)
#define HELP_QNN_TENSOR_GET_TYPE(__t) get_qnn_tensor_type(__t)
#define HELP_QNN_TENSOR_GET_DATA_FORMAT(__t) get_qnn_tensor_data_format(__t)
#define HELP_QNN_TENSOR_GET_DATA_TYPE(__t) get_qnn_tensor_data_type(__t)
#define HELP_QNN_TENSOR_GET_QUANT_PARAMS(__t) get_qnn_tensor_quant_params(__t)
#define HELP_QNN_TENSOR_GET_RANK(__t) get_qnn_tensor_rank(__t)
#define HELP_QNN_TENSOR_GET_DIMENSIONS(__t) get_qnn_tensor_dimensions(__t)
#define HELP_QNN_TENSOR_GET_MEM_TYPE(__t) get_qnn_tensor_mem_type(__t)
#define HELP_QNN_TENSOR_GET_CLIENT_BUF(__t) get_qnn_tensor_client_buf(__t)
#define HELP_QNN_TENSOR_GET_MEM_HANDLE(__t) get_qnn_tensor_mem_handle(__t)

#define HELP_QNN_TENSOR_SET_ID(__t, __v) set_qnn_tensor_id(__t, __v)
#define HELP_QNN_TENSOR_SET_NAME(__t, __v) set_qnn_tensor_name(__t, __v)
#define HELP_QNN_TENSOR_SET_TYPE(__t, __v) set_qnn_tensor_type(__t, __v)
#define HELP_QNN_TENSOR_SET_DATA_FORMAT(__t, __v) set_qnn_tensor_data_format(__t, __v)
#define HELP_QNN_TENSOR_SET_DATA_TYPE(__t, __v) set_qnn_tensor_data_type(__t, __v)
#define HELP_QNN_TENSOR_SET_QUANT_PARAMS(__t, __v) set_qnn_tensor_quant_params(__t, __v)
#define HELP_QNN_TENSOR_SET_RANK(__t, __v) set_qnn_tensor_rank(__t, __v)
#define HELP_QNN_TENSOR_SET_DIMENSIONS(__t, __v) set_qnn_tensor_dimensions(__t, __v)
#define HELP_QNN_TENSOR_SET_MEM_TYPE(__t, __v) set_qnn_tensor_mem_type(__t, __v)
#define HELP_QNN_TENSOR_SET_CLIENT_BUF(__t, __v) set_qnn_tensor_client_buf(__t, __v)
#define HELP_QNN_TENSOR_SET_MEM_HANDLE(__t, __v) set_qnn_tensor_mem_handle(__t, __v)

}  // namespace mllm::qnn
