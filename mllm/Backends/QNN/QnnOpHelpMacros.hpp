/**
 * @file QnnOpHelpMacros.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <QNN/QnnTypes.h>
#include "mllm/Utils/Common.hpp"

namespace mllm::qnn {

inline bool validate_op_config_version(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return true;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version); return false;
  }
}

inline bool validate_op_config_version(const Qnn_OpConfig_t& opConfig) {
  return validate_op_config_version(&opConfig);
}

inline const char* get_qnn_op_config_name(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.name;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return nullptr;
}

inline const char* get_qnn_op_config_name(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_name(&opConfig);
}

inline const char* get_qnn_op_config_package_name(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.packageName;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return nullptr;
}

inline const char* get_qnn_op_config_package_name(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_package_name(&opConfig);
}

inline const char* get_qnn_op_config_type_name(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.typeName;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return nullptr;
}

inline const char* get_qnn_op_config_type_name(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_type_name(&opConfig);
}

inline uint32_t get_qnn_op_config_num_params(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.numOfParams;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return 0u;
}

inline uint32_t get_qnn_op_config_num_params(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_num_params(&opConfig);
}

inline const Qnn_Param_t* get_qnn_op_config_params(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.params;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return nullptr;
}

inline const Qnn_Param_t* get_qnn_op_config_params(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_params(&opConfig);
}

inline uint32_t get_qnn_op_config_num_inputs(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.numOfInputs;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return 0u;
}

inline uint32_t get_qnn_op_config_num_inputs(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_num_inputs(&opConfig);
}

inline const Qnn_Tensor_t* get_qnn_op_config_inputs(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.inputTensors;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return nullptr;
}

inline const Qnn_Tensor_t* get_qnn_op_config_inputs(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_inputs(&opConfig);
}

inline uint32_t get_qnn_op_config_num_outputs(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.numOfOutputs;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return 0u;
}

inline uint32_t get_qnn_op_config_num_outputs(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_num_outputs(&opConfig);
}

inline const Qnn_Tensor_t* get_qnn_op_config_outputs(const Qnn_OpConfig_t* opConfig) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: return opConfig->v1.outputTensors;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
  return nullptr;
}

inline const Qnn_Tensor_t* get_qnn_op_config_outputs(const Qnn_OpConfig_t& opConfig) {
  return get_qnn_op_config_outputs(&opConfig);
}

inline void set_qnn_op_config_name(Qnn_OpConfig_t* opConfig, const char* name) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: opConfig->v1.name = name; break;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
}

inline void set_qnn_op_config_name(Qnn_OpConfig_t& opConfig, const char* name) {
  set_qnn_op_config_name(&opConfig, name);
}

inline void set_qnn_op_config_package_name(Qnn_OpConfig_t* opConfig, const char* packageName) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: opConfig->v1.packageName = packageName; break;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
}

inline void set_qnn_op_config_package_name(Qnn_OpConfig_t& opConfig, const char* packageName) {
  set_qnn_op_config_package_name(&opConfig, packageName);
}

inline void set_qnn_op_config_type_name(Qnn_OpConfig_t* opConfig, const char* typeName) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1: opConfig->v1.typeName = typeName; break;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
}

inline void set_qnn_op_config_type_name(Qnn_OpConfig_t& opConfig, const char* typeName) {
  set_qnn_op_config_type_name(&opConfig, typeName);
}

inline void set_qnn_op_config_params(Qnn_OpConfig_t* opConfig, uint32_t numOfParams,
                                     Qnn_Param_t* params) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1:
      opConfig->v1.numOfParams = numOfParams;
      opConfig->v1.params = params;
      break;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
}

inline void set_qnn_op_config_params(Qnn_OpConfig_t& opConfig, uint32_t numOfParams,
                                     Qnn_Param_t* params) {
  set_qnn_op_config_params(&opConfig, numOfParams, params);
}

inline void set_qnn_op_config_inputs(Qnn_OpConfig_t* opConfig, uint32_t numOfInputs,
                                     Qnn_Tensor_t* inputTensors) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1:
      opConfig->v1.numOfInputs = numOfInputs;
      opConfig->v1.inputTensors = inputTensors;
      break;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
}

inline void set_qnn_op_config_inputs(Qnn_OpConfig_t& opConfig, uint32_t numOfInputs,
                                     Qnn_Tensor_t* inputTensors) {
  set_qnn_op_config_inputs(&opConfig, numOfInputs, inputTensors);
}

inline void set_qnn_op_config_outputs(Qnn_OpConfig_t* opConfig, uint32_t numOfOutputs,
                                      Qnn_Tensor_t* outputTensors) {
  switch (opConfig->version) {
    case QNN_OPCONFIG_VERSION_1:
      opConfig->v1.numOfOutputs = numOfOutputs;
      opConfig->v1.outputTensors = outputTensors;
      break;
    default: NYI("Unsupported opConfig version: {}", (int)opConfig->version);
  }
}

inline void set_qnn_op_config_outputs(Qnn_OpConfig_t& opConfig, uint32_t numOfOutputs,
                                      Qnn_Tensor_t* outputTensors) {
  set_qnn_op_config_outputs(&opConfig, numOfOutputs, outputTensors);
}

#define HELP_QNN_OPCONFIG_VALIDATE_VERSION(__c) validate_op_config_version(__c)
#define HELP_QNN_OPCONFIG_GET_NAME(__c) get_qnn_op_config_name(__c)
#define HELP_QNN_OPCONFIG_GET_PACKAGE_NAME(__c) get_qnn_op_config_package_name(__c)
#define HELP_QNN_OPCONFIG_GET_TYPE_NAME(__c) get_qnn_op_config_type_name(__c)
#define HELP_QNN_OPCONFIG_GET_NUM_PARAMS(__c) get_qnn_op_config_num_params(__c)
#define HELP_QNN_OPCONFIG_GET_PARAMS(__c) get_qnn_op_config_params(__c)
#define HELP_QNN_OPCONFIG_GET_NUM_INPUTS(__c) get_qnn_op_config_num_inputs(__c)
#define HELP_QNN_OPCONFIG_GET_INPUTS(__c) get_qnn_op_config_inputs(__c)
#define HELP_QNN_OPCONFIG_GET_NUM_OUTPUTS(__c) get_qnn_op_config_num_outputs(__c)
#define HELP_QNN_OPCONFIG_GET_OUTPUTS(__c) get_qnn_op_config_outputs(__c)

#define HELP_QNN_OPCONFIG_SET_NAME(__c, __v) set_qnn_op_config_name(__c, __v)
#define HELP_QNN_OPCONFIG_SET_PACKAGE_NAME(__c, __v) set_qnn_op_config_package_name(__c, __v)
#define HELP_QNN_OPCONFIG_SET_TYPE_NAME(__c, __v) set_qnn_op_config_type_name(__c, __v)
#define HELP_QNN_OPCONFIG_SET_PARAMS(__c, __n, __p) set_qnn_op_config_params(__c, __n, __p)
#define HELP_QNN_OPCONFIG_SET_INPUTS(__c, __n, __i) set_qnn_op_config_inputs(__c, __n, __i)
#define HELP_QNN_OPCONFIG_SET_OUTPUTS(__c, __n, __o) set_qnn_op_config_outputs(__c, __n, __o)

}  // namespace mllm::qnn