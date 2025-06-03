/**
 * @file QnnIRGraph.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cstdint>
#include <numeric>
#include <QNN/HTP/QnnHtpGraph.h>
#include "mllm/Backends/QNN/Runtime/QnnIRGraph.hpp"
#include "mllm/Backends/QNN/QnnTensorHelpMacros.hpp"

namespace mllm::qnn {

const std::unordered_map<Qnn_DataType_t, size_t> QnnIRGraph::dtype_to_size_ = {
    {QNN_DATATYPE_INT_8, 1},           {QNN_DATATYPE_INT_16, 2},
    {QNN_DATATYPE_INT_32, 4},          {QNN_DATATYPE_INT_64, 8},
    {QNN_DATATYPE_UINT_8, 1},          {QNN_DATATYPE_UINT_16, 2},
    {QNN_DATATYPE_UINT_32, 4},         {QNN_DATATYPE_UINT_64, 8},
    {QNN_DATATYPE_FLOAT_16, 2},        {QNN_DATATYPE_FLOAT_32, 4},
    {QNN_DATATYPE_BOOL_8, 1},          {QNN_DATATYPE_SFIXED_POINT_8, 1},
    {QNN_DATATYPE_SFIXED_POINT_16, 2}, {QNN_DATATYPE_SFIXED_POINT_32, 4},
    {QNN_DATATYPE_UFIXED_POINT_8, 1},  {QNN_DATATYPE_UFIXED_POINT_16, 2},
    {QNN_DATATYPE_UFIXED_POINT_32, 4},
};

QnnIRGraph::QnnIRGraph(const std::string& name, const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                       const QnnFuncSymbols& qnn_func_symbols,
                       const QnnBackendDevice& qnn_bk_device)
    : name_(name),
      graph_ir_(graph_ir),
      qnn_func_symbols_(qnn_func_symbols),
      qnn_bk_device_(qnn_bk_device) {}

void QnnIRGraph::setUpInputsOutputs(
    const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
    const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) {
  // TODO
}

std::shared_ptr<QnnIRGraph> QnnIRGraph::build(const std::string& name,
                                              const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                                              const QnnFuncSymbols& qnn_func_symbols,
                                              const QnnBackendDevice& qnn_bk_device) {
  return std::make_shared<QnnIRGraph>(name, graph_ir, qnn_func_symbols, qnn_bk_device);
}

void QnnIRGraph::startRecord() {
  // create context for this graph
  auto status = qnn_func_symbols_.qnn_interface_.contextCreate(
      qnn_bk_device_.bk_handle_, qnn_bk_device_.device_handle_,
      (const QnnContext_Config_t**)&qnn_context_config, &qnn_cxt_handle_);
  MLLM_RT_ASSERT_EQ(QNN_CONTEXT_NO_ERROR, status);

  // Current only support HTP
  QnnHtpGraph_CustomConfig_t custom_cfg;
  custom_cfg.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  custom_cfg.vtcmSizeInMB = 8;

  QnnGraph_Config_t graph_cfg;
  graph_cfg.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_cfg.customConfig = &custom_cfg;

  // create graph
  status = qnn_func_symbols_.qnn_interface_.graphCreate(
      qnn_cxt_handle_, name_.c_str(), (const QnnGraph_Config_t**)&graph_cfg, &qnn_graph_handle_);
  MLLM_RT_ASSERT_EQ(status, QNN_GRAPH_NO_ERROR);
}

void QnnIRGraph::endRecord() {
  // TODO free context
}

bool QnnIRGraph::addTensor(const std::string& node_name, Qnn_Tensor_t* qnn_tensor_ptr,
                           bool save_tensor) {
  if (!qnn_tensor_ptr) { MLLM_ERROR_EXIT(kError, "tensor is nil"); }

  HELP_QNN_TENSOR_VALIDATE_VERSION(qnn_tensor_ptr);

  // Check tensor being added is not in this graph.
  MLLM_RT_ASSERT_EQ(qnn_tensor_map_.count(HELP_QNN_TENSOR_GET_NAME(qnn_tensor_ptr)), 0);

  // Check quant is correct.
  MLLM_RT_ASSERT_EQ(dtype_to_size_.count(HELP_QNN_TENSOR_GET_DATA_TYPE(qnn_tensor_ptr)), 1);

  // Check static tensor's mem type is raw
  if (HELP_QNN_TENSOR_GET_TYPE(qnn_tensor_ptr) == QNN_TENSOR_TYPE_STATIC) {
    if (HELP_QNN_TENSOR_GET_MEM_TYPE(qnn_tensor_ptr) != QNN_TENSORMEMTYPE_RAW) {
      MLLM_ERROR_EXIT(kError, "Static tensor's mem type must be raw.");
    }
  }

  // Check tensor size is rights
  uint32_t qnn_tensor_size = std::accumulate(
      HELP_QNN_TENSOR_GET_DIMENSIONS(qnn_tensor_ptr),
      HELP_QNN_TENSOR_GET_DIMENSIONS(qnn_tensor_ptr) + HELP_QNN_TENSOR_GET_RANK(qnn_tensor_ptr),
      (uint32_t)(dtype_to_size_.find(HELP_QNN_TENSOR_GET_DATA_TYPE(qnn_tensor_ptr))->second),
      std::multiplies<>());

  MLLM_RT_ASSERT_EQ(HELP_QNN_TENSOR_GET_CLIENT_BUF(qnn_tensor_ptr).dataSize, qnn_tensor_size);

  // Everything is OK. Make tensor.
  MLLM_RT_ASSERT(
      qnn_func_symbols_.qnn_interface_.tensorCreateGraphTensor(qnn_graph_handle_, qnn_tensor_ptr)
      != QNN_TENSOR_NO_ERROR);

  // Save tensor
  // If this tensor is input or output tensor, we need to save it for later pickup
  if (save_tensor) {
    Qnn_Tensor_t qnn_tensor_clone;
    HELP_QNN_TENSOR_CLONE(qnn_tensor_clone, *qnn_tensor_ptr);

    switch (HELP_QNN_TENSOR_GET_TYPE(qnn_tensor_ptr)) {
      case QNN_TENSOR_TYPE_APP_WRITE: qnn_input_tensors_.push_back(qnn_tensor_clone); break;
      case QNN_TENSOR_TYPE_APP_READ: qnn_output_tensors_.push_back(qnn_tensor_clone); break;
      default:
        MLLM_ERROR_EXIT(kError, "only [QNN_TENSOR_TYPE_APP_WRITE|QNN_TENSOR_TYPE_APP_READ] can be "
                                "saved as input and output");
    }

    qnn_tensor_map_.insert({HELP_QNN_TENSOR_GET_NAME(qnn_tensor_ptr), qnn_tensor_clone});
  }

  return true;
}

bool QnnIRGraph::addTensor(const std::string& node_name, Qnn_Tensor_t& qnn_tensor_ref,
                           bool save_tensor) {
  return addTensor(node_name, &qnn_tensor_ref, save_tensor);
}

void QnnIRGraph::freezeAndCompile() {}

void QnnIRGraph::free() {}

}  // namespace mllm::qnn
