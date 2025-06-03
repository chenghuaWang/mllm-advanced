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
#include <QNN/HTP/QnnHtpGraph.h>
#include "mllm/Backends/QNN/Runtime/QnnIRGraph.hpp"

namespace mllm::qnn {
QnnIRGraph::QnnIRGraph(const std::string& name, const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                       const QnnFuncSymbols& qnn_func_symbols,
                       const QnnBackendDevice& qnn_bk_device)
    : name_(name),
      graph_ir_(graph_ir),
      qnn_func_symbols_(qnn_func_symbols),
      qnn_bk_device_(qnn_bk_device) {}

void QnnIRGraph::setUpInputsOutputs(
    const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
    const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) {}

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

bool QnnIRGraph::addTensor(const std::string& tensor_name, Qnn_Tensor_t* qnn_tensor_ptr,
                           bool save_tensor) {
  if (!qnn_tensor_ptr) { MLLM_ERROR_EXIT(kError, "tensor is nil"); }

  // TODO macros

  return true;
}

bool QnnIRGraph::addTensor(const std::string& tensor_name, Qnn_Tensor_t& qnn_tensor_ref,
                           bool save_tensor) {
  return addTensor(tensor_name, &qnn_tensor_ref, save_tensor);
}

void QnnIRGraph::freezeAndCompile() {}

void QnnIRGraph::free() {}

}  // namespace mllm::qnn
