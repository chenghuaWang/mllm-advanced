/**
 * @file QnnIRGraph.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <string>
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"

// QNN SDK
#include <QNN/QnnTensor.h>
#include <QNN/QnnGraph.h>
#include <QNN/QnnContext.h>

namespace mllm::qnn {

class QnnIRGraph {
 public:
  QnnIRGraph(const std::string& name, const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
             const QnnFuncSymbols& qnn_func_symbols, const QnnBackendDevice& qnn_bk_device);

  void setUpInputsOutputs(const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
                          const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs);

  static std::shared_ptr<QnnIRGraph> build(const std::string& name,
                                           const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                                           const QnnFuncSymbols& qnn_func_symbols,
                                           const QnnBackendDevice& qnn_bk_device);

  void startRecord();

  void endRecord();

  bool addTensor(const std::string& tensor_name, Qnn_Tensor_t* qnn_tensor_ptr,
                 bool save_tensor = true);

  bool addTensor(const std::string& tensor_name, Qnn_Tensor_t& qnn_tensor_ref,
                 bool save_tensor = true);

  void freezeAndCompile();

  void free();

 private:
  // freezed
  bool freezed_ = false;

  // QNN meta info
  Qnn_GraphHandle_t qnn_graph_handle_ = nullptr;
  std::vector<Qnn_Tensor_t*> qnn_input_tensors_;
  std::vector<Qnn_Tensor_t*> qnn_output_tensors_;
  QnnGraph_Config_t qnn_graph_cfg_;
  Qnn_ContextHandle_t qnn_cxt_handle_ = nullptr;
  QnnContext_Config_t** qnn_context_config = nullptr;

  // wrapped qnn functions
  const QnnBackendDevice& qnn_bk_device_;
  const QnnFuncSymbols& qnn_func_symbols_;

  // mllm info
  std::string name_;
  ir::graph::SubGraphOp::self_ptr_t graph_ir_ = nullptr;
};

}  // namespace mllm::qnn
