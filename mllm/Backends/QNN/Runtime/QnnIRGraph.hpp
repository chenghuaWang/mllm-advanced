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
#include <vector>
#include <unordered_map>
#include "mllm/IR/Graph/Op.hpp"
#include "mllm/IR/Tensor/Value.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"

// QNN SDK
#include <QNN/QnnTensor.h>
#include <QNN/QnnGraph.h>
#include <QNN/QnnContext.h>

// QNN HTP
#include <QNN/HTP/QnnHtpGraph.h>

namespace mllm::qnn {

class QnnIRGraph;

class QnnIRGraph {
 public:
  inline ~QnnIRGraph() { free(); }

  QnnIRGraph(const std::string& name, const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
             const QnnFuncSymbols& qnn_func_symbols, const QnnBackendDevice& qnn_bk_device);

  void setupInputs(const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs);

  void setupOutputs(const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs);

  void setupInputsFromBinary(const std::vector<Qnn_Tensor_t>& inputs);

  void setupOutputsFromBinary(const std::vector<Qnn_Tensor_t>& outputs);

  inline std::vector<Qnn_Tensor_t>& getInputs() { return qnn_input_tensors_; }

  inline std::vector<Qnn_Tensor_t>& getOutputs() { return qnn_output_tensors_; }

  static std::shared_ptr<QnnIRGraph> build(const std::string& name,
                                           const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                                           const QnnFuncSymbols& qnn_func_symbols,
                                           const QnnBackendDevice& qnn_bk_device);

  void startRecord();

  void endRecord();

  bool addOp(Qnn_OpConfigVersion_t version, const std::string& op_name,
             const std::string& op_package_name, const std::string& type,
             const std::vector<Qnn_Param_t*>& params, const std::vector<std::string>& input_names,
             const std::vector<Qnn_Tensor_t>& output_tensors);

  bool addTensor(const std::string& node_name, Qnn_Tensor_t* qnn_tensor_ptr,
                 bool save_tensor = true);

  bool addTensor(const std::string& node_name, Qnn_Tensor_t& qnn_tensor_ref,
                 bool save_tensor = true);

  void getTensor(const std::string& node_name, const std::string& tensor_name,
                 Qnn_Tensor_t& tensor);

  void compile();

  void free();

  inline ir::graph::SubGraphOp::self_ptr_t ir() { return graph_ir_; }

  static const std::string QTI_AISW_OP_PACKAGE;
  static const std::string MLLM_QNN_OP_PACKAGE;

  inline const QnnBackendDevice& qnnBackendDevice() { return qnn_bk_device_; }
  inline const QnnFuncSymbols& qnnFuncSymbols() { return qnn_func_symbols_; }

  inline Qnn_GraphHandle_t qnnGraphHandle() { return qnn_graph_handle_; }

  inline Qnn_GraphHandle_t* qnnGraphHandlePtr() { return &qnn_graph_handle_; }

 private:
  // freezed
  bool freezed_ = false;

  // wrapped qnn functions
  const QnnBackendDevice& qnn_bk_device_;
  const QnnFuncSymbols& qnn_func_symbols_;

  // QNN meta info
  int32_t qnn_graph_cfg_cnt_ = 0;
  QnnGraph_Config_t qnn_graph_cfg_default_;
  QnnGraph_Config_t* qnn_graph_all_cfgs_[9];  // Only 8(max) cfg can be used. The last is nullptr
  Qnn_GraphHandle_t qnn_graph_handle_ = nullptr;
  std::vector<Qnn_Tensor_t> qnn_input_tensors_;
  std::vector<Qnn_Tensor_t> qnn_output_tensors_;

  // QNN HTP GRAPH Custom Config
  QnnHtpGraph_CustomConfig_t qnn_htp_graph_custom_cfg_default_;

  // map all input tensor and output tensor's name
  std::unordered_map<std::string, Qnn_Tensor_t> qnn_tensor_map_;
  // map op_name->this_op_output_tensor_names
  std::unordered_map<std::string, std::vector<std::string>> qnn_op_output_tensor_map_;

  // mllm info
  std::string name_;
  ir::graph::SubGraphOp::self_ptr_t graph_ir_ = nullptr;

  // static data segment
  static const std::unordered_map<Qnn_DataType_t, size_t> dtype_to_size_;
};

}  // namespace mllm::qnn
