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
#include "mllm/Backends/QNN/QnnAllocator.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"

// QNN SDK
#include <QNN/QnnTensor.h>
#include <QNN/QnnGraph.h>
#include <QNN/QnnContext.h>

namespace mllm::qnn {

class QnnIRGraph;

class QnnIRGraph {
 public:
  QnnIRGraph(const std::string& name, const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
             const QnnFuncSymbols& qnn_func_symbols, const QnnBackendDevice& qnn_bk_device,
             const std::shared_ptr<QnnAllocator>& allocator);

  void setupInputs(const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs);

  void setupOutputs(const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs);

  static std::shared_ptr<QnnIRGraph> build(const std::string& name,
                                           const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                                           const QnnFuncSymbols& qnn_func_symbols,
                                           const QnnBackendDevice& qnn_bk_device,
                                           const std::shared_ptr<QnnAllocator>& allocator);

  void startRecord();

  void endRecord();

  bool addOp(Qnn_OpConfigVersion_t version, const std::string& op_name,
             const std::string& op_package_name, const std::string& type,
             const std::vector<Qnn_Param_t*>& params, const std::vector<std::string>& input_names,
             const std::vector<Qnn_Tensor_t*>& output_tensors);

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

 private:
  // freezed
  bool freezed_ = false;

  // wrapped qnn functions
  const QnnBackendDevice& qnn_bk_device_;
  const QnnFuncSymbols& qnn_func_symbols_;

  // QNN meta info
  QnnGraph_Config_t qnn_graph_cfg_;
  Qnn_GraphHandle_t qnn_graph_handle_ = nullptr;
  std::vector<Qnn_Tensor_t> qnn_input_tensors_;
  std::vector<Qnn_Tensor_t> qnn_output_tensors_;
  std::vector<uint8_t*> qnn_input_shared_buffer_ptr_;
  std::vector<uint8_t*> qnn_output_shared_buffer_ptr_;

  // map all input tensor and output tensor's name
  std::unordered_map<std::string, Qnn_Tensor_t> qnn_tensor_map_;
  // map op_name->this_op_output_tensor_names
  std::unordered_map<std::string, std::vector<std::string>> qnn_op_output_tensor_map_;

  // mllm info
  std::string name_;
  ir::graph::SubGraphOp::self_ptr_t graph_ir_ = nullptr;
  std::shared_ptr<QnnAllocator> qnn_allocator_ = nullptr;

  // static data segment
  static const std::unordered_map<Qnn_DataType_t, size_t> dtype_to_size_;
};

}  // namespace mllm::qnn
