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
#include "mllm/Backends/QNN/QnnOpHelpMacros.hpp"
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"

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

const std::string QnnIRGraph::QTI_AISW_OP_PACKAGE = "qti.aisw";

const std::string QnnIRGraph::MLLM_QNN_OP_PACKAGE = "MllmQnnOpPackage";

QnnIRGraph::QnnIRGraph(const std::string& name, const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                       const QnnFuncSymbols& qnn_func_symbols,
                       const QnnBackendDevice& qnn_bk_device)
    : name_(name),
      graph_ir_(graph_ir),
      qnn_func_symbols_(qnn_func_symbols),
      qnn_bk_device_(qnn_bk_device) {}

void QnnIRGraph::setupInputs(const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs) {
  for (auto& input_tensor_ir : inputs) {
    auto qnn_tensor =
        QnnTensorTransform::instance().transform(input_tensor_ir, QNN_TENSOR_VERSION_2);

    addTensor(
        // use ir value's name instead of mllm_tensor name.
        // QnnTensorNaming Pass will make readable names for QNN, and the name is stored in ir.
        input_tensor_ir->name(), qnn_tensor);
  }
}

void QnnIRGraph::setupOutputs(const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) {
  // NOTE: addOp will add the outputs tensor into this graph.
  // setupOutputs's responsibility is to check if the outputs is registered to this graph
  // correctly!!!
  MLLM_RT_ASSERT_EQ(outputs.size(), qnn_output_tensors_.size());
  int os_size = outputs.size();
  for (int i = 0; i < os_size; ++i) {
    MLLM_RT_ASSERT_EQ(outputs[i]->name(), HELP_QNN_TENSOR_GET_NAME(qnn_output_tensors_[i]));
  }
}

std::shared_ptr<QnnIRGraph> QnnIRGraph::build(const std::string& name,
                                              const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                                              const QnnFuncSymbols& qnn_func_symbols,
                                              const QnnBackendDevice& qnn_bk_device) {
  return std::make_shared<QnnIRGraph>(name, graph_ir, qnn_func_symbols, qnn_bk_device);
}

void QnnIRGraph::startRecord() {
  MLLM_RT_ASSERT_EQ(freezed_, false);

  qnn_htp_graph_custom_cfg_default_.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  qnn_htp_graph_custom_cfg_default_.vtcmSizeInMB = 8;

  qnn_graph_cfg_default_.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  qnn_graph_cfg_default_.customConfig = &qnn_htp_graph_custom_cfg_default_;

  qnn_graph_all_cfgs_[qnn_graph_cfg_cnt_++] = &qnn_graph_cfg_default_;
  qnn_graph_all_cfgs_[qnn_graph_cfg_cnt_++] = nullptr;

  MLLM_RT_ASSERT(qnn_graph_cfg_cnt_ <= 9);

  // create graph
  auto status = qnn_func_symbols_.qnn_interface_.graphCreate(
      qnn_bk_device_.qnn_ctx_handle_, name_.c_str(),
      (const QnnGraph_Config_t**)(qnn_graph_all_cfgs_), &qnn_graph_handle_);

  MLLM_RT_ASSERT_EQ(status, QNN_GRAPH_NO_ERROR);
}

void QnnIRGraph::endRecord() {
  // Freeze this Graph
  freezed_ = true;
}

bool QnnIRGraph::addOp(Qnn_OpConfigVersion_t version, const std::string& op_name,
                       const std::string& op_package_name, const std::string& type,
                       const std::vector<Qnn_Param_t*>& params,
                       const std::vector<std::string>& input_names,
                       const std::vector<Qnn_Tensor_t>& output_tensors) {
  MLLM_RT_ASSERT_EQ(freezed_, false);

  Qnn_OpConfig_t op_definition = QNN_OPCONFIG_INIT;
  op_definition.version = version;

  HELP_QNN_OPCONFIG_VALIDATE_VERSION(op_definition);

  // TODO Handle memory here
  Qnn_Param_t* node_params = (Qnn_Param_t*)malloc(params.size() * sizeof(Qnn_Param_t));
  if (params.size() == 0) { node_params = nullptr; }
  Qnn_Tensor_t* inputs = (Qnn_Tensor_t*)malloc(input_names.size() * sizeof(Qnn_Tensor_t));
  Qnn_Tensor_t* outputs = (Qnn_Tensor_t*)malloc(output_tensors.size() * sizeof(Qnn_Tensor_t));

  MLLM_RT_ASSERT((params.size() == 0 || node_params != nullptr) && inputs != nullptr
                 && outputs != nullptr);

  // Add correspond params' tensor/scalar to context
  uint32_t node_params_cnt = 0;
  for (auto param_tensor : params) {
    switch (param_tensor->paramType) {
      // The parameter is tensor
      case QNN_PARAMTYPE_TENSOR: {
        Qnn_Tensor_t& tensor = param_tensor->tensorParam;

        // No need to release node_params, inputs, outputs. Will panic immediately
        MLLM_RT_ASSERT_EQ(addTensor(op_name, &tensor, false), true);

        node_params[node_params_cnt].paramType = QNN_PARAMTYPE_TENSOR;
        node_params[node_params_cnt].name = param_tensor->name;
        node_params[node_params_cnt++].tensorParam = tensor;
        break;
      }
      // The parameter is scalar
      case QNN_PARAMTYPE_SCALAR: {
        node_params[node_params_cnt].paramType = QNN_PARAMTYPE_SCALAR;
        node_params[node_params_cnt].name = param_tensor->name;
        node_params[node_params_cnt++].scalarParam = param_tensor->scalarParam;
        break;
      }
      default: MLLM_ERROR_EXIT(kError, "Not supported param tensor type"); break;
    }
  }

  size_t inputs_cnt = 0;
  for (auto const& input_tensor : input_names) {
    getTensor(op_name, input_tensor, inputs[inputs_cnt++]);
  }

  size_t output_cnt = 0;
  MLLM_RT_ASSERT_EQ(qnn_op_output_tensor_map_.count(op_name), 0);
  for (auto output_tensor : output_tensors) {
    // We need to save output tensor to tensor map
    addTensor(op_name, output_tensor, true);

    auto output_tensor_name = HELP_QNN_TENSOR_GET_NAME(output_tensor);
    qnn_op_output_tensor_map_[op_name].emplace_back(output_tensor_name);

    getTensor(op_name, output_tensor_name, outputs[output_cnt++]);
  }

  // TODO Handle string memory
  // Define and add op node to graph
  HELP_QNN_OPCONFIG_SET_NAME(op_definition, strdup(op_name.c_str()));
  HELP_QNN_OPCONFIG_SET_PACKAGE_NAME(op_definition, strdup(op_package_name.c_str()));
  HELP_QNN_OPCONFIG_SET_TYPE_NAME(op_definition, strdup(type.c_str()));
  HELP_QNN_OPCONFIG_SET_PARAMS(op_definition, params.size(), node_params);
  HELP_QNN_OPCONFIG_SET_INPUTS(op_definition, input_names.size(), inputs);
  HELP_QNN_OPCONFIG_SET_OUTPUTS(op_definition, output_tensors.size(), outputs);

  auto status = qnn_func_symbols_.qnn_interface_.backendValidateOpConfig(qnn_bk_device_.bk_handle_,
                                                                         op_definition);
  MLLM_RT_ASSERT(status != QNN_BACKEND_ERROR_NOT_SUPPORTED);
  MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);

  status = qnn_func_symbols_.qnn_interface_.graphAddNode(qnn_graph_handle_, op_definition);
  MLLM_RT_ASSERT_EQ(QNN_GRAPH_NO_ERROR, status);

  ::free(node_params);
  ::free(inputs);
  ::free(outputs);

  return true;
}

bool QnnIRGraph::addTensor(const std::string& node_name, Qnn_Tensor_t* qnn_tensor_ptr,
                           bool save_tensor) {
  MLLM_RT_ASSERT_EQ(freezed_, false);

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

    // Check tensor size is rights
    uint32_t qnn_tensor_size = std::accumulate(
        HELP_QNN_TENSOR_GET_DIMENSIONS(qnn_tensor_ptr),
        HELP_QNN_TENSOR_GET_DIMENSIONS(qnn_tensor_ptr) + HELP_QNN_TENSOR_GET_RANK(qnn_tensor_ptr),
        (uint32_t)(dtype_to_size_.find(HELP_QNN_TENSOR_GET_DATA_TYPE(qnn_tensor_ptr))->second),
        std::multiplies<>());

    MLLM_RT_ASSERT_EQ(HELP_QNN_TENSOR_GET_CLIENT_BUF(qnn_tensor_ptr).dataSize, qnn_tensor_size);
  }

  // Everything is OK. Make tensor.
  MLLM_RT_ASSERT_EQ(
      qnn_func_symbols_.qnn_interface_.tensorCreateGraphTensor(qnn_graph_handle_, qnn_tensor_ptr),
      QNN_TENSOR_NO_ERROR);

  // Save tensor
  // If this tensor is input or output tensor, we need to save it for later pickup
  if (save_tensor) {
    switch (HELP_QNN_TENSOR_GET_TYPE(qnn_tensor_ptr)) {
      case QNN_TENSOR_TYPE_APP_WRITE: qnn_input_tensors_.push_back(*qnn_tensor_ptr); break;
      case QNN_TENSOR_TYPE_APP_READ: qnn_output_tensors_.push_back(*qnn_tensor_ptr); break;
      default:
        MLLM_ERROR_EXIT(kError, "only [QNN_TENSOR_TYPE_APP_WRITE|QNN_TENSOR_TYPE_APP_READ] can be "
                                "saved as input and output");
    }

    qnn_tensor_map_.insert({HELP_QNN_TENSOR_GET_NAME(qnn_tensor_ptr), *qnn_tensor_ptr});
  }

  return true;
}

bool QnnIRGraph::addTensor(const std::string& node_name, Qnn_Tensor_t& qnn_tensor_ref,
                           bool save_tensor) {
  return addTensor(node_name, &qnn_tensor_ref, save_tensor);
}

void QnnIRGraph::getTensor(const std::string& node_name, const std::string& tensor_name,
                           Qnn_Tensor_t& tensor) {
  MLLM_RT_ASSERT_EQ(qnn_tensor_map_.count(tensor_name), 1);
  tensor = qnn_tensor_map_[tensor_name];
}

void QnnIRGraph::compile() {
  MLLM_RT_ASSERT_EQ(freezed_, true);

  // Compile QNN Graph First
  auto status = qnn_func_symbols_.qnn_interface_.graphFinalize(
      qnn_graph_handle_, qnn_bk_device_.profile_bk_handle_, /*signalHandle*/ nullptr);
  MLLM_RT_ASSERT_EQ(QNN_GRAPH_NO_ERROR, status);
}

void QnnIRGraph::free() {
  MLLM_RT_ASSERT_EQ(freezed_, true);
  // No need to free graph and Qnn seems not provide method to free graph.
  // Just free the context and backend. <- QnnBackend will handle this.
}

}  // namespace mllm::qnn
