/**
 * @file QnnBackend.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include <fstream>
#include "mllm/Backends/QNN/QnnBackend.hpp"
#include "mllm/Backends/QNN/Ops/ElewiseOp.hpp"
#include "mllm/Backends/QNN/Ops/MatMulOp.hpp"
#include "mllm/Backends/QNN/Ops/LinearOp.hpp"
#include "mllm/Backends/QNN/Ops/SiLUOp.hpp"
#include "mllm/Backends/QNN/Ops/ViewOp.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLog.hpp"
#include "mllm/Backends/QNN/QnnAllocator.hpp"
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"

namespace mllm::qnn {

QnnBackend::QnnBackend() : BackendBase(kQNN) {
  QnnLogger::instance();

  MLLM_RT_ASSERT_EQ(initHTPBackend(), true);

  // NOTE: Init HTP memory allocator AFTER HTPBackend is inited.
  allocator_ = std::make_shared<QnnAllocator>(qnn_htp_func_symbols_, qnn_htp_backend_);

  // TODO
  regOpFactory<QnnMatMulOpFactory, QnnLinearOpFactory, QnnSiLUOpFactory, QnnAddOpFactory,
               QnnSubOpFactory, QnnMulOpFactory, QnnDivOpFactory, QnnViewOpFactory>();
}

QnnBackend::~QnnBackend() {
  // Free HTP Context
  auto status = qnn_htp_func_symbols_.qnn_interface_.contextFree(
      qnn_htp_backend_.qnn_ctx_handle_, qnn_htp_backend_.profile_bk_handle_);
  MLLM_RT_ASSERT_EQ(status, QNN_CONTEXT_NO_ERROR);

  // Free HTP devices
  status = qnn_htp_func_symbols_.qnn_interface_.deviceFree(qnn_htp_backend_.device_handle_);
  MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);
}

bool QnnBackend::initHTPBackend() {
  auto& loader = QnnDynSymbolLoader::instance();

  // load qnn backend
  constexpr std::string htp_backend_lib_name = "libQnnHtp.so";

  // GLOBAL Load
  if (!loader.loadQnnDynLib(
          htp_backend_lib_name,
          QnnDynSymbolLoader::DynFlag::kRTLD_NOW | QnnDynSymbolLoader::DynFlag::kRTLD_GLOBAL)) {
    return false;
  }

  // Get provider function
  auto qnn_interface_get_providers_func =
      loader(htp_backend_lib_name)
          .func<QnnFuncSymbols::QnnInterfaceGetProvidersFuncType>("QnnInterface_getProviders");

  // Get provider
  QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;

  MLLM_RT_ASSERT_EQ(qnn_interface_get_providers_func((const QnnInterface_t***)&interface_providers,
                                                     &num_providers),
                    QNN_SUCCESS);
  MLLM_RT_ASSERT(interface_providers != nullptr);
  MLLM_RT_ASSERT(num_providers != 0);

  bool found_valid_interface = false;

  // Get correct provider
  for (size_t provider_id = 0; provider_id < num_providers; provider_id++) {
    if (QNN_API_VERSION_MAJOR == interface_providers[provider_id]->apiVersion.coreApiVersion.major
        && QNN_API_VERSION_MINOR
               <= interface_providers[provider_id]->apiVersion.coreApiVersion.minor) {
      found_valid_interface = true;
      qnn_htp_func_symbols_.qnn_interface_ =
          interface_providers[provider_id]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
  MLLM_RT_ASSERT_EQ(found_valid_interface, true);

  // Create logger and register callback.
  auto& _qnn_logger = QnnLogger::instance();
  MLLM_RT_ASSERT_EQ(
      qnn_htp_func_symbols_.qnn_interface_.logCreate(
          _qnn_logger.getLogCallback(), _qnn_logger.getMaxLevel(), &qnn_htp_backend_.log_),
      QNN_SUCCESS)
  MLLM_RT_ASSERT_EQ(QNN_BACKEND_NO_ERROR, qnn_htp_func_symbols_.qnn_interface_.backendCreate(
                                              qnn_htp_backend_.log_,
                                              (const QnnBackend_Config_t**)qnn_htp_backend_.bk_cfg_,
                                              &qnn_htp_backend_.bk_handle_))

  // Check if this HTP Backend has specific property
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability) {
    auto status =
        qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (status == QNN_PROPERTY_NOT_SUPPORTED) { MLLM_WARN("Device property is not supported"); }

    MLLM_RT_ASSERT(status != QNN_PROPERTY_ERROR_UNKNOWN_KEY);
  }

  // Create HTP Device
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.deviceCreate) {
    auto status = qnn_htp_func_symbols_.qnn_interface_.deviceCreate(
        qnn_htp_backend_.log_, nullptr, &qnn_htp_backend_.device_handle_);
    MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);
  }

  // Profile
  qnn_htp_func_symbols_.qnn_interface_.profileCreate(qnn_htp_backend_.bk_handle_,
                                                     QNN_PROFILE_LEVEL_DETAILED,
                                                     &qnn_htp_backend_.profile_bk_handle_);

  // Create context
  auto status = qnn_htp_func_symbols_.qnn_interface_.contextCreate(
      qnn_htp_backend_.bk_handle_, qnn_htp_backend_.device_handle_,
      (const QnnContext_Config_t**)&qnn_htp_backend_.qnn_context_config_,
      &qnn_htp_backend_.qnn_ctx_handle_);
  MLLM_RT_ASSERT_EQ(QNN_CONTEXT_NO_ERROR, status);

  return true;
}

std::shared_ptr<QnnIRGraph> QnnBackend::createQnnGraph(
    const std::string& name, const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
    const QnnFuncSymbols& qnn_func_symbols, const QnnBackendDevice& qnn_bk_device) {
  if (qnn_graphs_.count(name)) {
    MLLM_ERROR_EXIT(kError, "Graph {} already exists", name);
    return nullptr;
  }

  auto ret = QnnIRGraph::build(name, graph_ir, qnn_func_symbols, qnn_bk_device);
  qnn_graphs_.insert({name, ret});
  return ret;
}

std::shared_ptr<QnnIRGraph> QnnBackend::getCompiledQnnGraph(const std::string& name) {
  MLLM_RT_ASSERT_EQ(qnn_graphs_.count(name), 1);
  return qnn_graphs_[name];
}

void QnnBackend::saveHtpContextToBinaryFile(const std::string& file_path) {
  size_t required_buffer_size = 0;
  auto status = qnn_htp_func_symbols_.qnn_interface_.contextGetBinarySize(
      qnn_htp_backend_.qnn_ctx_handle_, &required_buffer_size);
  MLLM_RT_ASSERT_EQ(status, QNN_CONTEXT_NO_ERROR);

  auto binary_buffer_ptr = new uint8_t[required_buffer_size];

  size_t written_buffer_size = 0;

  status = qnn_htp_func_symbols_.qnn_interface_.contextGetBinary(
      qnn_htp_backend_.qnn_ctx_handle_, reinterpret_cast<void*>(binary_buffer_ptr),
      required_buffer_size, &written_buffer_size);
  MLLM_RT_ASSERT_EQ(status, QNN_CONTEXT_NO_ERROR);

  if (required_buffer_size < written_buffer_size) {
    MLLM_ERROR_EXIT(kError, "Memory write error when save qnn's HTP binary context.");
  }

  std::ofstream out_file(file_path, std::ios::binary);
  if (!out_file.is_open()) {
    MLLM_ERROR_EXIT(kError, "Failed to open file {} for writing", file_path);
  }

  out_file.write(reinterpret_cast<const char*>(binary_buffer_ptr), written_buffer_size);
  if (!out_file.good()) {
    MLLM_ERROR_EXIT(kError, "Error occurred when writing to file {}", file_path);
  }

  out_file.close();

  delete[] binary_buffer_ptr;
}

void QnnBackend::loadHtpContextFromBinaryFile(const std::string& file_path) {
  QnnSystemContext_Handle_t sys_context_handle = nullptr;
  auto status =
      qnn_htp_func_symbols_.qnn_system_interface_.systemContextCreate(&sys_context_handle);
  MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);

  QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size = 0;

  std::ifstream in_file(file_path, std::ios::binary | std::ios::ate);
  size_t file_size = in_file.tellg();
  if (file_size == 0) { MLLM_ERROR_EXIT(kError, "File {} is empty", file_path); }
  auto buffer_ptr = new uint8_t[file_size];

  in_file.read(reinterpret_cast<char*>(buffer_ptr), file_size);
  if (!in_file.good()) {
    MLLM_ERROR_EXIT(kError, "Error occurred when reading from file {}", file_path);
  }
  in_file.close();

  status = qnn_htp_func_symbols_.qnn_system_interface_.systemContextGetBinaryInfo(
      sys_context_handle, static_cast<void*>(buffer_ptr), file_size,
      (const QnnSystemContext_BinaryInfo_t**)&binary_info /*FIXME: UNSAFE?*/, &binary_info_size);
  MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);

  size_t graph_cnt = 0;
  {
    auto copy_graphs_info_from_binary_func =
        [this](const QnnSystemContext_GraphInfo_t* binary_graphs_inputs,
               const uint32_t num_graphs) -> bool {
      if (!binary_graphs_inputs) {
        MLLM_WARN("Received nullptr for binary_graphs_inputs.");
        return false;
      }

      for (size_t g_idx = 0; g_idx < num_graphs; g_idx++) {
        switch (binary_graphs_inputs[g_idx].version) {
          case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: {
            auto bg_info = binary_graphs_inputs[g_idx].graphInfoV1;

            // Get graph Name
            std::string g_name = std::string(bg_info.graphName);
            auto g = createQnnGraph(g_name, nullptr, qnn_htp_func_symbols_, qnn_htp_backend_);

            // Set Input tensor
            if (bg_info.numGraphInputs) {
              std::vector<Qnn_Tensor_t> inputs;
              for (int g_input_idx = 0; g_input_idx < bg_info.numGraphInputs; g_input_idx++) {
                // 1. Deep copy tensor info from bg_info to mllm context.
                auto g_input_tmp =
                    QnnTensorTransform::instance().deepCopy(&(bg_info.graphInputs[g_input_idx]));

                inputs.emplace_back(g_input_tmp);
              }
              // 2. Assign to mllm's qnn graph.
              g->setupInputsFromBinary(inputs);
            }

            // Set Output tensor
            if (bg_info.numGraphOutputs) {
              std::vector<Qnn_Tensor_t> outputs;
              for (int g_output_idx = 0; g_output_idx < bg_info.numGraphOutputs; g_output_idx++) {
                // 1. Deep copy tensor info from bg_info to mllm context.
                auto g_output_tmp =
                    QnnTensorTransform::instance().deepCopy(&(bg_info.graphOutputs[g_output_idx]));

                outputs.emplace_back(g_output_tmp);
              }
              // 2. Assign to mllm's qnn graph.
              g->setupOutputsFromBinary(outputs);
            }
            break;
          }
          case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: {
            auto bg_info = binary_graphs_inputs[g_idx].graphInfoV1;

            // Get graph Name
            std::string g_name = std::string(bg_info.graphName);
            auto g = createQnnGraph(g_name, nullptr, qnn_htp_func_symbols_, qnn_htp_backend_);

            // Set Input tensor
            if (bg_info.numGraphInputs) {
              std::vector<Qnn_Tensor_t> inputs;
              for (int g_input_idx = 0; g_input_idx < bg_info.numGraphInputs; g_input_idx++) {
                // 1. Deep copy tensor info from bg_info to mllm context.
                auto g_input_tmp =
                    QnnTensorTransform::instance().deepCopy(&(bg_info.graphInputs[g_input_idx]));

                inputs.emplace_back(g_input_tmp);
              }
              // 2. Assign to mllm's qnn graph.
              g->setupInputsFromBinary(inputs);
            }

            // Set Output tensor
            if (bg_info.numGraphOutputs) {
              std::vector<Qnn_Tensor_t> outputs;
              for (int g_output_idx = 0; g_output_idx < bg_info.numGraphOutputs; g_output_idx++) {
                // 1. Deep copy tensor info from bg_info to mllm context.
                auto g_output_tmp =
                    QnnTensorTransform::instance().deepCopy(&(bg_info.graphOutputs[g_output_idx]));

                outputs.emplace_back(g_output_tmp);
              }
              // 2. Assign to mllm's qnn graph.
              g->setupOutputsFromBinary(outputs);
            }
            break;
          }
          case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: {
            auto bg_info = binary_graphs_inputs[g_idx].graphInfoV1;

            // Get graph Name
            std::string g_name = std::string(bg_info.graphName);
            auto g = createQnnGraph(g_name, nullptr, qnn_htp_func_symbols_, qnn_htp_backend_);

            // Set Input tensor
            if (bg_info.numGraphInputs) {
              std::vector<Qnn_Tensor_t> inputs;
              for (int g_input_idx = 0; g_input_idx < bg_info.numGraphInputs; g_input_idx++) {
                // 1. Deep copy tensor info from bg_info to mllm context.
                auto g_input_tmp =
                    QnnTensorTransform::instance().deepCopy(&(bg_info.graphInputs[g_input_idx]));

                inputs.emplace_back(g_input_tmp);
              }
              // 2. Assign to mllm's qnn graph.
              g->setupInputsFromBinary(inputs);
            }

            // Set Output tensor
            if (bg_info.numGraphOutputs) {
              std::vector<Qnn_Tensor_t> outputs;
              for (int g_output_idx = 0; g_output_idx < bg_info.numGraphOutputs; g_output_idx++) {
                // 1. Deep copy tensor info from bg_info to mllm context.
                auto g_output_tmp =
                    QnnTensorTransform::instance().deepCopy(&(bg_info.graphOutputs[g_output_idx]));

                outputs.emplace_back(g_output_tmp);
              }
              // 2. Assign to mllm's qnn graph.
              g->setupOutputsFromBinary(outputs);
            }
            break;
          }
          default: {
            MLLM_ERROR_EXIT(kError, "Unsupported binary graph info version");
            return false;
          }
        }
      }

      return true;
    };

    // Create Mllm's QnnGraph in this backend context. Copy metadata to those graphs
    if (nullptr == binary_info) { MLLM_ERROR_EXIT(kError, "Binary info is null"); }
    switch (binary_info->version) {
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
        if (binary_info->contextBinaryInfoV1.graphs) {
          MLLM_RT_ASSERT_EQ(
              copy_graphs_info_from_binary_func(binary_info->contextBinaryInfoV1.graphs,
                                                binary_info->contextBinaryInfoV1.numGraphs),
              true);
        }
        graph_cnt = binary_info->contextBinaryInfoV1.numGraphs;
        break;
      }
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
        if (binary_info->contextBinaryInfoV2.graphs) {
          MLLM_RT_ASSERT_EQ(
              copy_graphs_info_from_binary_func(binary_info->contextBinaryInfoV2.graphs,
                                                binary_info->contextBinaryInfoV2.numGraphs),
              true);
        }
        graph_cnt = binary_info->contextBinaryInfoV2.numGraphs;
        break;
      }
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
        if (binary_info->contextBinaryInfoV3.graphs) {
          MLLM_RT_ASSERT_EQ(
              copy_graphs_info_from_binary_func(binary_info->contextBinaryInfoV3.graphs,
                                                binary_info->contextBinaryInfoV3.numGraphs),
              true);
        }
        graph_cnt = binary_info->contextBinaryInfoV3.numGraphs;
        break;
      }
      default: MLLM_ERROR_EXIT(kError, "Unrecognized system context binary info version.");
    }
  }

  qnn_htp_func_symbols_.qnn_system_interface_.systemContextFree(sys_context_handle);
  sys_context_handle = nullptr;

  qnn_htp_func_symbols_.qnn_interface_.contextCreateFromBinary(
      qnn_htp_backend_.bk_handle_, qnn_htp_backend_.device_handle_,
      (const QnnContext_Config_t**)qnn_htp_backend_.qnn_context_config_,
      reinterpret_cast<void*>(buffer_ptr), file_size, &qnn_htp_backend_.qnn_ctx_handle_,
      qnn_htp_backend_.profile_bk_handle_);

  // FIXME extract profiling numbers if desired

  // Obtain and save graph handles for each graph present in the context based on the saved graph
  // names in the metadata
  MLLM_RT_ASSERT_EQ(qnn_graphs_.size(), graph_cnt);
  for (auto& kv : qnn_graphs_) {
    auto name = kv.first;
    auto g = kv.second;
    qnn_htp_func_symbols_.qnn_interface_.graphRetrieve(qnn_htp_backend_.qnn_ctx_handle_,
                                                       name.c_str(), g->qnnGraphHandlePtr());
  }

  delete[] buffer_ptr;
}

std::shared_ptr<QnnBackend> createQnnBackend() { return std::make_shared<QnnBackend>(); }

}  // namespace mllm::qnn