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
#include "mllm/Backends/QNN/QnnBackend.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLog.hpp"

namespace mllm::qnn {

QnnBackend::QnnBackend() : BackendBase(kQNN) {
  QnnLogger::instance();

  MLLM_RT_ASSERT_EQ(initHTPBackend(), true);
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

std::shared_ptr<QnnBackend> createQnnBackend() { return std::make_shared<QnnBackend>(); }

}  // namespace mllm::qnn