/**
 * @file QnnCommon.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// Include common headers that qnn need
#include <QNN/QnnInterface.h>
#include <QNN/System/QnnSystemInterface.h>

// Collection of symbols that we need to load from qnn dyn lib.
struct QnnFuncSymbols {
  using QnnInterfaceGetProvidersFuncType = Qnn_ErrorHandle_t(const QnnInterface_t*** providerList,
                                                             uint32_t* numProviders);
  using QnnSystemInterfaceGetProvidersFuncType =
      Qnn_ErrorHandle_t(const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

  QNN_INTERFACE_VER_TYPE qnn_interface_;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_;
};

// Backend + Device
struct QnnBackendDevice {
  Qnn_LogHandle_t log_ = nullptr;
  Qnn_BackendHandle_t bk_handle_ = nullptr;
  Qnn_DeviceHandle_t device_handle_ = nullptr;
  QnnBackend_Config_t** bk_cfg_ = nullptr;
  QnnContext_Config_t** qnn_context_config_ = nullptr;
  Qnn_ProfileHandle_t profile_bk_handle_ = nullptr;
  Qnn_ContextHandle_t qnn_ctx_handle_;
};
