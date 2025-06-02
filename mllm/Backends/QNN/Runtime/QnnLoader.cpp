/**
 * @file QnnLoader.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-06-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"
#include "mllm/Utils/Log.hpp"

namespace mllm::qnn {

const std::vector<std::string> QnnDynSymbolLoader::possible_qnn_dyn_lib_paths_ = {
    "/system/lib64/",
    "/odm/lib64/",
    "/vendor/lib64/",
    "/data/local/tmp/mllm-advanced/bin/",
    "/data/local/tmp/mllm-advanced/lib64/",
};

QnnDynSymbolLoader::~QnnDynSymbolLoader() {
  for (auto& item : libs_) {
    if (item.second.handle_) { dlclose(item.second.handle_); }
  }
}

bool QnnDynSymbolLoader::initHTPBackend() {
  // load qnn backend
  constexpr std::string htp_backend_lib_name = "libQnnHtp.so";

  // GLOBAL Load
  if (!loadQnnDynLib(htp_backend_lib_name, DynFlag::kRTLD_NOW | DynFlag::kRTLD_GLOBAL)) {
    return false;
  }

  // get provider function
  auto qnn_interface_get_providers_func =
      libs_.at(htp_backend_lib_name)
          .func<QnnFuncSymbols::QnnInterfaceGetProvidersFuncType>("QnnInterface_getProviders");

  // get provider
  QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;

  MLLM_RT_ASSERT_EQ(qnn_interface_get_providers_func((const QnnInterface_t***)&interface_providers,
                                                     &num_providers),
                    QNN_SUCCESS);
  MLLM_RT_ASSERT(interface_providers != nullptr);
  MLLM_RT_ASSERT(num_providers != 0);

  bool found_valid_interface = false;

  // get correct provider
  for (size_t provider_id = 0; provider_id < num_providers; provider_id++) {
    if (QNN_API_VERSION_MAJOR == interface_providers[provider_id]->apiVersion.coreApiVersion.major
        && QNN_API_VERSION_MINOR
               <= interface_providers[provider_id]->apiVersion.coreApiVersion.minor) {
      found_valid_interface = true;
      qnn_htp_func_symbols_.qnn_interface_ver_ =
          interface_providers[provider_id]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
  MLLM_RT_ASSERT_EQ(found_valid_interface, true);

  return true;
}

bool QnnDynSymbolLoader::loadQnnDynLib(const std::string& lib_name, int flag) {
  for (auto const& path : possible_qnn_dyn_lib_paths_) {
    auto real_path = path + lib_name;
    auto handle = dlopen(real_path.c_str(), flag);
    if (handle) {
      auto descriptor =
          QnnDynLibDescriptor{.lib_name_ = lib_name, .lib_path_ = path, .handle_ = handle};
      libs_.insert({lib_name, descriptor});
      MLLM_INFO("QnnDynSymbolLoader::loadQnnDynLib {} success.", real_path);
      return true;
    }
  }
  MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib {} failed.", lib_name);
  return false;
}

}  // namespace mllm::qnn
