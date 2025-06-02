/**
 * @file QnnLoader.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <dlfcn.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/QNN/Runtime/QnnCommon.hpp"

// Only support android platform.
// The gpu arch we support is :
//  - Adreno(Fully Tested)
//  - Mali(Partially Tested)
#if !defined(__ANDROID__)
#error "Only support android platform with GPU Arch: Adreno and Mali."
#endif

namespace mllm::qnn {

struct QnnDynLibDescriptor {
  std::string lib_name_;
  std::string lib_path_;
  void* handle_ = nullptr;

  template<typename FuncType>
  std::function<FuncType> func(const std::string& symbol_name) {
    if (handle_ == nullptr) { MLLM_ERROR_EXIT(kError, "QnnDynSymbolLoader: handle is nullptr."); }
    auto func_ptr = dlsym(handle_, symbol_name.c_str());
    MLLM_RT_ASSERT(func_ptr != nullptr);
    return reinterpret_cast<FuncType>(func_ptr);
  };
};

class QnnDynSymbolLoader {
 public:
  enum DynFlag : int {  // NOLINT performance-enum-size
    kRTLD_NOW = RTLD_NOW,
    kRTLD_LOCAL = RTLD_LOCAL,
    kRTLD_GLOBAL = RTLD_GLOBAL,
  };

  // Collection of symbols that we need to load from qnn dyn lib.
  struct QnnFuncSymbols {
    using QnnInterfaceGetProvidersFuncType = Qnn_ErrorHandle_t(const QnnInterface_t*** providerList,
                                                               uint32_t* numProviders);
    using QnnSystemInterfaceGetProvidersFuncType =
        Qnn_ErrorHandle_t(const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

    QNN_INTERFACE_VER_TYPE qnn_interface_ver_;
    QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_ver_;
  };

  static QnnDynSymbolLoader& instance() {
    static QnnDynSymbolLoader instance;
    return instance;
  }

  ~QnnDynSymbolLoader();

  QnnDynSymbolLoader() = default;

  QnnDynSymbolLoader(const QnnDynSymbolLoader&) = delete;

  QnnDynSymbolLoader& operator=(const QnnDynSymbolLoader&) = delete;

  bool initHTPBackend();

  bool loadQnnDynLib(const std::string& lib_name, int flag);

  inline QnnDynLibDescriptor& operator()(const std::string& lib_name) { return libs_.at(lib_name); }

  inline QnnFuncSymbols& htpFuncSymbols() { return qnn_htp_func_symbols_; }

 private:
  QnnFuncSymbols qnn_htp_func_symbols_;
  std::unordered_map<std::string, QnnDynLibDescriptor> libs_;
  static const std::vector<std::string> possible_qnn_dyn_lib_paths_;
};

}  // namespace mllm::qnn