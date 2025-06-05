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
#include "QnnCommon.h"
#include "mllm/Utils/Common.hpp"
#include "mllm/Backends/QNN/Runtime/QnnCommon.hpp"

// Only support android platform.
#if !defined(__ANDROID__)
#error "Only support android platform with NPU Arch: Hexagon."
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
    return (FuncType*)(func_ptr);
  };
};

class QnnDynSymbolLoader {
 public:
  enum DynFlag : int {  // NOLINT performance-enum-size
    kRTLD_NOW = RTLD_NOW,
    kRTLD_LOCAL = RTLD_LOCAL,
    kRTLD_GLOBAL = RTLD_GLOBAL,
  };

  static QnnDynSymbolLoader& instance() {
    static QnnDynSymbolLoader instance;
    return instance;
  }

  ~QnnDynSymbolLoader();

  QnnDynSymbolLoader() = default;

  QnnDynSymbolLoader(const QnnDynSymbolLoader&) = delete;

  QnnDynSymbolLoader& operator=(const QnnDynSymbolLoader&) = delete;

  bool loadQnnDynLib(const std::string& lib_name, int flag);

  inline QnnDynLibDescriptor& operator()(const std::string& lib_name) { return libs_.at(lib_name); }

 private:
  std::unordered_map<std::string, QnnDynLibDescriptor> libs_;
  static const std::vector<std::string> possible_qnn_dyn_lib_paths_;
};

}  // namespace mllm::qnn