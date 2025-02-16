/**
 * @file Interface.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/IR/Builtin/Attribute.hpp"
#include "mllm/Utils/RTTIHelper.hpp"

namespace mllm::ir {

template<typename T>
class SymbolInterface {
 public:
  void setSymbolAttr(const std::shared_ptr<SymbolAttr>& symbol) {
    static_cast<T*>(this)->setAttr("symbol", symbol);
  }

  std::shared_ptr<SymbolAttr> getSymbolAttr() {
    return cast<SymbolAttr>(static_cast<T*>(this)->getAttr("symbol"));
  }

  bool hasSymbolAttr() { return static_cast<T*>(this)->getAttr("symbol") != nullptr; }
};

template<typename T>
class NamingInterface {
 public:
  void setName(const std::shared_ptr<StrAttr>& name) {
    static_cast<T*>(this)->setAttr("name", name);
  }

  std::shared_ptr<StrAttr> getName() {
    return cast<StrAttr>(static_cast<T*>(this)->getAttr("name"));
  }
};

}  // namespace mllm::ir