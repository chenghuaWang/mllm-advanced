/**
 * @file Module.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/Tensor.hpp"
#include "mllm/Nn/HierarchyBase.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Utils/DumpPrinter.hpp"
#include <memory>
#include <vector>
#include <string>

namespace mllm::nn {

class ModuleImpl : public HierarchyBase {
 public:
  ModuleImpl();

  void regHierarchy(std::shared_ptr<HierarchyBase> hb);

  void dump(DumpPrinter& printer);

 private:
  std::vector<std::shared_ptr<HierarchyBase>> reg_hierarchies_;
};

class Module {
 public:
  Module();

  void selfAssginName(const std::string& name);

  std::shared_ptr<ModuleImpl> impl();

  template<typename T, typename... Args>
  auto reg(const std::string& name, Args&&... args) {
    auto ret = T(std::forward<Args>(args)...);
    impl_->regHierarchy(ret.impl());
    ret.impl()->setAbsoluteName(impl_->absoluteName() + "." + name);
    // TODO
    // register layer's op to thisThread() op table.
    return ret;
  }

  template<typename... Args>
  std::vector<Tensor> operator()(Args&&... args) {
    std::vector<Tensor> inputs = {std::forward<Args>(args)...};

    // check inputs first
    for (auto& t : inputs) {
      if (t.memType() == TensorMemTypes::kExtraInput && t.name().empty()) {
        MLLM_ERROR_EXIT(kError,
                        "The inputs to nn::module is `ExtraInput` but its name is empty. You may "
                        "need to use t.setName().setMemType() when initializing the tensor.");
      }
    }

    return forward(inputs);
  }

  void print();

  virtual std::vector<Tensor> forward(std::vector<Tensor>& inputs) = 0;

 private:
  std::shared_ptr<ModuleImpl> impl_ = nullptr;
};

}  // namespace mllm::nn
