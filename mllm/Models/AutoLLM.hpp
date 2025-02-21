/**
 * @file AutoLLM.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/Core/Tensor.hpp"

namespace mllm::models {

template<typename T>
class AutoLLM {
 public:
  Tensor generate(Tensor inputs) {}

 private:
  std::shared_ptr<T> llm_;
};

}  // namespace mllm::models
