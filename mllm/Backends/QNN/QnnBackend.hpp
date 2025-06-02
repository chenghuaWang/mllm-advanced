/**
 * @file QnnBackend.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/Engine/BackendBase.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"

namespace mllm::qnn {

class QnnBackend final : public BackendBase {
 public:
  QnnBackend();

 private:
  QnnDynSymbolLoader loader_;
};

std::shared_ptr<QnnBackend> createQnnBackend();

}  // namespace mllm::qnn