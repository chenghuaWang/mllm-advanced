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

namespace mllm::qnn {

QnnBackend::QnnBackend() : BackendBase(kQNN) { loader_.initHTPBackend(); }

std::shared_ptr<QnnBackend> createQnnBackend() { return std::make_shared<QnnBackend>(); }

}  // namespace mllm::qnn