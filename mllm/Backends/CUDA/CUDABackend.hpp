/**
 * @file CUDABackend.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Engine/BackendBase.hpp"

namespace mllm::cuda {

class CUDABackend final : public BackendBase {
 public:
  CUDABackend();
};

std::shared_ptr<CUDABackend> createCUDABackend();

}  // namespace mllm::cuda