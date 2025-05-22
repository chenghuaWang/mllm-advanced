/**
 * @file OpenCLBackend.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-17
 *
 * @copyright Copyright (c) 2025
 *
 */
#define CL_HPP_TARGET_OPENCL_VERSION 300

#pragma once

#include <memory>
#include "mllm/Engine/BackendBase.hpp"

#ifndef __ANDROID__
#error "The OpenCL backend is only for Android"
#endif

namespace mllm::opencl {

class OpenCLBackend final : public BackendBase {
 public:
  OpenCLBackend();
};

std::shared_ptr<OpenCLBackend> createOpenCLBackend();

}  // namespace mllm::opencl
