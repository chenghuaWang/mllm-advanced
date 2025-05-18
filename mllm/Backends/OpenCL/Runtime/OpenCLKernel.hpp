/**
 * @file OpenCLKernel.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-17
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <CL/opencl.hpp>

namespace mllm::opencl {

class OpenCLKernel {
 public:
  using ptr_t = std::shared_ptr<OpenCLKernel>;

  // RUN!
  bool operator()() {
    // TODO
    return true;
  }

  OpenCLKernel& cfgLaunchKernel() {
    // TODO
    return *this;
  }

 private:
  std::vector<std::string> build_options_;
  std::shared_ptr<cl::Kernel> kernel_;
};

}  // namespace mllm::opencl
