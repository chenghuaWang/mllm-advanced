/**
 * @file Core.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "pymllm/_C/Core.hpp"
#include "mllm/Core/DeviceTypes.hpp"

void registerCoreBinding(py::module_& m) {
  py::enum_<mllm::DeviceTypes>(m, "DeviceTypes")
      .value("CPU", mllm::DeviceTypes::kCPU)
      .value("CUDA", mllm::DeviceTypes::kCUDA)
      .value("OpenCL", mllm::DeviceTypes::kOpenCL);
}