/**
 * @file PyWarp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <pybind11/pybind11.h>
#include "pymllm/_C/Core.hpp"
#include "pymllm/_C/Engine.hpp"

PYBIND11_MODULE(_C, m) {
  registerCoreBinding(m);
  registerBaseBackend(m);
  registerX86Backend(m);
  registerEngineBinding(m);
}
