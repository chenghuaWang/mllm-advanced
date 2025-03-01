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
#include "pymllm/_C/Engine.hpp"

PYBIND11_MODULE(_C, m) { registerEngineBinding(m); }
