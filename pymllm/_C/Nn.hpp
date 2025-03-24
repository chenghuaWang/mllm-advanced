/**
 * @file Nn.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

void registerNn(py::module_& m);
