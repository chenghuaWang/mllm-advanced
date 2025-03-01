/**
 * @file Engine.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

void registerEngineBinding(py::module_& m);