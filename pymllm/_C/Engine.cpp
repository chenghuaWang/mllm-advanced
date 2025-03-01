/**
 * @file Engine.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-01
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "pymllm/_C/Engine.hpp"
#include "mllm/Engine/Context.hpp"

void registerEngineBinding(py::module_& m) {
  auto engine_m = m.def_submodule("engine");

  py::class_<mllm::MllmEngineCtx>(engine_m, "MllmEngineCtx")
      .def("shutdown", &mllm::MllmEngineCtx::shutdown);

  m.def(
      "get_engine_ctx", []() -> mllm::MllmEngineCtx& { return mllm::MllmEngineCtx::instance(); },
      "get the singleton instance of the engine context", py::return_value_policy::reference);
}
