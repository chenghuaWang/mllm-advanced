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
#include "mllm/Backends/X86/X86Backend.hpp"
#include "mllm/Engine/BackendBase.hpp"
#include "mllm/Engine/Context.hpp"

void registerBaseBackend(py::module_& m) {
  py::class_<mllm::BackendBase, std::shared_ptr<mllm::BackendBase>>(m, "BackendBase");
}

void registerX86Backend(py::module_& m) {
  py::class_<mllm::X86::X86Backend, mllm::BackendBase, std::shared_ptr<mllm::X86::X86Backend>>(
      m, "X86Backend");

  m.def("create_x86_backend", &mllm::X86::createX86Backend);
}

void registerEngineBinding(py::module_& m) {
  py::class_<mllm::MemManager, std::shared_ptr<mllm::MemManager>>(m, "MemManager")
      .def("init_buddy_ctx", &mllm::MemManager::initBuddyCtx)
      .def("init_oc", &mllm::MemManager::initOC);

  py::class_<mllm::MllmEngineCtx, std::shared_ptr<mllm::MllmEngineCtx>>(m, "MllmEngineCtx")
      .def("shutdown", &mllm::MllmEngineCtx::shutdown)
      .def("mem", &mllm::MllmEngineCtx::mem)
      .def("register_backend", &mllm::MllmEngineCtx::registerBackend);

  m.def(
      "get_engine_ctx", []() -> mllm::MllmEngineCtx& { return mllm::MllmEngineCtx::instance(); },
      "get the singleton instance of the engine context", py::return_value_policy::reference);
}
