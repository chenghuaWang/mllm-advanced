/**
 * @file Nn.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "pymllm/_C/Nn.hpp"
#include "mllm/Nn/HierarchyBase.hpp"
#include "mllm/Nn/Module.hpp"

void registerNn(py::module_& m) {
  py::enum_<mllm::HierarchyTypes>(m, "HierarchyTypes")
      .value("Module", mllm::HierarchyTypes::kModule)
      .value("Layer", mllm::HierarchyTypes::kLayer);

  py::class_<mllm::HierarchyBase, std::shared_ptr<mllm::HierarchyBase>>(m, "HierarchyBase")
      .def("set_name", &mllm::HierarchyBase::setName)
      .def("set_absolute_name", &mllm::HierarchyBase::setAbsoluteName)
      .def("set_depth", &mllm::HierarchyBase::setAbsoluteName)
      .def("depth_increase", &mllm::HierarchyBase::depthIncrease)
      .def("depth_decrease", &mllm::HierarchyBase::depthDecrease)
      .def("name", &mllm::HierarchyBase::name)
      .def("absolute_name", &mllm::HierarchyBase::absoluteName)
      .def("depth", &mllm::HierarchyBase::depth)
      .def("set_compiled_as_obj", &mllm::HierarchyBase::setCompiledAsObj)
      .def("is_compiled_as_obj", &mllm::HierarchyBase::isCompiledAsObj);
}
