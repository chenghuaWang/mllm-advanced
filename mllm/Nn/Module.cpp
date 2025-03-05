/**
 * @file Module.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Nn/Module.hpp"
#include <memory>
#include "mllm/Nn/HierarchyBase.hpp"
#include "mllm/Nn/Layer.hpp"
#include "mllm/Utils/DumpPrinter.hpp"
#include "mllm/Utils/Log.hpp"

namespace mllm::nn {

ModuleImpl::ModuleImpl() : HierarchyBase(HierarchyTypes::kModule) {}

void ModuleImpl::regHierarchy(std::shared_ptr<HierarchyBase> hb) {
  hb->setDepth(depth() + 1);
  reg_hierarchies_.emplace_back(hb);
}

void ModuleImpl::dump(DumpPrinter& printer) {
  printer.print("Module: {}, device={}", absoluteName(), deviceTypes2Str(device()));
  auto _ = DumpPrinterGuard(printer);
  for (auto& hb : reg_hierarchies_) {
    switch (hb->type()) {
      case HierarchyTypes::kModule: std::static_pointer_cast<ModuleImpl>(hb)->dump(printer); break;
      case HierarchyTypes::kLayer: std::static_pointer_cast<LayerImpl>(hb)->dump(printer); break;
    }
  }
}

std::vector<std::shared_ptr<HierarchyBase>>& ModuleImpl::hierarchies() { return reg_hierarchies_; }

void ModuleImpl::load(const std::shared_ptr<ParameterLoader>& ploader) {
  param_loader_ = ploader;
  auto& h = hierarchies();
  for (auto& hb : h) {
    switch (hb->type()) {
      case HierarchyTypes::kModule: std::static_pointer_cast<ModuleImpl>(hb)->load(ploader); break;
      case HierarchyTypes::kLayer: std::static_pointer_cast<LayerImpl>(hb)->load(ploader); break;
    }
  }
}

std::shared_ptr<ParameterLoader> ModuleImpl::params() const { return param_loader_; }

void ModuleImpl::to(DeviceTypes device_type) {
  auto& h = hierarchies();
  for (auto& hb : h) {
    switch (hb->type()) {
      case HierarchyTypes::kModule:
        std::static_pointer_cast<ModuleImpl>(hb)->to(device_type);
        break;
      case HierarchyTypes::kLayer: std::static_pointer_cast<LayerImpl>(hb)->to(device_type); break;
    }
  }
  device_type_ = device_type;
}

Module::Module() { impl_ = std::make_shared<ModuleImpl>(); }

void Module::selfAssignName(const std::string& name) {
  if (!impl_->name().empty() && !impl_->absoluteName().empty()) {
    MLLM_WARN("When doing Module::selfAssignName. Found Module name/absolute_name is not empty. "
              "This Module may not a top level module! Mllm will still reset the "
              "name/absolute_name of this module.")
  }
  impl_->setName(name);
  impl_->setAbsoluteName(name);
}

std::shared_ptr<ModuleImpl> Module::impl() { return impl_; }

Module& Module::to(DeviceTypes device_type) {
  impl()->to(device_type);
  return *this;
}

void Module::print() {
  DumpPrinter p;
  impl_->dump(p);
}

Module& Module::load(const std::shared_ptr<ParameterLoader>& ploader) {
  impl_->load(ploader);
  return *this;
}

}  // namespace mllm::nn