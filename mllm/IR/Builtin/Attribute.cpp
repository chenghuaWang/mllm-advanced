/**
 * @file Attribute.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include "mllm/IR/Builtin/Attribute.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"

namespace mllm::ir {

BuiltinIRAttr::~BuiltinIRAttr() = default;

BuiltinIRAttr::BuiltinIRAttr() : Attr(RK_Attr_BuiltinIRAttr) {}

BuiltinIRAttr::BuiltinIRAttr(const NodeKind& kind) : Attr(kind) {}

IntAttr::~IntAttr() = default;

IntAttr::IntAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_IntAttr) {}

IntAttr::IntAttr(const NodeKind& kind) : BuiltinIRAttr(kind) {}

int& IntAttr::data() { return data_; }

IntAttr::self_ptr_t IntAttr::build(IRContext*, int data) {
  auto ret = std::make_shared<IntAttr>();
  ret->data() = data;
  return ret;
}

FPAttr::~FPAttr() = default;

FPAttr::FPAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_FPAttr) {}

FPAttr::FPAttr(const NodeKind& kind) : BuiltinIRAttr(kind) {}

float& FPAttr::data() { return data_; }

FPAttr::self_ptr_t FPAttr::build(IRContext*, float data) {
  auto ret = std::make_shared<FPAttr>();
  ret->data() = data;
  return ret;
}

StrAttr::~StrAttr() = default;

StrAttr::StrAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_StrAttr) {}

StrAttr::StrAttr(const NodeKind& kind) : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_StrAttr) {}

std::string& StrAttr::data() { return data_; }

StrAttr::self_ptr_t StrAttr::build(IRContext*, const std::string& data) {
  auto ret = std::make_shared<StrAttr>();
  ret->data() = data;
  return ret;
}

SymbolAttr::~SymbolAttr() = default;

SymbolAttr::SymbolAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_SymbolAttr) {}

std::string& SymbolAttr::str() { return data_; }

SymbolAttr::self_ptr_t SymbolAttr::build(IRContext*, const std::string& symbol_name) {
  auto ret = std::make_shared<SymbolAttr>();
  ret->str() = symbol_name;
  return ret;
}

BoolAttr::~BoolAttr() = default;

BoolAttr::BoolAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_BoolAttr) {}

BoolAttr::BoolAttr(const NodeKind& kind) : BuiltinIRAttr(kind) {}

bool& BoolAttr::data() { return data_; }

BoolAttr::self_ptr_t BoolAttr::build(IRContext*, bool data) {
  auto ret = std::make_shared<BoolAttr>();
  ret->data() = data;
  return ret;
}

}  // namespace mllm::ir