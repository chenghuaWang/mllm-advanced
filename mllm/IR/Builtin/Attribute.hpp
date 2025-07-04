/**
 * @file Attribute.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <string>
#include "mllm/IR/GeneratedRTTIKind.hpp"
#include "mllm/IR/Node.hpp"
#include "mllm/IR/NodeRTTIClassOfImpl.hpp"

namespace mllm::ir {

class BuiltinIRAttr : public Attr {
 public:
  DEFINE_SPECIFIC_IR_CLASS(BuiltinIRAttr);

  ~BuiltinIRAttr() override;
  BuiltinIRAttr();
  explicit BuiltinIRAttr(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_BUILTINIRATTR_IMPL(node); }
};

// do not mark this as final. Pywrapper needs to inherit this class. Same as classes below.
class IntAttr : public BuiltinIRAttr {
 public:
  DEFINE_SPECIFIC_IR_CLASS(IntAttr);

  ~IntAttr() override;
  IntAttr();
  explicit IntAttr(const NodeKind& kind);

  int& data();

  static self_ptr_t build(IRContext*, int data = 0);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_BUILTINIRATTR_INTATTR_IMPL(node); }

 private:
  int data_ = 0;
};

class FPAttr : public BuiltinIRAttr {
 public:
  DEFINE_SPECIFIC_IR_CLASS(FPAttr);

  ~FPAttr() override;
  FPAttr();
  explicit FPAttr(const NodeKind& kind);

  float& data();

  static self_ptr_t build(IRContext*, float data = 0.f);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_BUILTINIRATTR_FPATTR_IMPL(node); }

 private:
  float data_ = 0.f;
};

class StrAttr : public BuiltinIRAttr {
 public:
  DEFINE_SPECIFIC_IR_CLASS(StrAttr);

  ~StrAttr() override;
  StrAttr();
  explicit StrAttr(const NodeKind& kind);

  std::string& data();

  static self_ptr_t build(IRContext*, const std::string& data);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_BUILTINIRATTR_STRATTR_IMPL(node); }

 private:
  std::string data_;
};

class SymbolAttr : public BuiltinIRAttr {
 public:
  DEFINE_SPECIFIC_IR_CLASS(SymbolAttr);

  ~SymbolAttr() override;
  SymbolAttr();

  std::string& str();

  static self_ptr_t build(IRContext*, const std::string& symbol_name);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_BUILTINIRATTR_SYMBOLATTR_IMPL(node); }

 private:
  std::string data_;
};

class BoolAttr : public BuiltinIRAttr {
 public:
  DEFINE_SPECIFIC_IR_CLASS(BoolAttr);

  ~BoolAttr() override;
  BoolAttr();
  explicit BoolAttr(const NodeKind& kind);

  bool& data();

  static self_ptr_t build(IRContext*, bool data);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_BUILTINIRATTR_BOOLATTR_IMPL(node); }

 private:
  bool data_;
};

}  // namespace mllm::ir