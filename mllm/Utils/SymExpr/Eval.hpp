/**
 * @file Eval.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-24
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Utils/SymExpr/AST.hpp"
#include <cstdint>
#include <memory>

namespace mllm {

using SymExprDict = std::unordered_map<std::string, float>;

class SymExpr {
 public:
  SymExpr() = default;

  explicit SymExpr(const std::string& src);

  explicit SymExpr(int32_t v);

  explicit SymExpr(float v);

  explicit SymExpr(const std::shared_ptr<SymbolExprAST>& ast);

 public:
  SymExpr ceil();

  SymExpr floor();

  SymExpr abs();

  SymExpr operator+(const SymExpr& rhs) const;

  SymExpr operator-(const SymExpr& rhs) const;

  SymExpr operator*(const SymExpr& rhs) const;

  SymExpr operator/(const SymExpr& rhs) const;

  SymExpr operator%(const SymExpr& rhs) const;

 public:
  [[nodiscard]] bool isStatic() const;

  float eval(const SymExprDict& dict);

  int32_t evalAsInt(const SymExprDict& dict);

  [[nodiscard]] std::string str() const;

  void parse(const std::string& src);

 private:
  std::shared_ptr<SymbolExprAST> ast_ = nullptr;
};

}  // namespace mllm
