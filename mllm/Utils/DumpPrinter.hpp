/**
 * @file Dumpper.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>

#include "fmt/core.h"

namespace mllm {

class DumpPrinter {
 public:
  explicit DumpPrinter(int32_t indent = 4);

  void inc();

  void dec();

  template<typename... Args>
  inline void print(Args&&... args) {
    for (int i = 0; i < depth_ * indent_; ++i) fmt::print(" ");
    fmt::println(std::forward<Args>(args)...);
  }

 private:
  int32_t depth_ = 0;
  int32_t indent_ = 4;
};

class DumpPrinterGuard {
 public:
  explicit DumpPrinterGuard(DumpPrinter& printer);
  ~DumpPrinterGuard();

 private:
  DumpPrinter& printer_;
};

}  // namespace mllm