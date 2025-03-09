/**
 * @file Pass.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>
#include <memory>
#include "mllm/IR/Node.hpp"

namespace mllm::ir {

enum PassReturnState : uint8_t {
  PASS_RET_SUCCESS = 0x01,
  PASS_RET_FAILURE = 0x02,
  PASS_RET_CONTINUE = 0x80,
};

class Pass {
 public:
  virtual ~Pass() = default;
  Pass() = default;

  // Do not promise the T type is castable with external_data_.
  //
  // NOTE: Pass class will not free external_data_;
  template<typename T>
  T* getExternalData() {
    return (T*)external_data_;
  }

  template<typename T>
  void setExternalData(T* data) {
    external_data_ = (void*)data;
  }

  // the ret should be PassReturnState's binary expression's value
  // E.g.: PASS_RET_SUCCESS | PASS_RET_CONTINUE
  virtual uint8_t run(const node_ptr_t& op);

  void setCtx(const std::shared_ptr<IRContext>& ctx);

  std::shared_ptr<IRContext> getCtx();

 private:
  std::shared_ptr<IRContext> ctx_ = nullptr;
  void* external_data_ = nullptr;
};

typedef std::shared_ptr<Pass> pass_ptr_t;

}  // namespace mllm::ir