/**
 * @file KVCacheOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstddef>
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/DataTypes.hpp"

namespace mllm {

struct KVCacheOpCargo : public BaseOpCargo<KVCacheOpCargo> {
  size_t heads_num = 1;
  size_t dim_per_head = 0;
  size_t head_repeat_times = 1;
  DataTypes cached_elements_dtype = kFp32;
  size_t pre_alloc_seq_len = 1024;
  size_t re_alloc_multiplier = 2;
};

class KVCacheOp : public BaseOp {
 public:
  explicit KVCacheOp(const KVCacheOpCargo& cargo);

  void load(std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  Tensor cache_;
  KVCacheOpCargo cargo_;
};

}  // namespace mllm