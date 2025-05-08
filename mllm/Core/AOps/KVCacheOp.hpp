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
  enum KVCacheLayoutType {
    kBHSD_REPEAT = 0,     // for eager impl
    kBSHD_NO_REPEAT = 1,  // for flash attn impl
  };

  int32_t heads_num = 1;
  int32_t dim_per_head = 0;
  int32_t head_repeat_times = 1;
  DataTypes cached_elements_dtype = kFp32;
  int32_t pre_alloc_seq_len = 1024;
  int32_t re_alloc_multiplier = 2;
  KVCacheLayoutType layout_type = KVCacheLayoutType::kBHSD_REPEAT;
};

class KVCacheOp : public BaseOp {
 public:
  explicit KVCacheOp(const KVCacheOpCargo& cargo);

  void load(const std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  Tensor cache_;
  KVCacheOpCargo cargo_;
};

}  // namespace mllm