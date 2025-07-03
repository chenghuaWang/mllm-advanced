/**
 * @file KVCacheOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/KVCacheOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm::arm {

ArmKVCacheOp::ArmKVCacheOp(const KVCacheOpCargo& cargo) : KVCacheOp(cargo) {
  cur_kv_cache_limits_ = cargo_.pre_alloc_seq_len;

  switch (cargo_.layout_type) {
    case KVCacheOpCargo::KVCacheLayoutType::kBHSD_REPEAT:
      // init cache: [B, H * repeat_times, S, D]
      cache_ = Tensor::empty({1, cargo_.head_repeat_times * cargo_.heads_num, cur_kv_cache_limits_,
                              cargo_.dim_per_head},
                             cargo_.cached_elements_dtype, kCPU)
                   .setMemType(kNormal)
                   .alloc();
      break;
    case KVCacheOpCargo::KVCacheLayoutType::kBSHD_NO_REPEAT:
      // init cache: [B, S, H, D]
      cache_ = Tensor::empty({1, cur_kv_cache_limits_, cargo_.heads_num, cargo_.dim_per_head},
                             cargo_.cached_elements_dtype, kCPU)
                   .setMemType(kNormal)
                   .alloc();
      break;
  }

  cur_kv_cache_seq_len_ = 0;
}

void ArmKVCacheOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing.
}

void ArmKVCacheOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  switch (cargo_.layout_type) {
    case KVCacheOpCargo::KVCacheLayoutType::kBHSD_REPEAT: {
      // inputs is [B, H, S, D]
      // cache is [B, H, S, D]

      auto t = inputs[0];
      auto shape = t.shape();
      auto B = shape[0];
      auto S = shape[2];
      auto D = shape[3];

      // Parallel Copy has no speedup on current big.LITTLE Arch(8 Gen1, Gen3, ...).
      // copy data from inputs[0] to cache_
      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < cargo_.heads_num; ++h) {
          for (int s = 0; s < S; ++s) {
            for (int h_rep = 0; h_rep < cargo_.head_repeat_times; ++h_rep) {
              std::memcpy(cache_.offsettedRawPtr({b, h * cargo_.head_repeat_times + h_rep,
                                                  cur_kv_cache_seq_len_ + s, 0}),
                          t.offsettedRawPtr({b, h, s, 0}), D * dataTypeSize(t.dtype()));
            }
          }
        }
      }

      cur_kv_cache_seq_len_ += S;
      break;
    }
    case KVCacheOpCargo::KVCacheLayoutType::kBSHD_NO_REPEAT: {
      auto t = inputs[0];
      auto shape = t.shape();
      auto B = shape[0];
      auto S = shape[1];
      auto D = shape[3];

      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
          for (int h = 0; h < cargo_.heads_num; ++h) {
            std::memcpy(cache_.offsettedRawPtr({b, cur_kv_cache_seq_len_ + s, h, 0}),
                        t.offsettedRawPtr({b, s, h, 0}), D * dataTypeSize(t.dtype()));
          }
        }
      }

      cur_kv_cache_seq_len_ += S;
      break;
    }
  }
}

void ArmKVCacheOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  switch (cargo_.layout_type) {
    case KVCacheOpCargo::KVCacheLayoutType::kBHSD_REPEAT: {
      MLLM_RT_ASSERT_EQ(inputs.size(), 1);
      auto shape = inputs[0].shape();
      MLLM_RT_ASSERT_EQ(shape.size(), 4);

      // We assumed the inputs[0] is [B, H, S, D]
      auto B = shape[0];
      auto S = shape[2];
      auto D = shape[3];

      MLLM_RT_ASSERT_EQ(B, 1);
      MLLM_RT_ASSERT_EQ(D, cargo_.dim_per_head);

      // We need to realloc kv cache
      while (cur_kv_cache_seq_len_ + S > cur_kv_cache_limits_) {
        auto new_kv_cache_limits = cur_kv_cache_limits_ * cargo_.re_alloc_multiplier;

        MLLM_WARN("Trying to expand KVCache from {} to {}", cur_kv_cache_limits_,
                  new_kv_cache_limits);

        auto new_cache_ = Tensor::empty({1, cargo_.head_repeat_times * cargo_.heads_num,
                                         new_kv_cache_limits, cargo_.dim_per_head},
                                        cargo_.cached_elements_dtype, kCPU)
                              .setMemType(kGlobal)
                              .alloc();

        // copy data in [B, H, S, D] format
        auto old_cache_shape = cache_.shape();
        auto old_cache_B = old_cache_shape[0];
        auto old_cache_H = old_cache_shape[1];
        auto old_cache_S = old_cache_shape[2];
        auto old_cache_D = old_cache_shape[3];

        // Parallel Copy has no speedup on current big.LITTLE Arch(8 Gen1, Gen3, ...).
        for (int b = 0; b < old_cache_B; ++B) {
          for (int h = 0; h < old_cache_H; ++h) {
            for (int s = 0; s < old_cache_S; ++s) {
              std::memcpy(new_cache_.offsettedRawPtr({b, h, s, 0}),
                          cache_.offsettedRawPtr({b, h, s, 0}),
                          old_cache_D * dataTypeSize(cache_.dtype()));
            }
          }
        }

        cache_ = new_cache_;
        cur_kv_cache_limits_ = new_kv_cache_limits;
      }
      break;
    }
    case KVCacheOpCargo::KVCacheLayoutType::kBSHD_NO_REPEAT: {
      MLLM_RT_ASSERT_EQ(inputs.size(), 1);
      auto shape = inputs[0].shape();
      MLLM_RT_ASSERT_EQ(shape.size(), 4);

      // We assumed the inputs[0] is [B, S, H, D]
      auto B = shape[0];
      auto S = shape[1];
      auto D = shape[3];

      MLLM_RT_ASSERT_EQ(B, 1);
      MLLM_RT_ASSERT_EQ(D, cargo_.dim_per_head);

      // We need to realloc kv cache
      while (cur_kv_cache_seq_len_ + S > cur_kv_cache_limits_) {
        auto new_kv_cache_limits = cur_kv_cache_limits_ * cargo_.re_alloc_multiplier;

        MLLM_WARN("Trying to expand KVCache from {} to {}", cur_kv_cache_limits_,
                  new_kv_cache_limits);

        auto new_cache_ =
            Tensor::empty({1, new_kv_cache_limits, cargo_.heads_num, cargo_.dim_per_head},
                          cargo_.cached_elements_dtype, kCPU)
                .setMemType(kGlobal)
                .alloc();

        // copy data in [B, H, S, D] format
        auto old_cache_shape = cache_.shape();
        auto old_cache_B = old_cache_shape[0];
        auto old_cache_S = old_cache_shape[1];
        auto old_cache_H = old_cache_shape[2];
        auto old_cache_D = old_cache_shape[3];

        // Parallel Copy has no speedup on current big.LITTLE Arch(8 Gen1, Gen3, ...).
        for (int b = 0; b < old_cache_B; ++B) {
          for (int s = 0; s < old_cache_S; ++s) {
            for (int h = 0; h < old_cache_H; ++h) {
              std::memcpy(new_cache_.offsettedRawPtr({b, s, h, 0}),
                          cache_.offsettedRawPtr({b, s, h, 0}),
                          old_cache_D * dataTypeSize(cache_.dtype()));
            }
          }
        }

        cache_ = new_cache_;
        cur_kv_cache_limits_ = new_kv_cache_limits;
      }
      break;
    }
  }
}

void ArmKVCacheOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  switch (cargo_.layout_type) {
    case KVCacheOpCargo::KVCacheLayoutType::kBHSD_REPEAT: {
      // We assumed the inputs[0] is [B, H, S, D]
      auto s = inputs[0].shape()[2];
      // B, H, S, D
      auto o1 = cache_[{{}, {}, {kAll, (int32_t)(cur_kv_cache_seq_len_ + s)}, {}}];
      outputs.emplace_back(o1);
      break;
    }
    case KVCacheOpCargo::KVCacheLayoutType::kBSHD_NO_REPEAT: {
      // We assumed the inputs[0] is [B, S, H, D]
      auto s = inputs[0].shape()[1];
      auto o1 = cache_[{{}, {kAll, (int32_t)(cur_kv_cache_seq_len_ + s)}, {}, {}}];
      outputs.emplace_back(o1);
      break;
    }
  }
}

}  // namespace mllm::arm
