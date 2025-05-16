/**
 * @file Generator.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Server/Generator.hpp"

namespace mllm {

void MllmStreamModelGenerator::generate(const Tensor& inputs, int max_length,
                                        const std::function<void(const std::wstring&)>& callback) {
  // The inputs should be 2 dimension and Currently support Int64 only.
  MLLM_RT_ASSERT_EQ(inputs.shape().size(), 2);
  MLLM_RT_ASSERT_EQ(inputs.dtype(), kInt64);

  int cur_length = 0;
  int64_t pass_to_func;

  Tensor tmp = inputs;

  do {
    // input [B, S] output [B, S, Vocab]
    tmp = model_->operator()(tmp)[0];

    // update length
    cur_length += tmp.shape()[1];

    // sampling
    pass_to_func = sample(tmp);

    callback(tokenizer_decode_callback_(pass_to_func));

    // update inputs
    tmp = Tensor::empty({/*batch*/ 1, /*sequence*/ 1}, kInt64, tmp.device()).alloc();
    tmp.ptr<int64_t>()[0] = pass_to_func;
  } while (cur_length < max_length && pass_to_func != eos_token_id_);
}

}  // namespace mllm
