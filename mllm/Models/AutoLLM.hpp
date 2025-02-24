/**
 * @file AutoLLM.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/Core/Tensor.hpp"
#include "mllm/Utils/Common.hpp"
#include <functional>
#include <numeric>
#include <random>

namespace mllm::models {

template<typename T>
T _sample_element(const std::vector<T>& elements, const std::vector<float>& probabilities) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
  size_t index = dist(gen);
  return elements[index];
}

template<typename T>
class AutoLLM {
 public:
  template<typename... Args>
  AutoLLM(Args&&... args) {
    llm_ = std::make_shared<T>(std::forward<Args>(args)...);
  }

  void generate(Tensor inputs, int max_length, long eos_token, const std::function<void(long)> func,
                float top_p = 0.9) {
    // The inputs should be 2 dimension and Currently support Int64 only.
    MLLM_RT_ASSERT_EQ(inputs.shape().size(), 2);
    MLLM_RT_ASSERT_EQ(inputs.dtype(), kInt64);

    int cur_length = 0;
    long pass_to_func;

    Tensor tmp = inputs;

    do {
      // input [B, S] output [B, S, Vocab]
      tmp = llm_->operator()(tmp)[0];

      // update length
      cur_length += tmp.shape()[1];

      // sampling
      pass_to_func = _greedy_search(tmp);
      func(pass_to_func);

      // update inputs
      tmp = Tensor::empty({/*batch*/ 1, /*sequence*/ 1}, kInt64, tmp.device()).alloc();
      tmp.ptr<int64_t>()[0] = pass_to_func;
    } while (cur_length < max_length || pass_to_func != eos_token);
  }

  std::shared_ptr<T> model() { return llm_; }

 private:
  long _greedy_search(Tensor inputs) {
    auto S = inputs.shape()[1];
    auto D = inputs.shape()[2];
    switch (inputs.dtype()) {
      case kFp32: {
        auto _ptr = inputs.ptr<float>() + (S - 1) * D;
        float max = -1e10f;
        long pos = 0;
        for (int i = 0; i < D; ++i) {
          auto value = _ptr[i];
          if (value > max) {
            max = value;
            pos = i;
          }
        }
        return pos;
        break;
      }
      case kFp16: {
        auto _ptr = inputs.ptr<__fp16>() + (S - 1) * D;
        __fp16 max = -60000;
        long pos = 0;
        for (int i = 0; i < D; ++i) {
          auto value = _ptr[i];
          if (value > max) {
            max = value;
            pos = i;
          }
        }
        return pos;
        break;
      }
      default: NYI("Type Not Supported"); break;
    }
    return 0;
  }

  long _top_p_sampling(Tensor inputs, float top_p) {
    auto argmax = [](const std::vector<float>& vec) -> unsigned int {
      return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
    };
    std::vector<std::pair<float, unsigned int>> scores;

    auto S = inputs.shape()[1];
    auto D = inputs.shape()[2];
    auto _ptr = inputs.ptr<float>() + (S - 1) * D;
    for (int i = 0; i < D; ++i) {
      auto value = _ptr[i];
      scores.push_back(std::make_pair(value, i));
    }

    std::sort(scores.begin(), scores.end(),
              [](std::pair<float, unsigned int> a, std::pair<float, unsigned int> b) {
                return a.first > b.first;
              });
    std::vector<float> top_k_elements;
    std::vector<unsigned int> top_k_elements_idx;

    if (scores[0].first > 1.f) {
      MLLM_ERROR_EXIT(
          kError, "The input tensor t should go through softmax first.(0.f - 1.f is acceptable)");
    }

    float p = 0.f;
    size_t idx = 0;
    while (p < 5) {
      top_k_elements.emplace_back(scores[idx].first);
      top_k_elements_idx.emplace_back(scores[idx].second);
      p += scores[idx].first;
      idx++;
    }

    if (top_k_elements.size() == 1) { return top_k_elements_idx[0]; }

    // softmax with temperature
    std::vector<float> softmax(top_k_elements.size(), 0.f);
    double max_logit = top_k_elements[argmax(top_k_elements)];
    double sum_exp = 0.f;

    for (size_t i = 0; i < top_k_elements.size(); ++i) {
      softmax[i] = exp((top_k_elements[i] - max_logit) / top_p);
      sum_exp += softmax[i];
    }

    for (float& value : softmax) { value /= sum_exp; }

    // sampling
    float _sum = std::accumulate(softmax.begin(), softmax.end(), 0.0);
    for (float& value : softmax) { value /= _sum; }

    auto ret = _sample_element(top_k_elements_idx, softmax);
    return ret;
  }

  long _top_k_sampling(Tensor inputs) { NYI("Top K Sampling"); }

  std::shared_ptr<T> llm_;
};

}  // namespace mllm::models
