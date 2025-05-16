/**
 * @file Generator.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <functional>
#include <numeric>
#include <random>
#include <half/half.hpp>
#include <string>
#include "mllm/Nn/Module.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

template<typename T>
T sampleElement(const std::vector<T>& elements, const std::vector<float>& probabilities) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
  size_t index = dist(gen);
  return elements[index];
}

class MllmStreamModelGenerator {
  float top_k_ = 50;
  float top_p_ = 0.9;
  nn::Module* model_;
  std::function<Tensor(const std::string& str)> tokenizer_encode_callback_;
  std::function<std::wstring(int64_t)> tokenizer_decode_callback_;
  std::function<std::string(const std::vector<std::pair<std::string, std::string>>)>
      build_prompt_callback_;

 public:
  int64_t eos_token_id_ = -1;

  enum class SamplingStrategy {
    kGreedy,
    kTopK,
    kTopP,
  };

  explicit MllmStreamModelGenerator(nn::Module* model) : model_(model) {}

  void generate(const Tensor& inputs, int max_length,
                const std::function<void(const std::wstring&)>& callback);

  inline void setTokenizerEncodeCallback(
      const std::function<Tensor(const std::string& str)>& callback) {
    tokenizer_encode_callback_ = callback;
  }

  inline void setTokenizerDecodeCallback(const std::function<std::wstring(int64_t)>& callback) {
    tokenizer_decode_callback_ = callback;
  }

  inline void setBuildPromptCallback(
      const std::function<std::string(const std::vector<std::pair<std::string, std::string>>)>&
          callback) {
    build_prompt_callback_ = callback;
  }

  inline void setStrategy(SamplingStrategy strategy) { strategy_ = strategy; }

  inline void setTopK(float top_k) { top_k_ = top_k; }

  inline void setTopP(float top_p) { top_p_ = top_p; }

  inline void setModel(nn::Module* model) { model_ = model; }

  inline Tensor encode(const std::string& str) { return tokenizer_encode_callback_(str); }

  inline std::string buildPrompt(const std::vector<std::pair<std::string, std::string>>& prompts) {
    return build_prompt_callback_(prompts);
  }

 private:
  int64_t sample(const Tensor& inputs) {
    auto S = inputs.shape()[1];
    auto D = inputs.shape()[2];
    switch (inputs.dtype()) {
      case kFp32: {
        return sampleImpl<float>(inputs.ptr<float>() + (S - 1) * D, D);
      }
      case kFp16: {
        return sampleImpl<half_float::half>(inputs.ptr<half_float::half>() + (S - 1) * D, D);
      }
      default: NYI("Type Not Supported"); break;
    }
    return 0;
  }

  template<typename T>
  int64_t sampleImpl(T* ptr, int D) {
    std::vector<std::pair<float, int>> sorted_probs(D);
    for (int i = 0; i < D; ++i) { sorted_probs[i] = {static_cast<float>(ptr[i]), i}; }
    std::sort(sorted_probs.begin(), sorted_probs.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
              });

    switch (strategy_) {
      case SamplingStrategy::kGreedy: return sorted_probs[0].second;
      case SamplingStrategy::kTopK: {
        // Top-K Sampling
        std::vector<std::pair<float, int>> top_k_probs(
            sorted_probs.begin(), sorted_probs.begin() + static_cast<size_t>(top_k_));
        std::vector<float> probs(top_k_probs.size());
        std::transform(top_k_probs.begin(), top_k_probs.end(), probs.begin(),
                       [](const std::pair<float, int>& p) { return p.first; });
        std::vector<int> indices(top_k_probs.size());
        std::transform(top_k_probs.begin(), top_k_probs.end(), indices.begin(),
                       [](const std::pair<float, int>& p) { return p.second; });

        // Normalize probabilities
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum > 0.0f) {
          for (auto& prob : probs) { prob /= sum; }
        }

        return sampleElement(indices, probs);
      }
      case SamplingStrategy::kTopP: {
        // Top-P Sampling
        float cumulative_prob = 0.0f;
        std::vector<std::pair<float, int>> nucleus_probs;
        for (auto& sorted_prob : sorted_probs) {
          cumulative_prob += sorted_prob.first;
          nucleus_probs.emplace_back(sorted_prob.first, sorted_prob.second);
          if (cumulative_prob >= top_p_) { break; }
        }

        std::vector<float> probs(nucleus_probs.size());
        std::transform(nucleus_probs.begin(), nucleus_probs.end(), probs.begin(),
                       [](const std::pair<float, int>& p) { return p.first; });
        std::vector<int> indices(nucleus_probs.size());
        std::transform(nucleus_probs.begin(), nucleus_probs.end(), indices.begin(),
                       [](const std::pair<float, int>& p) { return p.second; });

        // Normalize probabilities
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum > 0.0f) {
          for (auto& prob : probs) { prob /= sum; }
        }

        return sampleElement(indices, probs);
      }
      default: NYI("Sampling Strategy Not Supported");
    }
    return 0;
  }

  SamplingStrategy strategy_ = SamplingStrategy::kGreedy;
};

}  // namespace mllm