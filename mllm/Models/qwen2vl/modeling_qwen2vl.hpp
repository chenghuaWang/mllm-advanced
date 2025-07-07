/**
 * @file modeling_qwen2vl.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Nn/F/F.hpp"
#include "mllm/Nn/Module.hpp"
#include "mllm/Core/Tensor.hpp"

#include "mllm/Models/qwen2vl/configuration_qwen2vl.hpp"

namespace mllm::models {

class PatchEmbed final : public nn::Module {
  int32_t patch_size_ = 14;
  int32_t in_channel_ = 3;
  int32_t embed_dim_ = 1152;
  int32_t temporal_patch_size_ = 2;

 public:
  PatchEmbed() = default;

  inline explicit PatchEmbed(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    // TODO
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    auto hidden_states = inputs[0];

    // [batch_size(1), in_channel(3), temporal_patch_size(2), patch_size(14), patch_size(14)]
    hidden_states =
        hidden_states.view({-1, in_channel_, temporal_patch_size_, patch_size_, patch_size_});

    // TODO Conv3D

    return {};
  }
};

class PatchMerger final : public nn::Module {
 public:
  PatchMerger() = default;

  inline explicit PatchMerger(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    // TODO
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // TODO ConvTranspose3D

    return {};
  }
};

class VisionMlp final : public nn::Module {
 public:
  VisionMlp() = default;

  inline explicit VisionMlp(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    // TODO
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override { return {}; }
};

class VisionAttention final : public nn::Module {
 public:
  VisionAttention() = default;

  inline explicit VisionAttention(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    // TODO
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override { return {}; }
};

class Qwen2VLVisionBlock final : public nn::Module {
 public:
  Qwen2VLVisionBlock() = default;

  inline explicit Qwen2VLVisionBlock(const std::string& name, const Qwen2VLConfig& cfg) {
    selfAssignName(name);
    // TODO
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override { return {}; }
};

}  // namespace mllm::models
