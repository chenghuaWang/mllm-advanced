#include <gtest/gtest.h>
#include <cstdlib>
#include <random>
#include <cmath>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/fa2_mma0.hpp"
#include "mllm/Utils/Dbg.hpp"

class FA2Mma0Test : public testing::Test {
 protected:
  FA2Mma0Test() = default;

  ~FA2Mma0Test() override = default;

  void SetUp() override {
    // Query: BSHD
    mllm::arm::arm_align_alloc((void**)(&query), B * Br * H * D * sizeof(float16_t), 16);
    mllm::arm::arm_align_alloc((void**)(&key), B * Bc * H * D * sizeof(float16_t), 16);
    mllm::arm::arm_align_alloc((void**)(&refW), Br * Br * H * sizeof(float), 16);
    mllm::arm::arm_align_alloc((void**)(&W), Br * Br * H * sizeof(float), 16);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < B * Br * H * D; ++i) { query[i] = static_cast<float16_t>(dist(gen)); }
    for (size_t i = 0; i < B * Bc * H * D; ++i) { key[i] = static_cast<float16_t>(dist(gen)); }
    for (size_t i = 0; i < Br * Br * H; ++i) { refW[i] = 0.0f; }
    for (size_t i = 0; i < Br * Br * H; ++i) { W[i] = 0.0f; }
  }

  void TearDown() override {
    mllm::arm::arm_align_free(query);
    mllm::arm::arm_align_free(key);
    mllm::arm::arm_align_free(refW);
    mllm::arm::arm_align_free(W);
  }

  void CalculateRef() {
    // Query is Br * D
    // Key is Bc * D
    // Output is Br * Bc
    for (size_t h = 0; h < H; ++h) {
      for (size_t r = 0; r < Br; ++r) {
        for (size_t c = 0; c < Br; ++c) {
          float sum = 0.0f;
          for (size_t k = 0; k < D; ++k) {
            sum += static_cast<float>(query[r * D + k]) * static_cast<float>(key[c * D + k]);
          }
          refW[h * Br * Br + r * Br + c] = sum;
        }
      }
    }
  }

  void Calculate() {
    mllm::arm::fa2_mma0_bshd_fp16_br4_bc4_neon_asm_micro_kernel(query, key, W, /*dim_size=*/D,
                                                                /*stride_q=*/D, /*stride_k=*/D,
                                                                /*stride_acc*/ 4);
  }

  void Compare() {
    for (size_t i = 0; i < Br * Br * H; ++i) { EXPECT_NEAR(W[i], refW[i], 1e-5f); }
  }

  size_t B = 1;
  size_t Br = 4;
  size_t Bc = 4;
  size_t H = 1;
  size_t D = 1024;
  float16_t* query = nullptr;
  float16_t* key = nullptr;
  float* refW = nullptr;
  float* W = nullptr;
};
