#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/transpose.hpp"
#include "mllm/Utils/Dbg.hpp"

class TransposeTest : public testing::Test {
 protected:
  TransposeTest() = default;

  ~TransposeTest() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  void SetUp() override {
    mllm::arm::arm_align_alloc(&X, B * S * H * D * 4, 16);
    mllm::arm::arm_align_alloc(&Y, B * S * H * D * 4, 16);
    mllm::arm::arm_align_alloc(&Y_ref, B * S * H * D * 4, 16);

    auto X_ptr = reinterpret_cast<float*>(X);
    auto Y_ptr = reinterpret_cast<float*>(Y);
    auto Y_ref_ptr = reinterpret_cast<float*>(Y_ref);

    for (int i = 0; i < B * S * H * D; ++i) { X_ptr[i] = (float)i; }
  }

  void Calculate(int threads = 0) {
    mllm::arm::transpose_bshd_bhsd((float*)X, (float*)Y, B, S, H, D);
  }

  void CalculateRef() {
    // B, S, H, D
    auto X_ptr = reinterpret_cast<float*>(X);
    // B, H, S, D
    auto Y_ref_ptr = reinterpret_cast<float*>(Y_ref);
    for (int b = 0; b < B; ++b) {
      for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h) {
          for (int d = 0; d < D; ++d) {
            Y_ref_ptr[b * S * H * D + h * S * D + s * D + d] =
                X_ptr[b * S * H * D + s * H * D + h * D + d];
          }
        }
      }
    }
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float*>(Y);
    auto rc_ptr = reinterpret_cast<float*>(Y_ref);
    for (int n = 0; n < B * H * S * D; ++n) {
      const auto imp_value = rc_ptr[n];
      const auto ref_value = c_ptr[n];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.000000001F) {
        Dbg(rel_error);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(X);
    mllm::arm::arm_align_free(Y);
    mllm::arm::arm_align_free(Y_ref);
  }

  int B = 1;
  int S = 1024;
  int H = 12;
  int D = 1536;
  void* X = nullptr;
  void* Y = nullptr;
  void* Y_ref = nullptr;
};

class TransposeHWFp32Test : public testing::Test {
 protected:
  TransposeHWFp32Test() = default;

  ~TransposeHWFp32Test() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  void SetUp() override {
    mllm::arm::arm_align_alloc(&X, H * W * 4, 16);
    mllm::arm::arm_align_alloc(&Y, H * W * 4, 16);
    mllm::arm::arm_align_alloc(&Y_ref, H * W * 4, 16);

    auto X_ptr = reinterpret_cast<float*>(X);
    auto Y_ptr = reinterpret_cast<float*>(Y);
    auto Y_ref_ptr = reinterpret_cast<float*>(Y_ref);

    for (int i = 0; i < H * W; ++i) { X_ptr[i] = (float)i; }
  }

  void Calculate(int threads = 0) { mllm::arm::transpose_hw_wh((float*)X, (float*)Y, H, W); }

  void CalculateRef() {
    // B, S, H, D
    auto X_ptr = reinterpret_cast<float*>(X);
    // B, H, S, D
    auto Y_ref_ptr = reinterpret_cast<float*>(Y_ref);

    for (size_t i = 0; i < H; ++i) {
      for (size_t j = 0; j < W; ++j) { Y_ref_ptr[j * H + i] = X_ptr[i * W + j]; }
    }
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float*>(Y);
    auto rc_ptr = reinterpret_cast<float*>(Y_ref);
    for (int n = 0; n < H * W; ++n) {
      const auto imp_value = rc_ptr[n];
      const auto ref_value = c_ptr[n];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.000000001F) {
        Dbg(rel_error);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(X);
    mllm::arm::arm_align_free(Y);
    mllm::arm::arm_align_free(Y_ref);
  }

  int H = 1536;
  int W = 1536;
  void* X = nullptr;
  void* Y = nullptr;
  void* Y_ref = nullptr;
};

class TransposeHWFp16Test : public testing::Test {
 protected:
  TransposeHWFp16Test() = default;

  ~TransposeHWFp16Test() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  void SetUp() override {
    mllm::arm::arm_align_alloc(&X, H * W * 2, 16);
    mllm::arm::arm_align_alloc(&Y, H * W * 2, 16);
    mllm::arm::arm_align_alloc(&Y_ref, H * W * 2, 16);

    auto X_ptr = reinterpret_cast<float16_t*>(X);
    auto Y_ptr = reinterpret_cast<float16_t*>(Y);
    auto Y_ref_ptr = reinterpret_cast<float16_t*>(Y_ref);

    for (int i = 0; i < H * W; ++i) { X_ptr[i] = (float16_t)i; }
  }

  void Calculate(int threads = 0) {
    mllm::arm::transpose_hw_wh_fp16((float16_t*)X, (float16_t*)Y, H, W);
  }

  void CalculateRef() {
    // B, S, H, D
    auto X_ptr = reinterpret_cast<float16_t*>(X);
    // B, H, S, D
    auto Y_ref_ptr = reinterpret_cast<float16_t*>(Y_ref);

    for (size_t i = 0; i < H; ++i) {
      for (size_t j = 0; j < W; ++j) { Y_ref_ptr[j * H + i] = X_ptr[i * W + j]; }
    }
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float16_t*>(Y);
    auto rc_ptr = reinterpret_cast<float16_t*>(Y_ref);
    for (int n = 0; n < H * W; ++n) {
      const auto imp_value = rc_ptr[n];
      const auto ref_value = c_ptr[n];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.000000001F) {
        Dbg(rel_error);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(X);
    mllm::arm::arm_align_free(Y);
    mllm::arm::arm_align_free(Y_ref);
  }

  int H = 1536;
  int W = 1536;
  void* X = nullptr;
  void* Y = nullptr;
  void* Y_ref = nullptr;
};