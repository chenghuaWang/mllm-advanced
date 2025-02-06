#include <gtest/gtest.h>
#include <cstdlib>
#include <random>
#include <cmath>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/hgemv.hpp"
#include "mllm/Utils/Dbg.hpp"

class HgemvTest : public testing::Test {
 protected:
  HgemvTest() = default;

  ~HgemvTest() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // A: 1xK
    mllm::arm::arm_align_alloc(&A, 1024 * 2, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&B, N * K * 2, 16);
    // C: 1xN
    mllm::arm::arm_align_alloc(&C, N * 2, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIAS, N * 2, 16);
    // rC: 1xN
    mllm::arm::arm_align_alloc(&rC, N * 2, 16);

    auto a_ptr = reinterpret_cast<float16_t*>(A);
    auto b_ptr = reinterpret_cast<float16_t*>(B);
    auto bias_ptr = reinterpret_cast<float16_t*>(BIAS);
    for (int i = 0; i < K; ++i) { a_ptr[i] = static_cast<float16_t>(0.5f); }
    for (int i = 0; i < N * K; ++i) { b_ptr[i] = static_cast<float16_t>(0.25f); }
    for (int i = 0; i < N; ++i) { bias_ptr[i] = static_cast<float16_t>(1.3f); }
  }

  void CalculateRef() {
    auto a_ptr = reinterpret_cast<float16_t*>(A);
    auto b_ptr = reinterpret_cast<float16_t*>(B);
    auto rc_ptr = reinterpret_cast<float16_t*>(rC);
    auto bias_ptr = reinterpret_cast<float16_t*>(BIAS);
    for (int n = 0; n < N; ++n) {
      rc_ptr[n] = bias_ptr[n];
      for (int k = 0; k < K; ++k) {
        auto acc = static_cast<float16_t>(static_cast<float16_t>(a_ptr[k])
                                          * static_cast<float16_t>(b_ptr[n * K + k]));
        acc = static_cast<float16_t>(static_cast<float16_t>(acc) + rc_ptr[n]);
        rc_ptr[n] = acc;
      }
    }
  }

  void Calculate() {
    mllm::arm::hgemv_1K_NK_V1((float16_t*)A, (float16_t*)B, (float16_t*)BIAS, (float16_t*)C, K, N);
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float16_t*>(C);
    auto rc_ptr = reinterpret_cast<float16_t*>(rC);
    for (int n = 0; n < N; ++n) {
      auto delta = c_ptr[n] - rc_ptr[n];
      if (std::abs(delta) >= 0.0001) {
        Dbg(delta);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(A);
    mllm::arm::arm_align_free(B);
    mllm::arm::arm_align_free(C);
    mllm::arm::arm_align_free(rC);
    mllm::arm::arm_align_free(BIAS);
  }

  size_t K = 1024;
  size_t N = 1024;
  void* BIAS = nullptr;
  void *A = nullptr, *B = nullptr, *C = nullptr;
  void* rC = nullptr;
};

class HgemvHighPrecisionTest : public testing::Test {
 protected:
  HgemvHighPrecisionTest() = default;

  ~HgemvHighPrecisionTest() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // A: 1xK
    mllm::arm::arm_align_alloc(&A, 1024 * 2, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&B, N * K * 2, 16);
    // C: 1xN
    mllm::arm::arm_align_alloc(&C, N * 2, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIAS, N * 2, 16);
    // rC: 1xN
    mllm::arm::arm_align_alloc(&rC, N * 2, 16);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto a_ptr = reinterpret_cast<float16_t*>(A);
    auto b_ptr = reinterpret_cast<float16_t*>(B);
    auto bias_ptr = reinterpret_cast<float16_t*>(BIAS);
    for (int i = 0; i < K; ++i) { a_ptr[i] = static_cast<float16_t>(dist(gen)); }
    for (int i = 0; i < N * K; ++i) { b_ptr[i] = static_cast<float16_t>(dist(gen)); }
    for (int i = 0; i < N; ++i) { bias_ptr[i] = static_cast<float16_t>(dist(gen)); }
  }

  void CalculateRef() {
    auto a_ptr = reinterpret_cast<float16_t*>(A);
    auto b_ptr = reinterpret_cast<float16_t*>(B);
    auto rc_ptr = reinterpret_cast<float16_t*>(rC);
    auto bias_ptr = reinterpret_cast<float16_t*>(BIAS);
    for (int n = 0; n < N; ++n) {
      rc_ptr[n] = bias_ptr[n];
      for (int k = 0; k < K; ++k) {
        auto acc = static_cast<float16_t>(static_cast<float16_t>(a_ptr[k])
                                          * static_cast<float16_t>(b_ptr[n * K + k]));
        acc = static_cast<float16_t>(static_cast<float16_t>(acc) + rc_ptr[n]);
        rc_ptr[n] = acc;
      }
    }
  }

  void Calculate() {
    mllm::arm::hgemv_1K_NK_V2_HP((float16_t*)A, (float16_t*)B, (float16_t*)BIAS, (float16_t*)C, K,
                                 N);
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float16_t*>(C);
    auto rc_ptr = reinterpret_cast<float16_t*>(rC);
    for (int n = 0; n < N; ++n) {
      auto delta = c_ptr[n] - rc_ptr[n];
      if (std::abs(delta) >= 1) {
        Dbg(delta);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(A);
    mllm::arm::arm_align_free(B);
    mllm::arm::arm_align_free(C);
    mllm::arm::arm_align_free(rC);
    mllm::arm::arm_align_free(BIAS);
  }

  size_t K = 1024;
  size_t N = 1024;
  void* BIAS = nullptr;
  void *A = nullptr, *B = nullptr, *C = nullptr;
  void* rC = nullptr;
};
