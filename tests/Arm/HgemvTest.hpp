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
    mllm::arm::arm_align_alloc(&A, K * 2, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&B, N * K * 2, 16);
    // C: 1xN
    mllm::arm::arm_align_alloc(&C, N * 2, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIAS, N * 2, 16);

    // A: 1xK
    mllm::arm::arm_align_alloc(&Afp32, K * 4, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&Bfp32, N * K * 4, 16);
    // C: 1xN
    mllm::arm::arm_align_alloc(&Cfp32, N * 4, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIASfp32, N * 4, 16);

    auto a_ptr = reinterpret_cast<float16_t*>(A);
    auto b_ptr = reinterpret_cast<float16_t*>(B);
    auto bias_ptr = reinterpret_cast<float16_t*>(BIAS);

    auto a_fp32_ptr = reinterpret_cast<float*>(Afp32);
    auto b_fp32_ptr = reinterpret_cast<float*>(Bfp32);
    auto bias_fp32_ptr = reinterpret_cast<float*>(BIASfp32);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist;

    for (int i = 0; i < K; ++i) {
      auto tmp = dist(gen);
      a_ptr[i] = static_cast<float16_t>(tmp);
      a_fp32_ptr[i] = tmp;
    }
    for (int i = 0; i < N * K; ++i) {
      auto tmp = dist(gen);
      b_ptr[i] = static_cast<float16_t>(tmp);
      b_fp32_ptr[i] = tmp;
    }
    for (int i = 0; i < N; ++i) {
      auto tmp = dist(gen);
      bias_ptr[i] = static_cast<float16_t>(tmp);
      bias_fp32_ptr[i] = tmp;
    }
  }

  void CalculateRef() {
    auto a_ptr = reinterpret_cast<float*>(Afp32);
    auto b_ptr = reinterpret_cast<float*>(Bfp32);
    auto rc_ptr = reinterpret_cast<float*>(Cfp32);
    auto bias_ptr = reinterpret_cast<float*>(BIASfp32);
    for (int n = 0; n < N; ++n) {
      rc_ptr[n] = bias_ptr[n];
      for (int k = 0; k < K; ++k) { rc_ptr[n] += a_ptr[k] * b_ptr[n * K + k]; }
    }
  }

  void Calculate(int threads = 0) {
    mllm::arm::hgemv_1K_NK_V1((float16_t*)A, (float16_t*)B, (float16_t*)BIAS, (float16_t*)C, K, N,
                              threads);
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float16_t*>(C);
    auto rc_ptr = reinterpret_cast<float*>(Cfp32);
    for (int n = 0; n < N; ++n) {
      const auto imp_value = rc_ptr[n];
      const auto ref_value = c_ptr[n];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.008F) {
        Dbg(rel_error);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(A);
    mllm::arm::arm_align_free(B);
    mllm::arm::arm_align_free(C);
    mllm::arm::arm_align_free(BIAS);

    mllm::arm::arm_align_free(Afp32);
    mllm::arm::arm_align_free(Bfp32);
    mllm::arm::arm_align_free(Cfp32);
    mllm::arm::arm_align_free(BIASfp32);
  }

  size_t K = 1024;
  size_t N = 1024;
  void *BIAS = nullptr, *A = nullptr, *B = nullptr, *C = nullptr;
  void *BIASfp32 = nullptr, *Afp32 = nullptr, *Bfp32 = nullptr, *Cfp32 = nullptr;
};

class HgemvHighPrecisionTest : public testing::Test {
 protected:
  HgemvHighPrecisionTest() = default;

  ~HgemvHighPrecisionTest() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // A: 1xK
    mllm::arm::arm_align_alloc(&A, K * 2, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&B, N * K * 2, 16);
    // C: 1xN
    mllm::arm::arm_align_alloc(&C, N * 2, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIAS, N * 2, 16);

    // A: 1xK
    mllm::arm::arm_align_alloc(&Afp32, K * 4, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&Bfp32, N * K * 4, 16);
    // C: 1xN
    mllm::arm::arm_align_alloc(&Cfp32, N * 4, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIASfp32, N * 4, 16);

    auto a_ptr = reinterpret_cast<float16_t*>(A);
    auto b_ptr = reinterpret_cast<float16_t*>(B);
    auto bias_ptr = reinterpret_cast<float16_t*>(BIAS);

    auto a_fp32_ptr = reinterpret_cast<float*>(Afp32);
    auto b_fp32_ptr = reinterpret_cast<float*>(Bfp32);
    auto bias_fp32_ptr = reinterpret_cast<float*>(BIASfp32);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist;

    for (int i = 0; i < K; ++i) {
      auto tmp = dist(gen);
      a_ptr[i] = static_cast<float16_t>(tmp);
      a_fp32_ptr[i] = tmp;
    }
    for (int i = 0; i < N * K; ++i) {
      auto tmp = dist(gen);
      b_ptr[i] = static_cast<float16_t>(tmp);
      b_fp32_ptr[i] = tmp;
    }
    for (int i = 0; i < N; ++i) {
      auto tmp = dist(gen);
      bias_ptr[i] = static_cast<float16_t>(tmp);
      bias_fp32_ptr[i] = tmp;
    }
  }

  void CalculateRef() {
    auto a_ptr = reinterpret_cast<float*>(Afp32);
    auto b_ptr = reinterpret_cast<float*>(Bfp32);
    auto rc_ptr = reinterpret_cast<float*>(Cfp32);
    auto bias_ptr = reinterpret_cast<float*>(BIASfp32);
    for (int n = 0; n < N; ++n) {
      rc_ptr[n] = bias_ptr[n];
      for (int k = 0; k < K; ++k) { rc_ptr[n] += a_ptr[k] * b_ptr[n * K + k]; }
    }
  }

  void Calculate(int threads = 0) {
    mllm::arm::hgemv_1K_NK_V2_HP((float16_t*)A, (float16_t*)B, (float16_t*)BIAS, (float16_t*)C, K,
                                 N, threads);
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float16_t*>(C);
    auto rc_ptr = reinterpret_cast<float*>(Cfp32);
    for (int n = 0; n < N; ++n) {
      const auto imp_value = rc_ptr[n];
      const auto ref_value = c_ptr[n];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.0008F) {
        Dbg(rel_error);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(A);
    mllm::arm::arm_align_free(B);
    mllm::arm::arm_align_free(C);
    mllm::arm::arm_align_free(BIAS);

    mllm::arm::arm_align_free(Afp32);
    mllm::arm::arm_align_free(Bfp32);
    mllm::arm::arm_align_free(Cfp32);
    mllm::arm::arm_align_free(BIASfp32);
  }

  size_t K = 1024;
  size_t N = 1024;
  void *BIAS = nullptr, *A = nullptr, *B = nullptr, *C = nullptr;
  void *BIASfp32 = nullptr, *Afp32 = nullptr, *Bfp32 = nullptr, *Cfp32 = nullptr;
};
