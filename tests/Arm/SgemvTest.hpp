#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <random>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/sgemv.hpp"
#include "mllm/Utils/Dbg.hpp"

class SgemvTest : public testing::Test {
 protected:
  SgemvTest() = default;

  ~SgemvTest() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // A: 1xK
    mllm::arm::arm_align_alloc(&A, 1024 * 4, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&B, N * K * 4, 16);
    // C: 1xN
    mllm::arm::arm_align_alloc(&C, N * 4, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIAS, N * 4, 16);
    // rC: 1xN
    mllm::arm::arm_align_alloc(&rC, N * 4, 16);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto a_ptr = reinterpret_cast<float*>(A);
    auto b_ptr = reinterpret_cast<float*>(B);
    auto bias_ptr = reinterpret_cast<float*>(BIAS);
    for (int i = 0; i < K; ++i) { a_ptr[i] = dist(gen); }
    for (int i = 0; i < N * K; ++i) { b_ptr[i] = dist(gen); }
    for (int i = 0; i < N; ++i) { bias_ptr[i] = dist(gen); }
  }

  void CalculateRef() {
    auto a_ptr = reinterpret_cast<float*>(A);
    auto b_ptr = reinterpret_cast<float*>(B);
    auto rc_ptr = reinterpret_cast<float*>(rC);
    auto bias_ptr = reinterpret_cast<float*>(BIAS);
    for (int n = 0; n < N; ++n) {
      rc_ptr[n] = bias_ptr[n];
      for (int k = 0; k < K; ++k) { rc_ptr[n] += a_ptr[k] * b_ptr[n * K + k]; }
    }
  }

  void Calculate() {
    mllm::arm::sgemv_1K_NK_V1((float*)A, (float*)B, (float*)BIAS, (float*)C, K, N);
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float*>(C);
    auto rc_ptr = reinterpret_cast<float*>(rC);
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
