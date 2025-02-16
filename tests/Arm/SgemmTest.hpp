#include <gtest/gtest.h>
#include <cstdlib>
#include <random>
#include <cmath>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/sgemm.hpp"
#include "mllm/Utils/Dbg.hpp"

class Sgemm_MK_NK_MN_V1_Test : public testing::Test {
 protected:
  Sgemm_MK_NK_MN_V1_Test() = default;

  ~Sgemm_MK_NK_MN_V1_Test() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // A: MxK
    mllm::arm::arm_align_alloc(&A, M * K * 4, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&B, N * K * 4, 16);
    // C: MxN
    mllm::arm::arm_align_alloc(&C, M * N * 4, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIAS, N * 4, 16);
    // C_ref
    mllm::arm::arm_align_alloc(&C_ref, M * N * 4, 16);

    auto a_ptr = reinterpret_cast<float*>(A);
    auto b_ptr = reinterpret_cast<float*>(B);
    auto bias_ptr = reinterpret_cast<float*>(BIAS);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist;

    for (int i = 0; i < M * K; ++i) {
      auto tmp = dist(gen);
      a_ptr[i] = tmp;
    }
    for (int i = 0; i < N * K; ++i) {
      auto tmp = dist(gen);
      b_ptr[i] = tmp;
    }
    for (int i = 0; i < N; ++i) {
      auto tmp = dist(gen);
      bias_ptr[i] = tmp;
    }
  }

  void CalculateRef() {
    auto a_ptr = reinterpret_cast<float*>(A);
    auto b_ptr = reinterpret_cast<float*>(B);
    auto rc_ptr = reinterpret_cast<float*>(C_ref);
    auto bias_ptr = reinterpret_cast<float*>(BIAS);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        rc_ptr[m * N + n] = bias_ptr[n];
        for (int k = 0; k < K; ++k) { rc_ptr[m * N + n] += a_ptr[m * K + k] * b_ptr[n * K + k]; }
      }
    }
  }

  void Calculate(int threads = 0) {
    mllm::arm::sgemm_mk_nk_mn_V1((float*)A, (float*)B, (float*)C, M, K, N, (float*)BIAS, threads);
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float*>(C);
    auto rc_ptr = reinterpret_cast<float*>(C_ref);
    for (int n = 0; n < N * M; ++n) {
      const auto imp_value = rc_ptr[n];
      const auto ref_value = c_ptr[n];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.0001F) {
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
    mllm::arm::arm_align_free(C_ref);
  }

  size_t M = 1024;
  size_t K = 1024;
  size_t N = 1024;
  void *BIAS = nullptr, *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
};

class Sgemm_MK_KN_MN_V1_Test : public testing::Test {
 protected:
  Sgemm_MK_KN_MN_V1_Test() = default;

  ~Sgemm_MK_KN_MN_V1_Test() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // A: MxK
    mllm::arm::arm_align_alloc(&A, M * K * 4, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&B, N * K * 4, 16);
    // C: MxN
    mllm::arm::arm_align_alloc(&C, M * N * 4, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIAS, N * 4, 16);
    // C_ref
    mllm::arm::arm_align_alloc(&C_ref, M * N * 4, 16);

    auto a_ptr = reinterpret_cast<float*>(A);
    auto b_ptr = reinterpret_cast<float*>(B);
    auto bias_ptr = reinterpret_cast<float*>(BIAS);

    for (int i = 0; i < M * K; ++i) { a_ptr[i] = 0.1; }
    for (int i = 0; i < N * K; ++i) { b_ptr[i] = 0.1; }
    for (int i = 0; i < N; ++i) { bias_ptr[i] = 10.f; }
  }

  void CalculateRef() {
    auto a_ptr = reinterpret_cast<float*>(A);
    auto b_ptr = reinterpret_cast<float*>(B);
    auto rc_ptr = reinterpret_cast<float*>(C_ref);
    auto bias_ptr = reinterpret_cast<float*>(BIAS);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        rc_ptr[m * N + n] = bias_ptr[n];
        for (int k = 0; k < K; ++k) { rc_ptr[m * N + n] += a_ptr[m * K + k] * b_ptr[k * N + n]; }
      }
    }
  }

  void Calculate(int threads = 0) {
    mllm::arm::sgemm_mk_nk_mn_V1((float*)A, (float*)B, (float*)C, M, K, N, (float*)BIAS, threads);
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float*>(C);
    auto rc_ptr = reinterpret_cast<float*>(C_ref);
    for (int n = 0; n < N * M; ++n) {
      const auto imp_value = rc_ptr[n];
      const auto ref_value = c_ptr[n];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.0001F) {
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
    mllm::arm::arm_align_free(C_ref);
  }

  size_t M = 1024;
  size_t K = 1024;
  size_t N = 1024;
  void *BIAS = nullptr, *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
};
