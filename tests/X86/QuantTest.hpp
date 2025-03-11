/**
 * @file QuantTest.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <random>
#include <gtest/gtest.h>
#include "mllm/Utils/Dbg.hpp"
#include "mllm/Backends/X86/Kernels/mem.hpp"
#include "mllm/Backends/X86/Kernels/quants.hpp"
#include "mllm/Backends/X86/Kernels/vec_dot_q4_k_q8_k_avx512f.hpp"
#include "mllm/Backends/X86/Kernels/vec_dot_q4_k_q8_k_avx2.hpp"

class Q4KQ8KTest : public testing::Test {
 protected:
  Q4KQ8KTest() = default;

  ~Q4KQ8KTest() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    mllm::X86::X86_align_alloc(&A, K * sizeof(float));
    mllm::X86::X86_align_alloc(&B, K * sizeof(float));
    mllm::X86::X86_align_alloc(&C, 1 * sizeof(float));
    mllm::X86::X86_align_alloc(&rC, 1 * sizeof(float));
    mllm::X86::X86_align_alloc(&rC_avx2, 1 * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist;

    auto a_ptr = reinterpret_cast<float*>(A);
    auto b_ptr = reinterpret_cast<float*>(B);

    for (int i = 0; i < K; ++i) {
      auto tmp = dist(gen);
      a_ptr[i] = tmp;
    }
    for (int i = 0; i < K; ++i) {
      auto tmp = dist(gen);
      b_ptr[i] = tmp;
    }
  }

  void CalculateRef() {
    auto rc_ptr = reinterpret_cast<float*>(rC);
    auto a_ptr = reinterpret_cast<float*>(A);
    auto b_ptr = reinterpret_cast<float*>(B);

    rc_ptr[0] = 0.f;

    for (int i = 0; i < K; ++i) { rc_ptr[0] += a_ptr[i] * b_ptr[i]; }
  }

  void Calculate(int threads = 0) {
    void* quantized_A = nullptr;
    void* quantized_B = nullptr;

    mllm::X86::X86_align_alloc(&quantized_A, (K / 256) * sizeof(mllm::block_q4_k_t));
    mllm::X86::X86_align_alloc(&quantized_B, (K / 256) * sizeof(mllm::block_q8_k_t));

    // do calculate
    mllm::X86::quantize_row_q4_k(static_cast<mllm::block_q4_k_t*>(quantized_A), (float*)A, K);
    mllm::X86::quantize_row_q8_k(static_cast<mllm::block_q8_k_t*>(quantized_B), (float*)B, K);

    mllm::X86::vec_dot_q4_k_q8_k_avx512f((float*)C, static_cast<mllm::block_q4_k_t*>(quantized_A),
                                         static_cast<mllm::block_q8_k_t*>(quantized_B), K);

    mllm::X86::vec_dot_q4_k_q8_k_avx2((float*)rC_avx2,
                                      static_cast<mllm::block_q4_k_t*>(quantized_A),
                                      static_cast<mllm::block_q8_k_t*>(quantized_B), K);

    mllm::X86::X86_align_free(quantized_A);
    mllm::X86::X86_align_free(quantized_B);
  }

  bool Compare() {
    auto c_ptr = (float*)C;
    auto rc_ptr = (float*)rC;
    auto rc_avx2_ptr = (float*)rC_avx2;
    // cmp once to fp32
    {
      const auto imp_value = rc_ptr[0];
      const auto ref_value = c_ptr[0];
      const auto ref_value_2 = rc_avx2_ptr[0];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      const auto rel_error_2 = ref_value_2 != 0 ? std::abs((imp_value - ref_value_2) / ref_value_2)
                                                : std::abs(imp_value);
      if (rel_error > 0.01f) {
        Dbg(imp_value, ref_value, ref_value_2, rel_error);
        return false;
      }
    }
    // cmp once to avx2
    {
      const auto imp_value = rc_avx2_ptr[0];
      const auto ref_value = c_ptr[0];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.01f) {
        Dbg(imp_value, ref_value, rel_error);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::X86::X86_align_free(A);
    mllm::X86::X86_align_free(B);
    mllm::X86::X86_align_free(C);
    mllm::X86::X86_align_free(rC);
    mllm::X86::X86_align_free(rC_avx2);
  }

  size_t K = 2048;
  void *A = nullptr, *B = nullptr, *C = nullptr, *rC = nullptr, *rC_avx2 = nullptr;
};
