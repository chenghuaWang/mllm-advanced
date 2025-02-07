/**
 * @file SoftmaxTest.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <gtest/gtest.h>
#include <cstdlib>
#include <random>
#include <cmath>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/softmax.hpp"
#include "mllm/Utils/Dbg.hpp"

class SoftmaxTest : public testing::Test {
 protected:
  SoftmaxTest() = default;

  ~SoftmaxTest() override = default;

  void SetUp() override {
    mllm::arm::arm_align_alloc(&rXfp32, L * 4, 16);
    mllm::arm::arm_align_alloc(&rYfp32, L * 4, 16);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-66.f, 66.f);

    auto rx_ptr = reinterpret_cast<float*>(rXfp32);
    for (int i = 0; i < L; ++i) { rx_ptr[i] = dist(gen); }
  }

  void CalculateRef() {
    auto input = static_cast<float*>(rXfp32);
    auto output = static_cast<float*>(rYfp32);

    float max_val = input[0];
    for (size_t i = 1; i < L; ++i) {
      if (input[i] > max_val) { max_val = input[i]; }
    }

    float sum = 0.0;
    for (size_t i = 0; i < L; ++i) {
      output[i] = std::expf(input[i] - max_val);
      sum += output[i];
    }

    sum = 1.f / sum;

    for (size_t i = 0; i < L; ++i) { output[i] *= sum; }
  }

  bool CalculateFp32AndCompare() {
    void *X, *Y;
    mllm::arm::arm_align_alloc(&X, L * 4, 16);
    mllm::arm::arm_align_alloc(&Y, L * 4, 16);

    auto x_ptr = reinterpret_cast<float*>(X);
    auto rx_ptr = reinterpret_cast<float*>(rXfp32);
    auto y_ptr = reinterpret_cast<float*>(Y);
    auto ry_ptr = reinterpret_cast<float*>(rYfp32);

    for (int i = 0; i < L; ++i) x_ptr[i] = rx_ptr[i];

    mllm::arm::softmax_V1(x_ptr, y_ptr, L, 1);

    bool flag = true;

    for (int i = 0; i < L; ++i) {
      const auto imp_value = ry_ptr[i];
      const auto ref_value = y_ptr[i];
      const auto rel_error =
          ref_value != 0 ? std::fabs((imp_value - ref_value) / ref_value) : std::fabs(imp_value);
      if (rel_error > 0.0001F) {
        Dbg(rel_error, i, imp_value, ref_value);
        flag = false;
        break;
      }
    }

    mllm::arm::arm_align_free(X);
    mllm::arm::arm_align_free(Y);

    return flag;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(rXfp32);
    mllm::arm::arm_align_free(rYfp32);
  }

  size_t L = 1024;
  void *rXfp32 = nullptr, *rYfp32 = nullptr;
};
