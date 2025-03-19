/**
 * @file SoftmaxTest.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <random>
#include <cmath>
#include "mllm/Utils/Dbg.hpp"
#include "mllm/Backends/X86/Kernels/mem.hpp"
#include "mllm/Backends/CUDA/Ops/OpSelection.hpp"

class SoftmaxTest : public testing::Test {
 protected:
  SoftmaxTest() = default;

  ~SoftmaxTest() override = default;

  void SetShapeAndAlloc(size_t M, size_t N) {
    M_ = M;
    N_ = N;

    mllm::X86::X86_align_alloc(&hZ, M_ * N_ * sizeof(float));
    mllm::X86::X86_align_alloc(&hX, M_ * N_ * sizeof(float));
    mllm::X86::X86_align_alloc(&rhZ, M_ * N_ * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-66.f, 66.f);

    auto rx_ptr = reinterpret_cast<float*>(hX);
    for (int i = 0; i < M_ * N_; ++i) { rx_ptr[i] = dist(gen); }
  }

  void CalculateRef() {
    auto input = static_cast<float*>(hX);
    auto output = static_cast<float*>(rhZ);

    for (int _m = 0; _m < M_; ++_m) {
      float max_val = input[_m * N_];
      for (size_t i = 1; i < N_; ++i) {
        if (input[_m * N_ + i] > max_val) { max_val = input[_m * N_ + i]; }
      }

      float sum = 0.0;
      for (size_t i = 0; i < N_; ++i) {
        output[_m * N_ + i] = expf(input[_m * N_ + i] - max_val);
        sum += output[_m * N_ + i];
      }

      sum = 1.f / sum;

      for (size_t i = 0; i < N_; ++i) { output[_m * N_ + i] *= sum; }
    }
  }

  bool CalculateFp32AndCompare() {
    cudaMalloc(&dZ, M_ * N_ * sizeof(float));
    cudaMalloc(&dX, M_ * N_ * sizeof(float));

    cudaMemcpy(dX, hX, M_ * N_ * sizeof(float), cudaMemcpyHostToDevice);
    mllm::cuda::safe_softmax_fp32(dZ, dX, M_, N_);
    cudaMemcpy(hZ, dZ, M_ * N_ * sizeof(float), cudaMemcpyDeviceToHost);

    bool flag = true;

    auto rz_ptr = static_cast<float*>(rhZ);
    auto z_ptr = static_cast<float*>(hZ);

    for (int i = 0; i < M_ * N_; ++i) {
      const auto imp_value = rz_ptr[i];
      const auto ref_value = z_ptr[i];
      const auto rel_error = std::fabs((imp_value - ref_value));
      if (rel_error > 0.000001F) {
        Dbg(rel_error, i, imp_value, ref_value);
        flag = false;
        break;
      }
    }

    cudaFree(dZ);
    cudaFree(dX);

    return flag;
  }

  void TearDown() override {
    mllm::X86::X86_align_free(hZ);
    mllm::X86::X86_align_free(hX);
    mllm::X86::X86_align_free(rhZ);
  }

  size_t M_ = 1024, N_ = 1024;
  void *hZ = nullptr, *hX = nullptr, *rhZ = nullptr, *dZ = nullptr, *dX = nullptr;
};
