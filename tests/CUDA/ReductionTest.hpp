/**
 * @file ReductionTest.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <limits>
#include <random>
#include <cmath>
#include "mllm/Utils/Dbg.hpp"
#include "mllm/Backends/X86/Kernels/mem.hpp"
#include "mllm/Backends/CUDA/Ops/OpSelection.hpp"

class Reduce1D : public testing::Test {
 protected:
  enum ReductionType {
    kAdd,
    kMax,
    kMin,
    kMul,
  };

  Reduce1D() = default;

  ~Reduce1D() override = default;

  void SetShapeAndAlloc(size_t N) {
    N_ = N;

    mllm::X86::X86_align_alloc(&hZ, 4 * sizeof(float));
    mllm::X86::X86_align_alloc(&hX, N_ * sizeof(float));
    mllm::X86::X86_align_alloc(&rhZ, 4 * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-66.f, 66.f);

    auto rx_ptr = reinterpret_cast<float*>(hX);
    for (int i = 0; i < N_; ++i) { rx_ptr[i] = dist(gen); }
  }

  void CalculateRef(ReductionType t) {
    auto input = static_cast<float*>(hX);
    auto output = static_cast<float*>(rhZ);

    float ans;
    switch (t) {
      case kAdd: ans = 0.F; break;
      case kMul: ans = 1.F; break;
      case kMax: ans = std::numeric_limits<float>::lowest(); break;
      case kMin: ans = std::numeric_limits<float>::max(); break;
    }
    for (int i = 0; i < N_; ++i) {
      switch (t) {
        case kAdd: ans = ans + input[i]; break;
        case kMul: ans = ans * input[i]; break;
        case kMax: ans = ans > input[i] ? ans : input[i]; break;
        case kMin: ans = ans < input[i] ? ans : input[i]; break;
      }
    }

    output[0] = ans;
  }

  bool CalculateFp32AndCompare(ReductionType t) {
    cudaMalloc(&dZ, 4 * sizeof(float));
    cudaMalloc(&dX, N_ * sizeof(float));

    cudaMemcpy(dX, hX, N_ * sizeof(float), cudaMemcpyHostToDevice);
    switch (t) {
      case kAdd: mllm::cuda::array_reduce_sum_fp32(dZ, dX, N_); break;
      case kMul: break;
      case kMax: break;
      case kMin: break;
    }

    cudaMemcpy(hZ, dZ, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    bool flag = true;

    auto rz_ptr = static_cast<float*>(rhZ);
    auto z_ptr = static_cast<float*>(hZ);

    const auto imp_value = rz_ptr[0];
    const auto ref_value = z_ptr[0];
    const auto rel_error =
        ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
    if (rel_error > 0.0001F) {
      Dbg(rel_error, imp_value, ref_value);
      flag = false;
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

  size_t N_ = 1024;
  void *hZ = nullptr, *hX = nullptr, *rhZ = nullptr, *dZ = nullptr, *dX = nullptr;
};
