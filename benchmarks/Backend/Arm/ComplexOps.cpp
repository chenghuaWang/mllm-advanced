#include <benchmark/benchmark.h>
#include <algorithm>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/softmax.hpp"
#include "mllm/Backends/Arm/Kernels/transpose.hpp"

using namespace mllm::arm;

static void softmax_baseline(benchmark::State& state) {
  size_t size = state.range(0);
  void *X, *Y;
  arm_align_alloc(&X, size * 4, 16);
  arm_align_alloc(&Y, size * 4, 16);

  std::fill((float*)X, (float*)X + size, 66.f);

  for (auto _ : state) {
    auto input = static_cast<float*>(X);
    auto output = static_cast<float*>(Y);

    float max_val = input[0];
    for (size_t i = 1; i < size; ++i) {
      if (input[i] > max_val) { max_val = input[i]; }
    }

    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
      output[i] = std::expf(input[i] - max_val);
      sum += output[i];
    }

    sum = 1.f / sum;

    for (size_t i = 0; i < size; ++i) { output[i] *= sum; }
  }

  arm_align_free(X);
  arm_align_free(Y);
}

static void softmax_v1_f32(benchmark::State& state) {
  size_t size = state.range(0);
  void *X, *Y;
  arm_align_alloc(&X, size * 4, 16);
  arm_align_alloc(&Y, size * 4, 16);

  std::fill((float*)X, (float*)X + size, 66.f);

  for (auto _ : state) { softmax_V1((float*)X, (float*)Y, size, 1); }

  arm_align_free(X);
  arm_align_free(Y);
}

static void softmax_v1_f32_kxk(benchmark::State& state) {
  size_t size = state.range(0);
  void *X, *Y;
  arm_align_alloc(&X, size * size * 4, 16);
  arm_align_alloc(&Y, size * size * 4, 16);

  std::fill((float*)X, (float*)X + size * size, 66.f);

  for (auto _ : state) {
    for (int i = 0; i < size; ++i) {
      softmax_V1((float*)X + i * size, (float*)Y + i * size, size, 1);
    }
  }

  arm_align_free(X);
  arm_align_free(Y);
}

static void softmax_v1_f32_kxk_4_threads(benchmark::State& state) {
  size_t size = state.range(0);
  void *X, *Y;
  arm_align_alloc(&X, size * size * 4, 16);
  arm_align_alloc(&Y, size * size * 4, 16);

  std::fill((float*)X, (float*)X + size * size, 66.f);

  for (auto _ : state) {
#pragma omp parallel for num_threads(4) schedule(auto)
    for (int i = 0; i < size; ++i) {
      softmax_V1((float*)X + i * size, (float*)Y + i * size, size, 1);
    }
  }

  arm_align_free(X);
  arm_align_free(Y);
}

static void softmax_v1_f16(benchmark::State& state) {
  size_t size = state.range(0);
  void *X, *Y;
  arm_align_alloc(&X, size * 2, 16);
  arm_align_alloc(&Y, size * 2, 16);

  std::fill((float16_t*)X, (float16_t*)X + size, 66.f);

  for (auto _ : state) { hsoftmax_V1((float16_t*)X, (float16_t*)Y, size, 1); }

  arm_align_free(X);
  arm_align_free(Y);
}

static void softmax_v1_f16_kxk(benchmark::State& state) {
  size_t size = state.range(0);
  void *X, *Y;
  arm_align_alloc(&X, size * size * 2, 16);
  arm_align_alloc(&Y, size * size * 2, 16);

  std::fill((float16_t*)X, (float16_t*)X + size * size, 66.f);

  for (auto _ : state) {
    for (int i = 0; i < size; ++i) {
      hsoftmax_V1((float16_t*)X + i * size, (float16_t*)Y + i * size, size, 1);
    }
  }

  arm_align_free(X);
  arm_align_free(Y);
}

static void softmax_v1_f16_kxk_4_threads(benchmark::State& state) {
  size_t size = state.range(0);
  void *X, *Y;
  arm_align_alloc(&X, size * size * 2, 16);
  arm_align_alloc(&Y, size * size * 2, 16);

  std::fill((float16_t*)X, (float16_t*)X + size * size, 66.f);

  for (auto _ : state) {
#pragma omp parallel for num_threads(4) schedule(auto)
    for (int i = 0; i < size; ++i) {
      hsoftmax_V1((float16_t*)X + i * size, (float16_t*)Y + i * size, size, 1);
    }
  }

  arm_align_free(X);
  arm_align_free(Y);
}

static void transpose_fp32_bshd2bhsd(benchmark::State& state) {
  int B = 1;
  int S = 1024;
  int H = 12;
  int D = 1536;
  void *X, *Y;
  arm_align_alloc(&X, B * S * H * D * 4, 16);
  arm_align_alloc(&Y, B * S * H * D * 4, 16);

  for (auto _ : state) { transpose_bshd_bhsd((float*)X, (float*)Y, B, S, H, D); }

  arm_align_free(X);
  arm_align_free(Y);
}

BENCHMARK(softmax_baseline)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(softmax_v1_f32)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(softmax_v1_f32_kxk)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(softmax_v1_f32_kxk_4_threads)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(softmax_v1_f16)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(softmax_v1_f16_kxk)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(softmax_v1_f16_kxk_4_threads)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(transpose_fp32_bshd2bhsd);
BENCHMARK_MAIN();
