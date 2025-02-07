#include <benchmark/benchmark.h>
#include <algorithm>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/softmax.hpp"

using namespace mllm::arm;

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

static void softmax_v1_f32_kxk_threads(benchmark::State& state) {
  size_t size = state.range(0);
  void *X, *Y;
  arm_align_alloc(&X, size * size * 4, 16);
  arm_align_alloc(&Y, size * size * 4, 16);

  std::fill((float*)X, (float*)X + size * size, 66.f);

  for (auto _ : state) {
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; ++i) {
      softmax_V1((float*)X + i * size, (float*)Y + i * size, size, 1);
    }
  }

  arm_align_free(X);
  arm_align_free(Y);
}

BENCHMARK(softmax_v1_f32)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(softmax_v1_f32_kxk)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(softmax_v1_f32_kxk_threads)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK_MAIN();
