#include <benchmark/benchmark.h>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/hgemv.hpp"
#include "mllm/Backends/Arm/Kernels/sgemv.hpp"
#include "mllm/Utils/Log.hpp"

using namespace mllm::arm;

static void hgemv_v1(benchmark::State& state) {
  mllm::Logger::level() = mllm::LogLevel::kError;

  size_t size = state.range(0);

  void *A, *B, *C;
  arm_align_alloc(&A, size * 2, 16);
  arm_align_alloc(&B, size * size * 2, 16);
  arm_align_alloc(&C, size * 2, 16);

  for (auto _ : state) {
    hgemv_1K_NK_V1((float16_t*)A, (float16_t*)B, nullptr, (float16_t*)C, size, size);
  }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
}

static void hgemv_v2_hp(benchmark::State& state) {
  mllm::Logger::level() = mllm::LogLevel::kError;

  size_t size = state.range(0);

  void *A, *B, *C;
  arm_align_alloc(&A, size * 2, 16);
  arm_align_alloc(&B, size * size * 2, 16);
  arm_align_alloc(&C, size * 2, 16);

  for (auto _ : state) {
    hgemv_1K_NK_V2_HP((float16_t*)A, (float16_t*)B, nullptr, (float16_t*)C, size, size);
  }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
}

static void sgemv_v1(benchmark::State& state) {
  size_t size = state.range(0);

  void *A, *B, *C;
  arm_align_alloc(&A, size * 4, 16);
  arm_align_alloc(&B, size * size * 4, 16);
  arm_align_alloc(&C, size * 4, 16);

  for (auto _ : state) { sgemv_1K_NK_V1((float*)A, (float*)B, nullptr, (float*)C, size, size); }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
}

BENCHMARK(hgemv_v1)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(hgemv_v2_hp)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(sgemv_v1)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK_MAIN();
