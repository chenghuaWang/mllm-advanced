#include <benchmark/benchmark.h>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/sgemm.hpp"
#include "mllm/Utils/Log.hpp"

using namespace mllm::arm;

static void sgemm_v1(benchmark::State& state) {
  mllm::Logger::level() = mllm::LogLevel::kError;

  size_t size = state.range(0);

  void *A, *B, *C, *BIAS;
  arm_align_alloc(&A, size * size * 4, 16);
  arm_align_alloc(&B, size * size * 4, 16);
  arm_align_alloc(&C, size * size * 4, 16);
  arm_align_alloc(&BIAS, size * 4, 16);

  for (auto _ : state) {
    sgemm_mk_nk_mn_V1((float*)A, (float*)B, (float*)C, size, size, size, (float*)BIAS);
  }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
  arm_align_free(BIAS);
}

static void sgemm_v1_4_threads(benchmark::State& state) {
  mllm::Logger::level() = mllm::LogLevel::kError;

  size_t size = state.range(0);

  void *A, *B, *C, *BIAS;
  arm_align_alloc(&A, size * size * 4, 16);
  arm_align_alloc(&B, size * size * 4, 16);
  arm_align_alloc(&C, size * size * 4, 16);
  arm_align_alloc(&BIAS, size * 4, 16);

  for (auto _ : state) {
    sgemm_mk_nk_mn_V1((float*)A, (float*)B, (float*)C, size, size, size, (float*)BIAS, 4);
  }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
  arm_align_free(BIAS);
}

static void sgemm_v1_128x1536x8960_4_threads(benchmark::State& state) {
  mllm::Logger::level() = mllm::LogLevel::kError;

  int M = 128;
  int K = 1536;
  int N = 8960;

  void *A, *B, *C, *BIAS;
  arm_align_alloc(&A, M * K * 4, 16);
  arm_align_alloc(&B, N * K * 4, 16);
  arm_align_alloc(&C, M * N * 4, 16);
  arm_align_alloc(&BIAS, N * 4, 16);

  for (auto _ : state) {
    sgemm_mk_nk_mn_V1((float*)A, (float*)B, (float*)C, M, K, N, (float*)BIAS, 4);
  }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
  arm_align_free(BIAS);
}

static void sgemm_mk_kn_v1(benchmark::State& state) {
  mllm::Logger::level() = mllm::LogLevel::kError;

  size_t size = state.range(0);

  void *A, *B, *C, *BIAS;
  arm_align_alloc(&A, size * size * 4, 16);
  arm_align_alloc(&B, size * size * 4, 16);
  arm_align_alloc(&C, size * size * 4, 16);
  arm_align_alloc(&BIAS, size * 4, 16);

  for (auto _ : state) {
    sgemm_mk_kn_mn_V1((float*)A, (float*)B, (float*)C, size, size, size, (float*)BIAS);
  }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
  arm_align_free(BIAS);
}

static void sgemm_mk_kn_v1_4_threads(benchmark::State& state) {
  mllm::Logger::level() = mllm::LogLevel::kError;

  size_t size = state.range(0);

  void *A, *B, *C, *BIAS;
  arm_align_alloc(&A, size * size * 4, 16);
  arm_align_alloc(&B, size * size * 4, 16);
  arm_align_alloc(&C, size * size * 4, 16);
  arm_align_alloc(&BIAS, size * 4, 16);

  for (auto _ : state) {
    sgemm_mk_kn_mn_V1((float*)A, (float*)B, (float*)C, size, size, size, (float*)BIAS, 4);
  }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
  arm_align_free(BIAS);
}

static void sgemm_mk_kn_v1_128x1536x8960_4_threads(benchmark::State& state) {
  mllm::Logger::level() = mllm::LogLevel::kError;

  int M = 128;
  int K = 1536;
  int N = 8960;

  void *A, *B, *C, *BIAS;
  arm_align_alloc(&A, M * K * 4, 16);
  arm_align_alloc(&B, N * K * 4, 16);
  arm_align_alloc(&C, M * N * 4, 16);
  arm_align_alloc(&BIAS, N * 4, 16);

  for (auto _ : state) {
    sgemm_mk_kn_mn_V1((float*)A, (float*)B, (float*)C, M, K, N, (float*)BIAS, 4);
  }

  arm_align_free(A);
  arm_align_free(B);
  arm_align_free(C);
  arm_align_free(BIAS);
}

BENCHMARK(sgemm_v1)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(sgemm_v1_4_threads)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(sgemm_v1_128x1536x8960_4_threads);
BENCHMARK(sgemm_mk_kn_v1)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(sgemm_mk_kn_v1_4_threads)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(sgemm_mk_kn_v1_128x1536x8960_4_threads);
BENCHMARK_MAIN();
