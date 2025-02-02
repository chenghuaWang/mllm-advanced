#include <benchmark/benchmark.h>
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/element_wise.hpp"

using namespace mllm::arm;

static void add_f32(benchmark::State& state) {
  size_t size = state.range(0) * state.range(0) * 4;

  for (auto _ : state) {
    state.PauseTiming();
    void *A, *B, *C;
    arm_align_alloc(&A, size, 32);
    arm_align_alloc(&B, size, 32);
    arm_align_alloc(&C, size, 32);
    auto a_ptr = (float*)A;
    auto b_ptr = (float*)B;
    auto c_ptr = (float*)C;
    state.ResumeTiming();
    ew_add_fp32(a_ptr, b_ptr, c_ptr, size / 4);
    state.PauseTiming();
    arm_align_free(A);
    arm_align_free(B);
    arm_align_free(C);
    state.ResumeTiming();
  }
}

static void add_f16(benchmark::State& state) {
  size_t size = state.range(0) * state.range(0) * 2;

  for (auto _ : state) {
    state.PauseTiming();
    void *A, *B, *C;
    arm_align_alloc(&A, size, 32);
    arm_align_alloc(&B, size, 32);
    arm_align_alloc(&C, size, 32);
    auto a_ptr = (float16_t*)A;
    auto b_ptr = (float16_t*)B;
    auto c_ptr = (float16_t*)C;
    state.ResumeTiming();
    ew_add_fp16(a_ptr, b_ptr, c_ptr, size / 2);
    state.PauseTiming();
    arm_align_free(A);
    arm_align_free(B);
    arm_align_free(C);
    state.ResumeTiming();
  }
}

static void add_f32_4_threads(benchmark::State& state) {
  size_t size = state.range(0) * state.range(0) * 4;

  for (auto _ : state) {
    state.PauseTiming();
    void *A, *B, *C;
    arm_align_alloc(&A, size, 32);
    arm_align_alloc(&B, size, 32);
    arm_align_alloc(&C, size, 32);
    auto a_ptr = (float*)A;
    auto b_ptr = (float*)B;
    auto c_ptr = (float*)C;
    state.ResumeTiming();
    ew_add_fp32(a_ptr, b_ptr, c_ptr, size / 4, 4);
    state.PauseTiming();
    arm_align_free(A);
    arm_align_free(B);
    arm_align_free(C);
    state.ResumeTiming();
  }
}

static void add_f16_4_threads(benchmark::State& state) {
  size_t size = state.range(0) * state.range(0) * 2;

  for (auto _ : state) {
    state.PauseTiming();
    void *A, *B, *C;
    arm_align_alloc(&A, size, 32);
    arm_align_alloc(&B, size, 32);
    arm_align_alloc(&C, size, 32);
    auto a_ptr = (float16_t*)A;
    auto b_ptr = (float16_t*)B;
    auto c_ptr = (float16_t*)C;
    state.ResumeTiming();
    ew_add_fp16(a_ptr, b_ptr, c_ptr, size / 2, 4);
    state.PauseTiming();
    arm_align_free(A);
    arm_align_free(B);
    arm_align_free(C);
    state.ResumeTiming();
  }
}

BENCHMARK(add_f32)->Range(64, 2048);
BENCHMARK(add_f32_4_threads)->Range(64, 2048);
BENCHMARK(add_f16)->Range(64, 2048);
BENCHMARK(add_f16_4_threads)->Range(64, 2048);
BENCHMARK_MAIN();