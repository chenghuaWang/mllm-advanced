#include <random>
#include <benchmark/benchmark.h>
#include "mllm/Backends/X86/Kernels/mem.hpp"
#include "mllm/Backends/X86/Kernels/quants.hpp"
#include "mllm/Backends/X86/Kernels/vec_dot_q4_k_q8_k_avx2.hpp"
#include "mllm/Backends/X86/Kernels/vec_dot_q4_k_q8_k_avx512f.hpp"
#include "mllm/Core/DataTypes.hpp"

static void q4_k_q8_k_avx2_gemv(benchmark::State& state) {
  size_t K = state.range(0);

  void *A, *B;
  mllm::X86::X86_align_alloc(&A, K * sizeof(float));
  mllm::X86::X86_align_alloc(&B, K * sizeof(float));

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

  void *qA, *qB;
  mllm::X86::X86_align_alloc(&qA, K / 256 * sizeof(mllm::block_q4_k_t));
  mllm::X86::X86_align_alloc(&qB, K / 256 * sizeof(mllm::block_q8_k_t));

  mllm::X86::quantize_row_q4_k((mllm::block_q4_k_t*)qA, (float*)A, K);
  mllm::X86::quantize_row_q8_k((mllm::block_q8_k_t*)qB, (float*)B, K);

  for (auto _ : state) {
    float C;
    mllm::X86::vec_dot_q4_k_q8_k_avx2(&C, (mllm::block_q4_k_t*)qA, (mllm::block_q8_k_t*)qB, K);
  }

  mllm::X86::X86_align_free(A);
  mllm::X86::X86_align_free(B);
  mllm::X86::X86_align_free(qA);
  mllm::X86::X86_align_free(qB);
}

#if defined(__AVX512F__)
static void q4_k_q8_k_avx512_gemv(benchmark::State& state) {
  size_t K = state.range(0);

  void *A, *B;
  mllm::X86::X86_align_alloc(&A, K * sizeof(float));
  mllm::X86::X86_align_alloc(&B, K * sizeof(float));

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

  void *qA, *qB;
  mllm::X86::X86_align_alloc(&qA, K / 256 * sizeof(mllm::block_q4_k_t));
  mllm::X86::X86_align_alloc(&qB, K / 256 * sizeof(mllm::block_q8_k_t));

  mllm::X86::quantize_row_q4_k((mllm::block_q4_k_t*)qA, (float*)A, K);
  mllm::X86::quantize_row_q8_k((mllm::block_q8_k_t*)qB, (float*)B, K);

  for (auto _ : state) {
    float C;
    mllm::X86::vec_dot_q4_k_q8_k_avx512f(&C, (mllm::block_q4_k_t*)qA, (mllm::block_q8_k_t*)qB, K);
  }

  mllm::X86::X86_align_free(A);
  mllm::X86::X86_align_free(B);
  mllm::X86::X86_align_free(qA);
  mllm::X86::X86_align_free(qB);
}
#endif

BENCHMARK(q4_k_q8_k_avx2_gemv)->RangeMultiplier(2)->Range(512, 4096);
#if defined(__AVX512F__)
BENCHMARK(q4_k_q8_k_avx512_gemv)->RangeMultiplier(2)->Range(512, 4096);
#endif
BENCHMARK_MAIN();
