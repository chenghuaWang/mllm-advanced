// The test is highly inspired by kleidiai's example.
// ref:
// https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/examples/matmul_clamp_f32_qai8dxp_qsi4c32p/matmul_clamp_f32_qai8dxp_qsi4c32p.cpp

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Dbg.hpp"
#include "mllm/Utils/ThreadPool.hpp"
#include "mllm/Backends/Arm/Kernels/mem.hpp"
#include "mllm/Backends/Arm/Kernels/kai_linear.hpp"

class KaiLinear_fp16_fp16_fp16p_mxk_kxn_Test : public testing::Test {
 protected:
  KaiLinear_fp16_fp16_fp16p_mxk_kxn_Test() = default;

  ~KaiLinear_fp16_fp16_fp16p_mxk_kxn_Test() override = default;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // A: 1xK
    mllm::arm::arm_align_alloc(&A, K * 2, 16);
    // B: NxK
    mllm::arm::arm_align_alloc(&B, N * K * 2, 16);
    // C: 1xN
    mllm::arm::arm_align_alloc(&C, N * 2, 16);
    // BIAS: 1xN
    mllm::arm::arm_align_alloc(&BIAS, N * 2, 16);

    // C: 1xN
    mllm::arm::arm_align_alloc(&Cfp16, N * 2, 16);

    auto a_ptr = reinterpret_cast<float16_t*>(A);
    auto b_ptr = reinterpret_cast<float16_t*>(B);
    auto bias_ptr = reinterpret_cast<float16_t*>(BIAS);

    for (int i = 0; i < K; ++i) { a_ptr[i] = float16_t(i * 0.1); }
    for (int i = 0; i < N * K; ++i) { b_ptr[i] = float16_t(i * 0.1); }
    for (int i = 0; i < N; ++i) { bias_ptr[i] = float16_t(i * 10); }

    mllm::arm::arm_align_alloc(&PackedB, kai_linear.pack_rhs_size(K, N) * 2, 16);
    kai_linear.pack_rhs_offline((float16_t*)PackedB, (float16_t*)B, (float16_t*)BIAS, K, N);
  }

  void CalculateRef() {
    auto m = 1;
    auto k = K;
    auto n = N;
    auto scalar_min = -FLT_MAX;
    auto scalar_max = FLT_MAX;
    auto lhs = reinterpret_cast<float16_t*>(A);
    auto rhs = reinterpret_cast<float16_t*>(B);
    auto dst = reinterpret_cast<float16_t*>(Cfp16);
    auto bias = reinterpret_cast<float16_t*>(BIAS);
    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
      for (size_t col_idx = 0; col_idx < n; ++col_idx) {
        float16_t acc = bias[col_idx];

        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
          acc += lhs[row_idx * k + k_idx] * rhs[col_idx + n * k_idx];
        }
        acc = std::max(acc, static_cast<float16_t>(scalar_min));
        acc = std::min(acc, static_cast<float16_t>(scalar_max));

        dst[row_idx * n + col_idx] = acc;
      }
    }
  };

  void Calculate(int threads = 0) {
    using mllm::MllmThreadPool;
    MLLM_THREAD_POOL_INIT(threads);
    kai_linear.matmul((float16_t*)C, (float16_t*)A, (float16_t*)PackedB, 1, K, N);
  }

  bool Compare() {
    auto c_ptr = reinterpret_cast<float16_t*>(C);
    auto rc_ptr = reinterpret_cast<float16_t*>(Cfp16);
    for (int n = 0; n < N; ++n) {
      const auto imp_value = rc_ptr[n];
      const auto ref_value = c_ptr[n];
      const auto rel_error = std::fabs(imp_value - ref_value);
      if (rel_error > 0.0001F) {
        Dbg(n, rel_error);
        return false;
      }
    }
    return true;
  }

  void TearDown() override {
    mllm::arm::arm_align_free(A);
    mllm::arm::arm_align_free(B);
    mllm::arm::arm_align_free(C);
    mllm::arm::arm_align_free(BIAS);
    mllm::arm::arm_align_free(Cfp16);
    mllm::arm::arm_align_free(PackedB);
  }

  mllm::arm::KaiLinear_fp16_fp16_fp16p_mxk_kxn kai_linear;

  size_t K = 1024;
  size_t N = 1024;
  void *BIAS = nullptr, *A = nullptr, *B = nullptr, *C = nullptr;
  void* Cfp16 = nullptr;
  void* PackedB = nullptr;
};

class KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_Test : public testing::Test {
 protected:
  KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_Test() = default;

  ~KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_Test() override = default;

  inline uint16_t cast_bf16_f32(float f32) {
    uint16_t bf16 = 0;
#ifdef __ARM_FEATURE_BF16
    __asm__ __volatile__("bfcvt %h[output], %s[input]" : [output] "=w"(bf16) : [input] "w"(f32));
#else
    const uint32_t* i32 = (uint32_t*)(&f32);
    bf16 = (*i32 >> 16);
#endif
    return bf16;
  }

  inline float cast_f32_bf16(uint16_t bf16) {
    const uint32_t i32 = (bf16 << 16);
    float f32 = 0;
    memcpy(&f32, &i32, sizeof(i32));
    return f32;
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void fill_uniform_random(size_t num_rows, size_t num_cols, float* dst, size_t seed) {
    std::srand(seed);

    // Fill the array with random values between -1 and 1
    for (size_t i = 0; i < num_rows * num_cols; i++) {
      dst[i] = (float)((double)std::rand() / RAND_MAX) * 2 - 1;
    }
  }

  void ref_quant_qa8dx_f32(size_t m, size_t k, const float* lhs_f32, int8_t* lhs_qa8dx) {
    const size_t dst_stride = (k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t));

    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
      const float* src_ptr = lhs_f32 + row_idx * k;

      float max0 = -FLT_MAX;
      float min0 = FLT_MAX;

      // Find min/max for each channel
      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        const float src0_0 = src_ptr[k_idx];

        max0 = std::max(src0_0, max0);
        min0 = std::min(src0_0, min0);
      }

      // Maximum/minimum int8 values
      const float qmin = (float)INT8_MIN;
      const float qmax = (float)INT8_MAX;

      const float rmin0 = std::min(0.0f, min0);
      const float rmax0 = std::max(0.0f, max0);

      const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

      // Reciprocal to quantize
      const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

      const float descaled_min0 = rmin0 * scale0;
      const float descaled_max0 = rmax0 * scale0;

      const float zero_point_from_min_error0 = qmin + descaled_min0;
      const float zero_point_from_max_error0 = qmax + descaled_max0;

      float zero_point0 = zero_point_from_min_error0 + zero_point_from_max_error0 > 0
                              ? qmin - descaled_min0
                              : qmax - descaled_max0;

      zero_point0 = std::max(zero_point0, qmin);
      zero_point0 = std::min(zero_point0, qmax);

      // Round to nearest integer
      const int32_t nudged_zero_point0 = lrintf(zero_point0);

      int8_t* dst_ptr = (int8_t*)lhs_qa8dx + row_idx * dst_stride;

      // LHS offset at the beginning of the row
      *((float*)(dst_ptr)) = recip_scale0;
      dst_ptr += sizeof(float);
      *((int32_t*)(dst_ptr)) = -nudged_zero_point0;
      dst_ptr += sizeof(int32_t);

      // Quantize the channels
      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        const float src0_0 = src_ptr[k_idx];

        // Scale the values
        int32_t v0_s32 = (int32_t)(round(src0_0 * scale0));

        v0_s32 = v0_s32 + nudged_zero_point0;
        v0_s32 = std::max(v0_s32, INT8_MIN);
        v0_s32 = std::min(v0_s32, INT8_MAX);
        dst_ptr[0] = (int8_t)v0_s32;
        dst_ptr += sizeof(int8_t);
      }
    }
  }

  void quant_nxk_qs4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32,
                            uint8_t* rhs_qs4c32, uint16_t* rhs_scales_bf16) {
    const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
    const size_t rhs_qs4c32_stride = get_rhs_native_stride(k);

    // Make sure the output is filled with zeros
    std::memset(rhs_qs4c32, 0, n * rhs_qs4c32_stride);

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
      const float* src_ptr = rhs_f32 + row_idx * k;

      for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
        float amax = 0.0f;
        float max = 0.0f;

        for (size_t b = 0; b < bl; ++b) {
          const size_t k_idx = block_idx * bl + b;

          if (k_idx >= k) { break; }

          const float src0_0 = src_ptr[k_idx];
          const float asrc0_0 = fabsf(src0_0);

          if (amax < asrc0_0) {
            amax = asrc0_0;
            max = src0_0;
          }
        }

        const float scale = max / -8.0;
        const float recip_scale = scale ? 1.0f / scale : 0.0f;

        // Store the scale in the dedicated buffer
        *rhs_scales_bf16 = cast_bf16_f32(scale);

        rhs_scales_bf16 += 1;

        for (size_t i = 0; i < bl; ++i) {
          const size_t k_idx = block_idx * bl + i;

          if (k_idx >= k) { break; }

          const float src0_0 = src_ptr[k_idx];

          // Scale the values
          int32_t v0_s32 = (int32_t)(round(src0_0 * recip_scale));

          // Maximum/minimum int4 values
          v0_s32 = std::max(v0_s32, -8);
          v0_s32 = std::min(v0_s32, 7);

          const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

          const size_t dst_addr = (k_idx / 2) + row_idx * rhs_qs4c32_stride;
          uint8_t rhs_v0 = rhs_qs4c32[dst_addr];

          if ((k_idx % 2) == 0) {
            rhs_v0 = v0_u8;
          } else {
            rhs_v0 |= (v0_u8 << 4);
          }

          rhs_qs4c32[dst_addr] = rhs_v0;
        }
      }
    }
  }

  inline size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

  inline size_t get_num_blocks_per_row(size_t k, size_t bl) { return roundup(k, bl) / bl; }

  inline size_t get_rhs_native_stride(size_t x) { return roundup(x, 2) / 2; }

  inline size_t get_rhs_scale_stride(size_t k, size_t bl) {
    const size_t num_blocks_per_row = get_num_blocks_per_row(k, bl);
    return num_blocks_per_row * sizeof(uint16_t);
  }

  void ref_matmul_mxn_mxk_nxk_f32_qa8dx_qs4c32(size_t m, size_t n, size_t k, size_t bl,
                                               const int8_t* lhs_qa8dx, const uint8_t* rhs_qs4c32,
                                               const uint16_t* scale_bf16, float* dst_f32,
                                               float scalar_min, float scalar_max) {
    const size_t num_blocks_row = get_num_blocks_per_row(k, bl);

    const size_t lhs_stride = k + sizeof(float) + sizeof(int32_t);
    const size_t rhs_stride = get_rhs_native_stride(k);

    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
      const int8_t* lhs_ptr_start = lhs_qa8dx + row_idx * lhs_stride;

      for (size_t col_idx = 0; col_idx < n; ++col_idx) {
        // Main f32 accumulator
        float main_acc = 0.0f;

        const int8_t* lhs_ptr = lhs_ptr_start;
        const uint8_t* rhs_ptr = rhs_qs4c32 + col_idx * rhs_stride;

        // Get the LHS quantization parameters stored at the
        // beginning of each row
        const float lhs_scale = *(const float*)lhs_ptr;
        lhs_ptr += sizeof(float);

        const int32_t lhs_offset = *(const int32_t*)lhs_ptr;
        lhs_ptr += sizeof(int32_t);

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
          const uint16_t rhs_scale_bf16 = scale_bf16[block_idx + col_idx * num_blocks_row];
          const float rhs_scale = cast_f32_bf16(rhs_scale_bf16);

          int32_t iacc = 0;

          for (size_t i = 0; i < bl; ++i) {
            const size_t k_idx = block_idx * bl + i;

            if (k_idx >= k) { break; }

            // Get the LHS values
            const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

            // Get the RHS values
            const uint8_t rhs_byte = rhs_ptr[0];

            // Unpack the RHS values
            int32_t rhs_v0 = 0;
            if ((k_idx % 2) == 0) {
              rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
            } else {
              rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
            }

            iacc += lhs_v0 * rhs_v0;
            iacc += lhs_offset * rhs_v0;

            lhs_ptr += 1;

            // Increment only when k_idx is not a multiple of 2
            rhs_ptr += k_idx % 2;
          }

          main_acc += iacc * rhs_scale;
        }

        main_acc = main_acc * lhs_scale;

        // Clamp (min-max) operation
        main_acc = std::max(main_acc, scalar_min);
        main_acc = std::min(main_acc, scalar_max);

        dst_f32[0] = main_acc;
        dst_f32 += 1;
      }
    }
  };

  bool Compare(int threads = 0) {
    using mllm::MllmThreadPool;
    MLLM_THREAD_POOL_INIT(threads);

    const size_t seed_lhs = 4568;
    const size_t seed_rhs = seed_lhs + 4;

    const size_t lhs_native_size_f32 = M * K * sizeof(float);
    const size_t rhs_native_size_f32 = N * K * sizeof(float);
    const size_t rhs_native_size_qs4c32 = N * get_rhs_native_stride(K);
    const size_t rhs_scales_size_bf16 = N * get_rhs_scale_stride(K, bl);

    uint8_t* lhs_native_mtx_f32 = new uint8_t[lhs_native_size_f32];
    uint8_t* rhs_native_mtx_f32 = new uint8_t[rhs_native_size_f32];
    uint8_t* rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
    uint8_t* rhs_scales_mtx_bf16 = new uint8_t[rhs_scales_size_bf16];

    fill_uniform_random(M, K, (float*)lhs_native_mtx_f32, seed_lhs);
    fill_uniform_random(N, K, (float*)rhs_native_mtx_f32, seed_rhs);

    quant_nxk_qs4c32_f32(N, K, bl,                          // Dimensions
                         (const float*)rhs_native_mtx_f32,  // RHS (F32)
                         rhs_native_mtx_qs4c32,             // RHS (QS4C32)
                         (uint16_t*)rhs_scales_mtx_bf16);   // Scales (Bf16)

    const size_t lhs_ref_size_qa8dx = M * (K + sizeof(int32_t) + sizeof(float));
    const size_t dst_ref_size_f32 = M * N * sizeof(float);

    uint8_t* lhs_ref_mtx_qa8dx = new uint8_t[lhs_ref_size_qa8dx];
    uint8_t* dst_ref_mtx_f32 = new uint8_t[dst_ref_size_f32];

    ref_quant_qa8dx_f32(M, K, (const float*)lhs_native_mtx_f32, (int8_t*)lhs_ref_mtx_qa8dx);

    ref_matmul_mxn_mxk_nxk_f32_qa8dx_qs4c32(M, N, K,                                // Dimensions
                                            bl,                                     // Block length,
                                            (const int8_t*)lhs_ref_mtx_qa8dx,       // LHS
                                            (const uint8_t*)rhs_native_mtx_qs4c32,  // RHS
                                            (const uint16_t*)rhs_scales_mtx_bf16,   // Scale
                                            (float*)dst_ref_mtx_f32,                // DST
                                            -FLT_MAX,
                                            FLT_MAX);  // Min and max for the clamp operation

    // Remove the unnecessary buffer
    delete[] lhs_ref_mtx_qa8dx;

    auto lhs_packed_mtx_qa8dx = new uint8_t[kai_linear.workspace_size(
        M, K,
        mllm::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32)];
    auto rhs_packed_mtx_qs4c32 = new uint8_t[kai_linear.quant_pack_rhs_size(
        N, K,
        mllm::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32)];
    auto dst_act_mtx_f32 = new float[M * N];
    memset(dst_act_mtx_f32, 0, M * N * sizeof(float));

    kai_linear.quant_pack_rhs_offline(
        rhs_packed_mtx_qs4c32, (float*)rhs_native_mtx_f32, nullptr, N, K,
        mllm::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32);

    delete[] rhs_native_mtx_f32;

    kai_linear.matmul(
        dst_act_mtx_f32, (float*)lhs_native_mtx_f32, rhs_packed_mtx_qs4c32, lhs_packed_mtx_qa8dx, M,
        K, N,
        mllm::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32);

    bool is_valid = true;

    auto ref = (float*)dst_ref_mtx_f32;
    auto act = (float*)dst_act_mtx_f32;
    for (size_t i = 0; i < M * N; ++i) {
      if (std::fabs(ref[i] - act[i]) > 0.0001F) {
        const size_t x = i % N;
        const size_t y = i / N;
        Dbg(x, y, ref[i], act[i], std::fabs(ref[i] - act[i]));
        is_valid = false;
        break;
      }
    }

    delete[] dst_ref_mtx_f32;
    delete[] dst_act_mtx_f32;

    return is_valid;
  }

  mllm::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_linear;
  int M = 1;
  int K = 1024;
  int N = 1024;
  int bl = 32;
};

class KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_Test : public testing::Test {
 protected:
  KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_Test() = default;

  ~KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_Test() override = default;

  void fill_uniform_random(size_t num_rows, size_t num_cols, float* dst, size_t seed) {
    std::srand(seed);

    // Fill the array with random values between -1 and 1
    for (size_t i = 0; i < num_rows * num_cols; i++) {
      dst[i] = (float)((double)std::rand() / RAND_MAX) * 2 - 1;
    }
  }

  void fill_uniform_random(size_t num_rows, size_t num_cols, float16_t* dst, size_t seed) {
    std::srand(seed);

    // Fill the array with random values between -1 and 1
    for (size_t i = 0; i < num_rows * num_cols; i++) {
      dst[i] = (float16_t)((double)std::rand() / RAND_MAX) * 2 - 1;
    }
  }

  bool Compare(int threads = 0) {
    using mllm::MllmThreadPool;
    MLLM_THREAD_POOL_INIT(threads);

    const size_t seed_lhs = 4568;
    const size_t seed_rhs = seed_lhs + 4;
    const size_t seed_bias = seed_lhs + 8;

    // Alloc memory
    auto lhs_native_mtx_f16 = new float16_t[M * K];
    auto rhs_native_mtx_f32 = new float[N * K];
    auto dst_mtx_fp16 = new float16_t[M * N];
    auto ref_dst_mtx_fp16 = new float16_t[M * N];
    auto bias_mtx_fp32 = new float[N];

    fill_uniform_random(M, K, lhs_native_mtx_f16, seed_lhs);
    fill_uniform_random(N, K, rhs_native_mtx_f32, seed_rhs);
    fill_uniform_random(N, 1, bias_mtx_fp32, seed_bias);

    auto rhs_quant_size =
        kai_linear.quant_pack_rhs_size(N, K,
                                       mllm::arm::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::
                                           qsi8d32p4x8_qai4c32p4x8_8x4_i8mm);

    auto rhs_quant_data = new uint8_t[rhs_quant_size];
    kai_linear.quant_pack_rhs_offline(rhs_quant_data, rhs_native_mtx_f32, bias_mtx_fp32, K, N,
                                      mllm::arm::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::
                                          qsi8d32p4x8_qai4c32p4x8_8x4_i8mm);

    auto workspace_size =
        kai_linear.workspace_size(M, K, mllm::kFp16,
                                  mllm::arm::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::
                                      qsi8d32p4x8_qai4c32p4x8_8x4_i8mm);
    auto workspace = new uint8_t[workspace_size];

    kai_linear.matmul(dst_mtx_fp16, lhs_native_mtx_f16, rhs_quant_data, workspace, M, K, N,
                      mllm::arm::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::
                          qsi8d32p4x8_qai4c32p4x8_8x4_i8mm);

    // Calculate reference: (M x K) * (N x K)^T = M x N
    for (int m_idx = 0; m_idx < M; ++m_idx) {
      for (int n_idx = 0; n_idx < N; ++n_idx) {
        float16_t sum = 0.0f;
        for (int k_idx = 0; k_idx < K; ++k_idx) {
          // (m_idx, k_idx)
          float16_t lhs_val = lhs_native_mtx_f16[m_idx * K + k_idx];
          // (n_idx, k_idx)
          float16_t rhs_val = static_cast<float16_t>(rhs_native_mtx_f32[n_idx * K + k_idx]);

          sum += lhs_val * rhs_val;

          sum = std::max(sum, static_cast<float16_t>(-65504.0f));
          sum = std::min(sum, static_cast<float16_t>(65504.0f));
        }
        sum += static_cast<float16_t>(bias_mtx_fp32[n_idx]);
        sum = std::max(sum, static_cast<float16_t>(-65504.0f));
        sum = std::min(sum, static_cast<float16_t>(65504.0f));
        ref_dst_mtx_fp16[m_idx * N + n_idx] = sum;
      }
    }

    // All close ?
    bool is_valid = true;
    for (size_t i = 0; i < M * N; ++i) {
      const auto imp_value = dst_mtx_fp16[i];
      const auto ref_value = ref_dst_mtx_fp16[i];
      const auto rel_error =
          ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);
      if (rel_error > 0.01F) {
        const size_t x = i % N;
        const size_t y = i / N;
        Dbg(x, y, ref_dst_mtx_fp16[i], dst_mtx_fp16[i], rel_error);
        is_valid = false;
        break;
      }
    }

    delete[] lhs_native_mtx_f16;
    delete[] rhs_native_mtx_f32;
    delete[] dst_mtx_fp16;
    delete[] ref_dst_mtx_fp16;
    delete[] rhs_quant_data;
    delete[] bias_mtx_fp32;
    return is_valid;
  }

  mllm::arm::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk kai_linear;
  int M = 32;
  int K = 64;
  int N = 64;
  int bl = 32;
};
