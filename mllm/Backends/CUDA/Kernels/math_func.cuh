/**
 * @file math_func.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cutlass/fast_math.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define MLLM_LDG128(DST, SRC) *(uint4*)(DST) = *(uint4*)(SRC)
#define MLLM_STG128(DST, SRC) *(uint4*)(DST) = *(uint4*)(SRC)

namespace mllm::cuda::mllm_math {

// ============================================================================================
// Numeric Limits
// ============================================================================================
template<typename T>
struct NumericLimitsMax {
  static __forceinline__ __device__ T apply() {
    static_assert(sizeof(T) == 0, "Unsupported type for NumericLimitsMax");
    return T();
  }
};

template<>
struct NumericLimitsMax<double> {
  static __forceinline__ __device__ double apply() { return CUDART_INF; }
};

template<>
struct NumericLimitsMax<float> {
  static __forceinline__ __device__ float apply() { return CUDART_INF_F; }
};

template<>
struct NumericLimitsMax<__half> {
  static __forceinline__ __device__ __half apply() { return __half_raw(__ushort_as_half(0x7C00)); }
};

template<>
struct NumericLimitsMax<__nv_bfloat16> {
  static __forceinline__ __device__ __nv_bfloat16 apply() {
    return __nv_bfloat16_raw(__ushort_as_bfloat16(0x7F80));
  }
};

template<>
struct NumericLimitsMax<__nv_fp8_e4m3> {
  static __forceinline__ __device__ __nv_fp8_e4m3 apply() { return __nv_fp8_e4m3(240.0f); }
};

template<>
struct NumericLimitsMax<__nv_fp8_e5m2> {
  static __forceinline__ __device__ __nv_fp8_e5m2 apply() { return __nv_fp8_e5m2(57344.0f); }
};

template<typename T>
struct NumericLimitsMin {
  static __forceinline__ __device__ T apply() {
    static_assert(sizeof(T) == 0, "Unsupported type for NumericLimitsMin");
    return T();
  }
};

template<>
struct NumericLimitsMin<double> {
  static __forceinline__ __device__ double apply() { return -CUDART_INF; }
};

template<>
struct NumericLimitsMin<float> {
  static __forceinline__ __device__ float apply() { return -CUDART_INF_F; }
};

template<>
struct NumericLimitsMin<__half> {
  static __forceinline__ __device__ __half apply() { return __half_raw(__ushort_as_half(0xFC00)); }
};

template<>
struct NumericLimitsMin<__nv_bfloat16> {
  static __forceinline__ __device__ __nv_bfloat16 apply() {
    return __nv_bfloat16_raw(__ushort_as_bfloat16(0xFF80));
  }
};

template<>
struct NumericLimitsMin<__nv_fp8_e4m3> {
  static __forceinline__ __device__ __nv_fp8_e4m3 apply() { return __nv_fp8_e4m3(-240.0f); }
};

template<>
struct NumericLimitsMin<__nv_fp8_e5m2> {
  static __forceinline__ __device__ __nv_fp8_e5m2 apply() { return __nv_fp8_e5m2(-57344.0f); }
};

template<typename T>
struct NumericLimitsPosZero {
  static __forceinline__ __device__ T apply() {
    static_assert(sizeof(T) == 0, "Unsupported type for NumericLimitsPosZero");
    return T();
  }
};

template<>
struct NumericLimitsPosZero<double> {
  static __forceinline__ __device__ double apply() { return 0.0; }
};

template<>
struct NumericLimitsPosZero<float> {
  static __forceinline__ __device__ float apply() { return 0.0f; }
};

template<>
struct NumericLimitsPosZero<__half> {
  static __forceinline__ __device__ __half apply() { return __half{0}; }
};

template<>
struct NumericLimitsPosZero<__nv_bfloat16> {
  static __forceinline__ __device__ __nv_bfloat16 apply() { return __nv_bfloat16{0}; }
};

template<>
struct NumericLimitsPosZero<__nv_fp8_e4m3> {
  static __forceinline__ __device__ __nv_fp8_e4m3 apply() { return __nv_fp8_e4m3(0.0f); }
};

template<>
struct NumericLimitsPosZero<__nv_fp8_e5m2> {
  static __forceinline__ __device__ __nv_fp8_e5m2 apply() { return __nv_fp8_e5m2(0.0f); }
};

template<typename T>
struct NumericLimitsNegZero {
  static __forceinline__ __device__ T apply() {
    static_assert(sizeof(T) == 0, "Unsupported type for NumericLimitsNegZero");
    return T();
  }
};

template<>
struct NumericLimitsNegZero<double> {
  static __forceinline__ __device__ double apply() { return -0.0; }
};

template<>
struct NumericLimitsNegZero<float> {
  static __forceinline__ __device__ float apply() { return -0.0f; }
};

template<>
struct NumericLimitsNegZero<__half> {
  static __forceinline__ __device__ __half apply() { return __half_raw(__ushort_as_half(0x8000)); }
};

template<>
struct NumericLimitsNegZero<__nv_bfloat16> {
  static __forceinline__ __device__ __nv_bfloat16 apply() {
    return __nv_bfloat16_raw(__ushort_as_bfloat16(0x8000));
  }
};

template<>
struct NumericLimitsNegZero<__nv_fp8_e4m3> {
  static __forceinline__ __device__ __nv_fp8_e4m3 apply() { return __nv_fp8_e4m3(0x80); }
};

template<>
struct NumericLimitsNegZero<__nv_fp8_e5m2> {
  static __forceinline__ __device__ __nv_fp8_e5m2 apply() { return __nv_fp8_e5m2(0x80); }
};

template<typename T>
__forceinline__ __device__ T numeric_limits_max() {
  return NumericLimitsMax<T>::apply();
}

template<typename T>
__forceinline__ __device__ T numeric_limits_min() {
  return NumericLimitsMin<T>::apply();
}

template<typename T>
__forceinline__ __device__ T numeric_limits_pos_zero() {
  return NumericLimitsPosZero<T>::apply();
}

template<typename T>
__forceinline__ __device__ T numeric_limits_neg_zero() {
  return NumericLimitsNegZero<T>::apply();
}

// ============================================================================================
// Constant Helper Functions
// ============================================================================================
template<typename T>
struct ConstantOne {
  static __forceinline__ __device__ T apply() {
    static_assert(sizeof(T) == 0, "Unsupported type for ConstantOne");
    return T();
  }
};

template<>
struct ConstantOne<double> {
  static __forceinline__ __device__ double apply() { return 1.0; }
};

template<>
struct ConstantOne<float> {
  static __forceinline__ __device__ float apply() { return 1.0f; }
};

template<>
struct ConstantOne<half> {
  static __forceinline__ __device__ half apply() { return __float2half_rn(1.0f); }
};

template<>
struct ConstantOne<__nv_bfloat16> {
  static __forceinline__ __device__ __nv_bfloat16 apply() { return __float2bfloat16_rn(1.0f); }
};

template<>
struct ConstantOne<__nv_fp8_e4m3> {
  static __forceinline__ __device__ __nv_fp8_e4m3 apply() { return __nv_fp8_e4m3(1.0f); }
};

template<>
struct ConstantOne<__nv_fp8_e5m2> {
  static __forceinline__ __device__ __nv_fp8_e5m2 apply() { return __nv_fp8_e5m2(1.0f); }
};

// Support double, float, half, bfloat16, fp8_e4m3, fp8_e5m2
template<typename T>
__forceinline__ __device__ T constant_one() {
  return ConstantOne<T>::apply();
}

// Support double, float, half, bfloat16, fp8_e4m3, fp8_e5m2
template<typename T>
__forceinline__ __device__ T constant_pos_zero() {
  return NumericLimitsPosZero<T>::apply();
}

// Support double, float, half, bfloat16, fp8_e4m3, fp8_e5m2
template<typename T>
__forceinline__ __device__ T constant_neg_zero() {
  return NumericLimitsNegZero<T>::apply();
}

// ============================================================================================
// Max and Min
// ============================================================================================
template<typename T>
struct MaxImpl {
  static __forceinline__ __device__ T apply(T a, T b) {
    static_assert(sizeof(T) == 0, "Unsupported type for MaxImpl");
    return T();
  }
};

// specialization
template<>
struct MaxImpl<float> {
  static __forceinline__ __device__ float apply(float a, float b) {
    return cutlass::fast_max(a, b);
  }
};

template<>
struct MaxImpl<half> {
  static __forceinline__ __device__ half apply(half a, half b) { return cutlass::fast_max(a, b); }
};

template<>
struct MaxImpl<nv_bfloat16> {
  static __forceinline__ __device__ nv_bfloat16 apply(nv_bfloat16 a, nv_bfloat16 b) {
    return cutlass::fast_max(a, b);
  }
};

template<>
struct MaxImpl<__nv_fp8_e4m3> {
  static __forceinline__ __device__ __nv_fp8_e4m3 apply(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
    // cast to float
    auto a_half = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E4M3);
    auto b_half = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E4M3);
    auto min_half = MaxImpl<half>::apply(a_half, b_half);
    return __nv_fp8_e4m3(__nv_cvt_halfraw_to_fp8(min_half, __NV_SATFINITE, __NV_E4M3));
  }
};

template<>
struct MaxImpl<__nv_fp8_e5m2> {
  static __forceinline__ __device__ __nv_fp8_e5m2 apply(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
    // cast to float
    auto a_half = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E5M2);
    auto b_half = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E5M2);
    auto min_half = MaxImpl<half>::apply(a_half, b_half);
    return __nv_fp8_e5m2(__nv_cvt_halfraw_to_fp8(min_half, __NV_SATFINITE, __NV_E5M2));
  }
};

template<typename T>
struct MinImpl {
  static __forceinline__ __device__ T apply(T a, T b) {
    static_assert(sizeof(T) == 0, "Unsupported type for MinImpl");
    return T();
  }
};

// specialization
template<>
struct MinImpl<float> {
  static __forceinline__ __device__ float apply(float a, float b) {
    return cutlass::fast_min(a, b);
  }
};

template<>
struct MinImpl<half> {
  static __forceinline__ __device__ half apply(half a, half b) { return cutlass::fast_min(a, b); }
};

template<>
struct MinImpl<nv_bfloat16> {
  static __forceinline__ __device__ nv_bfloat16 apply(nv_bfloat16 a, nv_bfloat16 b) {
    return cutlass::fast_min(a, b);
  }
};

template<>
struct MinImpl<__nv_fp8_e4m3> {
  static __forceinline__ __device__ __nv_fp8_e4m3 apply(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
    // cast to float
    auto a_half = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E4M3);
    auto b_half = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E4M3);
    auto min_half = MinImpl<half>::apply(a_half, b_half);
    return __nv_fp8_e4m3(__nv_cvt_halfraw_to_fp8(min_half, __NV_SATFINITE, __NV_E4M3));
  }
};

template<>
struct MinImpl<__nv_fp8_e5m2> {
  static __forceinline__ __device__ __nv_fp8_e5m2 apply(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
    // cast to float
    auto a_half = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E5M2);
    auto b_half = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E5M2);
    auto min_half = MinImpl<half>::apply(a_half, b_half);
    return __nv_fp8_e5m2(__nv_cvt_halfraw_to_fp8(min_half, __NV_SATFINITE, __NV_E5M2));
  }
};

// mllm_math::max(...)
//
// Support float, half, nv_bfloat16, __nv_fp8_e4m3, __nv_fp8_e5m2
template<typename T>
__forceinline__ __device__ T max(T a, T b) {
  return MaxImpl<T>::apply(a, b);
}

// mllm_math::max(...)
//
// Support float, half, nv_bfloat16, __nv_fp8_e4m3, __nv_fp8_e5m2
template<typename T>
__forceinline__ __device__ T min(T a, T b) {
  return MinImpl<T>::apply(a, b);
}

// ============================================================================================
// Log2
// ============================================================================================

// ============================================================================================
// Ln
// ============================================================================================

// ============================================================================================
// exp
// ============================================================================================
template<typename T>
__forceinline__ __device__ T fast_exp(T a) {
  return cutlass::fast_exp(a);
}

// ============================================================================================
// pow
// ============================================================================================

// ============================================================================================
// others
// ============================================================================================
__device__ __forceinline__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ static float atomicMax(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(::mllm::cuda::mllm_math::max(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

}  // namespace mllm::cuda::mllm_math