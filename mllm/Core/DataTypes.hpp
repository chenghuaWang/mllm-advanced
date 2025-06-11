/**
 * @file DataTypes.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>
#include <half/half.hpp>
#include <tuple>
#include "mllm/Utils/Common.hpp"

namespace mllm {

enum DataTypes : uint32_t {  // NOLINT
  kDataTypes_Start = 0,

  // normal
  kInt4,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kFp4,
  kFp8,
  kFp16,
  kFp32,

  // Per Tensor Quantization
  kPT_Start,
  kPTInt4_Sym,
  KPTInt4_Asy,
  kPTInt8_Sym,
  kPTInt8_Asy,
  kPT_End,

  // Per Channel Quantization
  kPC_Start,
  kPCInt4_Sym,
  kPCInt4_Asy,
  kPCInt8_Sym,
  kPCInt8_Asy,
  kPC_End,

  // Group Quantization
  kPG_Start,
  // TODO
  kPG_End,

  kBF16,

  // GGML
  kGGML_Start,
  kGGML_Q4_0,
  kGGML_Q4_K,
  kGGML_Q8_0,
  kGGML_Q8_K,
  kGGML_End,

  kDataTypes_End,
};

inline const char* dataTypes2Str(DataTypes type) {
  switch (type) {
    case DataTypes::kInt4: return "Int4";
    case DataTypes::kInt8: return "Int8";
    case DataTypes::kInt16: return "Int16";
    case DataTypes::kInt32: return "Int32";
    case DataTypes::kInt64: return "Int64";
    case DataTypes::kFp4: return "Fp4";
    case DataTypes::kFp8: return "Fp8";
    case DataTypes::kFp16: return "Fp16";
    case DataTypes::kFp32: return "Fp32";
    case DataTypes::kPT_Start: return "PT_Start";
    case DataTypes::kPTInt4_Sym: return "PTInt4_Sym";
    case DataTypes::KPTInt4_Asy: return "PTInt4_Asy";
    case DataTypes::kPTInt8_Sym: return "PTInt8_Sym";
    case DataTypes::kPTInt8_Asy: return "PTInt8_Asy";
    case DataTypes::kPT_End: return "PT_End";
    case DataTypes::kPC_Start: return "PC_Start";
    case DataTypes::kPCInt4_Sym: return "PCInt4_Sym";
    case DataTypes::kPCInt4_Asy: return "PCInt4_Asy";
    case DataTypes::kPCInt8_Sym: return "PCInt8_Sym";
    case DataTypes::kPCInt8_Asy: return "PCInt8_Asy";
    case DataTypes::kPC_End: return "PC_End";
    case DataTypes::kPG_Start: return "PG_Start";
    case DataTypes::kPG_End: return "PG_End";
    case DataTypes::kGGML_Start: return "GGML_Start";
    case DataTypes::kGGML_Q4_0: return "GGML_Q4_0";
    case DataTypes::kGGML_Q4_K: return "GGML_Q4_K";
    case DataTypes::kGGML_Q8_0: return "GGML_Q8_0";
    case DataTypes::kGGML_Q8_K: return "GGML_Q8_K";
    case DataTypes::kGGML_End: return "GGML_End";
    case DataTypes::kBF16: return "BF16";
    default: return "Unknown";
  }
}

/*
 * This code is based on ggml(https://github.com/ggerganov/ggml),
 * please see https://github.com/ggerganov/ggml/blob/master/src/ggml.c
 * ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
 *
 * MIT License
 * Copyright (c) 2022 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// using pack(1) instead of __attribute__ for compatibility with MSVC
#pragma pack(push, 1)
using __block_q4_0 = struct {
  half_float::half d;  // delta
  uint8_t qs[16];      // nibbles / quants
};
#pragma pack(pop)
// 4-bit round-to-nearest quantization (q). Each block has 32 weights. Weight formula: w = q *
// block_scale. Legacy quantization method (not used widely as of today).
using block_q4_0_t = __block_q4_0;
static_assert(sizeof(block_q4_0_t) == 16 + 2);  // 16B(32x4bits) + 2B(delta)

#pragma pack(push, 1)
struct __block_q4_k {
  half_float::half d;     // super-block scale for quantized scales
  half_float::half dmin;  // super-block scale for quantized mins
  uint8_t scales[12];     // scales, mins quantized with 6 bits
  uint8_t qs[128];        // 4--bit quants
};
#pragma pack(pop)
// 4-bit quantization (q). Super-blocks with 8 blocks, each block has 32 weights. Weight formula: w
// = q * block_scale(6-bit) + block_min(6-bit), resulting in 4.5 bits-per-weight.
using block_q4_k_t = __block_q4_k;
static_assert(sizeof(block_q4_k_t)
              == 128 + 2 + 2
                     + 12);  // 128B(256x4bits) + 2B(scale) + 2B(min) + 12B((6bits + 6bits) x 8)

#pragma pack(push, 1)
struct __block_q8_k {
  float d;            // delta
  int8_t qs[256];     // quants
  int16_t bsums[16];  // sum of quants in groups of 16
};
#pragma pack(pop)

//  8-bit quantization (q). Each block has 256 weights. Only used for quantizing intermediate
//  results. All 2-6 bit dot products are implemented for this quantization type. Weight formula: w
//  = q * block_scale.
using block_q8_k_t = __block_q8_k;
static_assert(sizeof(block_q8_k_t) == 256 + 4 + 32);

inline float dataTypeSize(DataTypes type) {
  switch (type) {
    case DataTypes::kInt4: return 1.f / 2;
    case DataTypes::kInt8: return 1.f;
    case DataTypes::kInt16: return 2.f;
    case DataTypes::kInt32: return 4.f;
    case DataTypes::kInt64: return 8.f;
    case DataTypes::kFp4: return 1.f / 2;
    case DataTypes::kFp8: return 1.f;
    case DataTypes::kFp16: return 2.f;
    case DataTypes::kFp32: return 4.f;
    case DataTypes::kBF16: return 2.f;
    case DataTypes::kGGML_Q4_0: return float(sizeof(block_q4_0_t)) / 32.f;
    case DataTypes::kGGML_Q4_K: return float(sizeof(block_q4_k_t)) / 256.f;
    case DataTypes::kGGML_Q8_0: return float(sizeof(block_q8_k_t)) / 256.f;
    default: MLLM_ERROR_EXIT(kError, "dataTypeSize of {} is not defined yet.", dataTypes2Str(type));
  }
  return 4.f;
}

inline std::tuple<uint32_t, uint32_t> dataTypesBitsAndLanes(DataTypes type) {
  switch (type) {
    case DataTypes::kInt4: return {4, 1};
    case DataTypes::kInt8: return {8, 1};
    case DataTypes::kInt16: return {16, 1};
    case DataTypes::kInt32: return {32, 1};
    case DataTypes::kInt64: return {64, 1};
    case DataTypes::kFp4: return {4, 1};
    case DataTypes::kFp8: return {8, 1};
    case DataTypes::kFp16: return {16, 1};
    case DataTypes::kFp32: return {32, 1};
    case DataTypes::kBF16: return {16, 1};
    case DataTypes::kGGML_Q4_0: return {sizeof(block_q4_0_t) * 8, 32};
    case DataTypes::kGGML_Q4_K: return {sizeof(block_q4_k_t) * 8, 256};
    case DataTypes::kGGML_Q8_0: return {sizeof(block_q8_k_t) * 8, 256};
    default:
      MLLM_ERROR_EXIT(kError, "dtypesBitsAndLanes of {} is not defined yet.", dataTypes2Str(type));
  }
}

// FIXME: Impl this struct. compatible with dlpack's definition
struct DataType {
  DataType(DataTypes data_type_code)  // NOLINT: google-explicit-constructor
      : data_type_code_(data_type_code) {
    auto [bits, lanes] = dataTypesBitsAndLanes(data_type_code);
    bits_ = bits;
    lanes_ = lanes;
  }

  float everyLanesBytes() { return ((float)bits_ / 8.f) / (float)lanes_; }

  float everyLanesBits() { return (float)bits_ / (float)lanes_; }

  DataTypes data_type_code_ = DataTypes::kFp32;
  uint32_t bits_ = 32;
  uint32_t lanes_ = 1;
};

}  // namespace mllm
