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
#include "mllm/Utils/Common.hpp"

namespace mllm {

enum DataTypes : uint32_t {
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

  kDataTypes_End,
};

inline const char* dataTypes2Str(DataTypes type) {
  switch (type) {
    case DataTypes::kInt4: return "kInt4";
    case DataTypes::kInt8: return "kInt8";
    case DataTypes::kInt16: return "kInt16";
    case DataTypes::kInt32: return "kInt32";
    case DataTypes::kInt64: return "kInt64";
    case DataTypes::kFp4: return "kFp4";
    case DataTypes::kFp8: return "kFp8";
    case DataTypes::kFp16: return "kFp16";
    case DataTypes::kFp32: return "kFp32";
    case DataTypes::kPT_Start: return "kPT_Start";
    case DataTypes::kPTInt4_Sym: return "kPTInt4_Sym";
    case DataTypes::KPTInt4_Asy: return "KPTInt4_Asy";
    case DataTypes::kPTInt8_Sym: return "kPTInt8_Sym";
    case DataTypes::kPTInt8_Asy: return "kPTInt8_Asy";
    case DataTypes::kPT_End: return "kPT_End";
    case DataTypes::kPC_Start: return "kPC_Start";
    case DataTypes::kPCInt4_Sym: return "kPCInt4_Sym";
    case DataTypes::kPCInt4_Asy: return "kPCInt4_Asy";
    case DataTypes::kPCInt8_Sym: return "kPCInt8_Sym";
    case DataTypes::kPCInt8_Asy: return "kPCInt8_Asy";
    case DataTypes::kPC_End: return "kPC_End";
    case DataTypes::kPG_Start: return "kPG_Start";
    case DataTypes::kPG_End: return "kPG_End";
    case DataTypes::kBF16: return "kBF16";
    default: return "Unknown";
  }
}

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
    default: MLLM_ERROR_EXIT(kError, "dataTypeSize of {} is not defined yet.", dataTypes2Str(type));
  }
  return 4.f;
}

}  // namespace mllm
