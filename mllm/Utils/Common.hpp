/**
 * @file Common.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-26
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Utils/Log.hpp"

#define MLLM_NAMESPACE_ANONYMOUS

#define MLLM_EMPTY_SCOPE

#define MLLM_ENABLE_RT_ASSERT 1

namespace mllm {

enum ExitCode : int32_t {  // NOLINT
  kSuccess = 0,
  kError,
  kAssert,
  kSliceOB,  // slice out of bound
  kMemory,
  kCudaError,
};

// mllm runtime assert
#if (MLLM_ENABLE_RT_ASSERT)
#define MLLM_RT_ASSERT(statement) \
  if (!(statement)) { MLLM_ASSERT_EXIT(ExitCode::kAssert, "{}", #statement); }

#define MLLM_RT_ASSERT_EQ(statement1, statement2)                              \
  if ((statement1) != (statement2)) {                                          \
    MLLM_ASSERT_EXIT(ExitCode::kAssert, "{} != {}", #statement1, #statement2); \
  }
#else
#define MLLM_RT_ASSERT_EQ(statement1, statement2)
#endif

#define NYI(...)                                                           \
  fmt::print(fg(fmt::color::green_yellow) | fmt::emphasis::bold, "[NYI]"); \
  fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__));

}  // namespace mllm
