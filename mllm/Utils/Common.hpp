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

#define MLLM_ENABLE_RT_ASSERT 1

namespace mllm {

enum ExitCode : int32_t {
  kSuccess = 0,
  kError,
  kAssert,
  kSliceOB,  // slice out of bound
  kMemory,
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

}  // namespace mllm
