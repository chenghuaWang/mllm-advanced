/**
 * @file Thread.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/Thread.hpp"

namespace mllm {

std::thread::id MllmEngineThread::threadId() const { return tcb_.system_tid; }

}  // namespace mllm
