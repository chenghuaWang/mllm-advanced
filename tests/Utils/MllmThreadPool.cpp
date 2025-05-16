#include <iostream>
#include "mllm/Utils/Log.hpp"
#include "mllm/Utils/ThreadPool.hpp"

using namespace mllm;  // NOLINT

int main() {
  MLLM_THREAD_POOL_INIT(4);

  MLLM_BIND_WORKER(0, 0x1);
  MLLM_BIND_WORKER(1, 0x1);
  MLLM_BIND_WORKER(2, 0x1);
  MLLM_BIND_WORKER(3, 0x1);
  MLLM_PARALLEL_FOR(i, 0, 8) {
    MLLM_INFO("i={}, thread={}, core={}", i, MLLM_THIS_THREAD_ID, MLLM_CUR_RUN_ON_CPU_ID);
  }
  MLLM_PARALLEL_FOR_END

  std::cout << " --- \n";

  MLLM_BIND_WORKER(0, 0xFF);
  MLLM_BIND_WORKER(1, 0xFF);
  MLLM_BIND_WORKER(2, 0xFF);
  MLLM_BIND_WORKER(3, 0xFF);
  MLLM_PARALLEL_FOR_CHUNK(i, 0, 8, 1) {
    MLLM_INFO("i={}, thread={}, core={}", i, MLLM_THIS_THREAD_ID, MLLM_CUR_RUN_ON_CPU_ID);
  }
  MLLM_PARALLEL_FOR_END

  std::cout << " --- \n";

  MLLM_PARALLEL_FOR_STEP(i, 0, 8, 1) {
    MLLM_INFO("i={}, thread={}, core={}", i, MLLM_THIS_THREAD_ID, MLLM_CUR_RUN_ON_CPU_ID);
  }
  MLLM_PARALLEL_FOR_END
}