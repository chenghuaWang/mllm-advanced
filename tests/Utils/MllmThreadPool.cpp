#include <iostream>
#include "mllm/Utils/ThreadPool.hpp"

using namespace mllm;  // NOLINT

int main() {
  MLLM_THREAD_POOL_INIT(4);

  MLLM_PARALLEL_FOR(i, 0, 8) {
    std::cout << "i=" << i << " processed by thread " << MLLM_THIS_THREAD_ID << std::endl;
  }
  MLLM_PARALLEL_FOR_END

  std::cout << " --- \n";

  MLLM_PARALLEL_FOR_CHUNK(i, 0, 8, 1) {
    std::cout << "i=" << i << " by thread " << MLLM_THIS_THREAD_ID << std::endl;
  }
  MLLM_PARALLEL_FOR_END

  std::cout << " --- \n";

  MLLM_PARALLEL_FOR_STEP(i, 0, 8, 1) {
    std::cout << "i=" << i << " by thread " << MLLM_THIS_THREAD_ID << std::endl;
  }
  MLLM_PARALLEL_FOR_END
}