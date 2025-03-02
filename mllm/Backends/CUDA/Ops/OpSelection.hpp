/**
 * @file OpSelection.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

namespace mllm::cuda {

void vector_add_bf16_v0_call(void* Z, void* const X, void* const Y, int size, float a, float b,
                             float c);

}