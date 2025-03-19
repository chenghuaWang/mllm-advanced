/**
 * @file gpu_info.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// see:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
#define MLLM_CUDA_WARP_SIZE 32
#define MLLM_CUDA_PER_BLOCK_MAX_THREAD_NUM 1024;
#define MLLM_CUDA_PER_SM_32BIT_REGISTER_NUM 64 * 1024;
#define MLLM_CUDA_PER_THREAD_32BIT_REGISTER_NUM 255;