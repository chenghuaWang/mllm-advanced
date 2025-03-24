/**
 * @file transpose.cuh
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-24
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// The transpose op is highly bottleneck on memory throughput.
// Someone who has Hopper Arch Device can use TMA to accelerate this OP.

// H, D -> D, H
template<typename T>
__global__ void transpose_0_1(T* z, T* x, int _1_num, int _2_num) {}

// B, S, H, D -> B, H, S, D
template<typename T>
__global__ void transpose_1_2(T* z, T* x, int _1_num, int _2_num, int last_num) {}
