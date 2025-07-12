/**
 * @file Kernels.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "QuantTest.hpp"
#include <gtest/gtest.h>

#if defined(__AVX512F__)
TEST_F(Q4KQ8KTest, AVX512) {
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}
#endif
