#include "SoftmaxTest.hpp"
#include <gtest/gtest.h>

TEST_F(SoftmaxTest, _1024x1024) {
  SetShapeAndAlloc(1024, 1024);
  CalculateRef();
  EXPECT_EQ(CalculateFp32AndCompare(), true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
