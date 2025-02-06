#include "HgemvTest.hpp"
#include "SgemvTest.hpp"
#include <gtest/gtest.h>

TEST_F(HgemvTest, Hgemv) {
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(HgemvHighPrecisionTest, Hgemv) {
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(SgemvTest, Sgemv) {
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
