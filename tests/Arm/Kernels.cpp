#include "HgemvTest.hpp"
#include "SgemvTest.hpp"
#include "SoftmaxTest.hpp"
#include <gtest/gtest.h>

TEST_F(HgemvTest, Hgemv) {
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(HgemvTest, Hgemv4Threads) {
  CalculateRef();
  Calculate(4);
  EXPECT_EQ(Compare(), true);
}

TEST_F(HgemvHighPrecisionTest, Hgemv) {
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(HgemvHighPrecisionTest, Hgemv4threads) {
  CalculateRef();
  Calculate(4);
  EXPECT_EQ(Compare(), true);
}

TEST_F(SgemvTest, Sgemv) {
  CalculateRef();
  Calculate();
  EXPECT_EQ(Compare(), true);
}

TEST_F(SgemvTest, Sgemv4threads) {
  CalculateRef();
  Calculate(4);
  EXPECT_EQ(Compare(), true);
}

TEST_F(SoftmaxTest, SoftmaxFp32) {
  CalculateRef();
  EXPECT_EQ(CalculateFp32AndCompare(), true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
