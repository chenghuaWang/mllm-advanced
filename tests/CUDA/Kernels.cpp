#include "SoftmaxTest.hpp"
#include "ReductionTest.hpp"
#include <gtest/gtest.h>

TEST_F(SoftmaxTest, _4096x4096) {
  SetShapeAndAlloc(4096, 4096);
  CalculateRef();
  EXPECT_EQ(CalculateFp32AndCompare(), true);
}

TEST_F(SoftmaxTest, _1024x1024) {
  SetShapeAndAlloc(1024, 1024);
  CalculateRef();
  EXPECT_EQ(CalculateFp32AndCompare(), true);
}

TEST_F(SoftmaxTest, _512x512) {
  SetShapeAndAlloc(512, 512);
  CalculateRef();
  EXPECT_EQ(CalculateFp32AndCompare(), true);
}

TEST_F(SoftmaxTest, _256x256) {
  SetShapeAndAlloc(256, 256);
  CalculateRef();
  EXPECT_EQ(CalculateFp32AndCompare(), true);
}

TEST_F(SoftmaxTest, _17x256) {
  SetShapeAndAlloc(17, 256);
  CalculateRef();
  EXPECT_EQ(CalculateFp32AndCompare(), true);
}

TEST_F(SoftmaxTest, _17x128) {
  SetShapeAndAlloc(17, 128);
  CalculateRef();
  EXPECT_EQ(CalculateFp32AndCompare(), true);
}

TEST_F(Reduce1D, _128) {
  SetShapeAndAlloc(128);
  CalculateRef(Reduce1D::kAdd);
  EXPECT_EQ(CalculateFp32AndCompare(Reduce1D::kAdd), true);
}

TEST_F(Reduce1D, _126) {
  SetShapeAndAlloc(126);
  CalculateRef(Reduce1D::kAdd);
  EXPECT_EQ(CalculateFp32AndCompare(Reduce1D::kAdd), true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
