#include "HgemvTest.hpp"
#include "SgemvTest.hpp"
#include "SoftmaxTest.hpp"
#include "TransposeTest.hpp"
#include "SgemmTest.hpp"
#include "KaiLinearTest.hpp"
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

TEST_F(SoftmaxTest, SoftmaxFp16) {
  CalculateRef();
  EXPECT_EQ(CalculateFp16AndCompare(), true);
}

TEST_F(TransposeTest, BSHD2BHSD) {
  CalculateRef();
  Calculate(4);
  EXPECT_EQ(Compare(), true);
}

TEST_F(Sgemm_MK_NK_MN_V1_Test, Sgemm_MK_NK_MN_V1) {
  CalculateRef();
  Calculate(0);
  EXPECT_EQ(Compare(), true);
}

TEST_F(Sgemm_MK_NK_MN_V1_Test, Sgemm_MK_NK_MN_V1_4threads) {
  CalculateRef();
  Calculate(4);
  EXPECT_EQ(Compare(), true);
}

TEST_F(Sgemm_MK_KN_MN_V1_Test, Sgemm_MK_KN_MN_V1) {
  CalculateRef();
  Calculate(0);
  EXPECT_EQ(Compare(), true);
}

TEST_F(Sgemm_MK_KN_MN_V1_Test, Sgemm_MK_KN_MN_V1_4threads) {
  CalculateRef();
  Calculate(4);
  EXPECT_EQ(Compare(), true);
}

TEST_F(KaiLinear_fp16_fp16_fp16p_mxk_kxn_Test, _4threads) {
  CalculateRef();
  Calculate(4);
  EXPECT_EQ(Compare(), true);
}

TEST_F(KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_Test, _4threads) { EXPECT_EQ(Compare(4), true); }

TEST_F(KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_Test, _4threads) { EXPECT_EQ(Compare(4), true); }

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
