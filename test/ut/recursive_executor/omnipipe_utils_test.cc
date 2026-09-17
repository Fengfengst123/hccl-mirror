/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>
#include "omnipipe_utils.h"

// omnipipe_utils 中的函数位于 ops_hccl 命名空间
using namespace ops_hccl;

namespace {
constexpr double EPS = 1e-4;
} // anonymous namespace

// ============================================================
// CalcOmniPipeSteps 测试
// ============================================================

class CalcOmniPipeStepsTest : public ::testing::Test {};

// yRankSize=1 时 CalcBandwidth2D 直接返回 xB, steps=1
TEST_F(CalcOmniPipeStepsTest, Bandwidth2D_yRankSize1_returns_xB)
{
    u32 steps = 0;
    double scale = 0;
    double bw = CalcBandwidth2D(56.0, 112.0, 4, 1, 5, steps, scale);
    EXPECT_EQ(steps, 1u);
    EXPECT_DOUBLE_EQ(bw, 56.0);
}

// xRankSize=1 时 CalcBandwidth2D 直接返回 yB, steps=1
TEST_F(CalcOmniPipeStepsTest, Bandwidth2D_xRankSize1_returns_yB)
{
    u32 steps = 0;
    double scale = 0;
    double bw = CalcBandwidth2D(56.0, 112.0, 1, 4, 5, steps, scale);
    EXPECT_EQ(steps, 1u);
    EXPECT_DOUBLE_EQ(bw, 112.0);
}

// growth==1 (xRankSize-1 == bandwidthRatio) → steps = bandwidthRatio + 1
TEST_F(CalcOmniPipeStepsTest, GrowthEquals1_stepsIsBandwidthRatioPlus1)
{
    // bandwidthRatio = 4.0, xRankSize = 5 → growth = (5-1)/4 = 1.0
    double scale = 0;
    u32 steps = CalcOmniPipeSteps(4.0, 5, 5, scale);
    EXPECT_EQ(steps, 5u); // bandwidthRatio+1 = 5
    EXPECT_DOUBLE_EQ(scale, 1.0);
}

// growth > 1 且 steps <= maxStep → 自然步数, scale=1.0
TEST_F(CalcOmniPipeStepsTest, GrowthGreaterThan1_naturalSteps)
{
    // bandwidthRatio = 2.0, xRankSize = 8 → growth = 7/2 = 3.5
    double scale = 0;
    u32 steps = CalcOmniPipeSteps(2.0, 8, 10, scale);
    // 自然步数 = ceil(log(3.5 * 2 + 1) / log(3.5)) + 1 = ceil(log(8) / log(3.5)) + 1 ≈ ceil(1.67) + 1 = 3
    EXPECT_LE(steps, 10u);
    EXPECT_GE(steps, 2u);
    EXPECT_DOUBLE_EQ(scale, 1.0);
}

// growth > 1 但 steps > maxStep → 截断到 maxStep, scale 保持非 1
TEST_F(CalcOmniPipeStepsTest, GrowthGreaterThan1_stepsTruncatedToMax)
{
    // bandwidthRatio = 6.0, xRankSize = 8 → growth = 7/6 ≈ 1.167
    // 自然步数 = ceil(log(2.0)/log(1.167)) + 1 = ceil(4.49) + 1 = 6 > maxStep=5
    double scale = 0;
    u32 steps = CalcOmniPipeSteps(6.0, 8, 5, scale);
    EXPECT_EQ(steps, 5u); // 截断到 maxStep=5
    // scale 不为 1.0 (被截断)
    EXPECT_NE(scale, 1.0);
}

// growth < 1 (xRankSize <= bandwidthRatio) → 走满 maxStep, scale 非 1
TEST_F(CalcOmniPipeStepsTest, GrowthLessThan1_usesMaxStep)
{
    // bandwidthRatio = 10.0, xRankSize = 2 → growth = 1/10 = 0.1
    double scale = 0;
    u32 steps = CalcOmniPipeSteps(10.0, 2, 5, scale);
    EXPECT_EQ(steps, 5u);
    EXPECT_NE(scale, 1.0);
}

// maxStep=1 边界: steps 应为 1
TEST_F(CalcOmniPipeStepsTest, MaxStep1_returns1)
{
    double scale = 0;
    u32 steps = CalcOmniPipeSteps(4.0, 8, 1, scale);
    EXPECT_EQ(steps, 1u);
}

// ============================================================
// CalcOmniPipeDataRatio 测试
// ============================================================

class CalcOmniPipeDataRatioTest : public ::testing::Test {};

// yDataSizeRatio[0] 始终为 1.0
TEST_F(CalcOmniPipeDataRatioTest, FirstYDataRatioIsOne)
{
    double scale = 1.0;
    u32 steps = 3;
    std::vector<double> xRatio(steps);
    std::vector<double> yRatio(steps);
    CalcOmniPipeDataRatio(2.0, 4, 4, steps, scale, xRatio.data(), yRatio.data());
    EXPECT_DOUBLE_EQ(yRatio[0], 1.0);
}

// xDataSizeRatio[0] = scale / bandwidthRatio
TEST_F(CalcOmniPipeDataRatioTest, FirstXDataRatioIsScaleOverBandwidth)
{
    double scale = 1.0;
    u32 steps = 3;
    std::vector<double> xRatio(steps);
    std::vector<double> yRatio(steps);
    CalcOmniPipeDataRatio(2.0, 4, 4, steps, scale, xRatio.data(), yRatio.data());
    EXPECT_DOUBLE_EQ(xRatio[0], scale / 2.0);
}

// 最后一步 yDataSizeRatio[steps-1] = xDataSizeRatio[steps-1] * bandwidthRatio
TEST_F(CalcOmniPipeDataRatioTest, LastYIsXTimesBandwidth)
{
    double scale = 1.0;
    u32 steps = 3;
    std::vector<double> xRatio(steps);
    std::vector<double> yRatio(steps);
    CalcOmniPipeDataRatio(2.0, 4, 4, steps, scale, xRatio.data(), yRatio.data());
    EXPECT_DOUBLE_EQ(yRatio[steps - 1], xRatio[steps - 1] * 2.0);
}

// steps=2: 只有第一步和斜对角步，无中间步骤
TEST_F(CalcOmniPipeDataRatioTest, TwoStepsOnlyNoMiddleLoop)
{
    double scale = 1.0;
    u32 steps = 2;
    std::vector<double> xRatio(steps);
    std::vector<double> yRatio(steps);
    CalcOmniPipeDataRatio(2.0, 4, 4, steps, scale, xRatio.data(), yRatio.data());
    EXPECT_DOUBLE_EQ(yRatio[0], 1.0);
    EXPECT_DOUBLE_EQ(xRatio[0], 0.5); // 1.0 / 2.0
    // 最后一步由公式计算
    EXPECT_GT(xRatio[1], 0.0);
    EXPECT_GT(yRatio[1], 0.0);
}

// ============================================================
// CalcOmniPipeDataSlice 测试
// ============================================================

class CalcOmniPipeDataSliceTest : public ::testing::Test {};

// ySliceCount[0] == sliceCount
TEST_F(CalcOmniPipeDataSliceTest, FirstYIsSliceCount)
{
    u64 sliceCount = 1024;
    std::vector<u64> xSlice(3, 0);
    std::vector<u64> ySlice(3, 0);
    OmniPipeXYdata xy{3, 1.0, 2.0, 4, 4};
    CalcOmniPipeDataSlice(xy, sliceCount, xSlice, ySlice);
    EXPECT_EQ(ySlice[0], sliceCount);
}

// xSliceCount[0] = sliceCount * scale / bandwidthRatio (向下取整)
TEST_F(CalcOmniPipeDataSliceTest, FirstXIsSliceCountTimesScaleOverBandwidth)
{
    u64 sliceCount = 1024;
    double scale = 1.0;
    double bandwidthRatio = 2.0;
    std::vector<u64> xSlice(3, 0);
    std::vector<u64> ySlice(3, 0);
    OmniPipeXYdata xy{3, scale, bandwidthRatio, 4, 4};
    CalcOmniPipeDataSlice(xy, sliceCount, xSlice, ySlice);
    u64 expected = static_cast<u64>(sliceCount * scale / bandwidthRatio);
    EXPECT_EQ(xSlice[0], expected);
}

// 所有切片不超过 sliceCount（检出 u64 下溢回绕）
TEST_F(CalcOmniPipeDataSliceTest, AllSlicesNotExceedSliceCount)
{
    u64 sliceCount = 4096;
    // steps=3 是 CalcOmniPipeSteps(2.0, 4, 5) 的真实返回值；xRankSize=4, yRankSize=2 保证末步公式不超 sliceCount
    u32 steps = 3;
    std::vector<u64> xSlice(steps, 0);
    std::vector<u64> ySlice(steps, 0);
    CalcOmniPipeDataSlice({steps, 1.0, 2.0, 4, 2}, sliceCount, xSlice, ySlice);
    for (u32 i = 0; i < steps; ++i) {
        EXPECT_LE(xSlice[i], sliceCount);
        EXPECT_LE(ySlice[i], sliceCount);
    }
}

// 最后一步 xSliceCount 必须是 (yRankSize-1) 的倍数
TEST_F(CalcOmniPipeDataSliceTest, LastXIsMultipleOfYRankSizeMinus1)
{
    u64 sliceCount = 4096;
    u32 yRankSize = 4;
    u32 steps = 3;
    std::vector<u64> xSlice(steps, 0);
    std::vector<u64> ySlice(steps, 0);
    CalcOmniPipeDataSlice({steps, 1.0, 2.0, 8, yRankSize}, sliceCount, xSlice, ySlice);
    EXPECT_EQ(xSlice[steps - 1] % (yRankSize - 1), 0ull);
}

// 最后一步 ySliceCount 必须是 (xRankSize-1) 的倍数
TEST_F(CalcOmniPipeDataSliceTest, LastYIsMultipleOfXRankSizeMinus1)
{
    u64 sliceCount = 4096;
    u32 xRankSize = 8;
    u32 steps = 3;
    std::vector<u64> xSlice(steps, 0);
    std::vector<u64> ySlice(steps, 0);
    CalcOmniPipeDataSlice({steps, 1.0, 2.0, xRankSize, 4}, sliceCount, xSlice, ySlice);
    EXPECT_EQ(ySlice[steps - 1] % (xRankSize - 1), 0ull);
}

// 非法输入：xRankSize=1 应返回错误码
TEST_F(CalcOmniPipeDataSliceTest, InvalidRankSizeReturnsError)
{
    u64 sliceCount = 1024;
    std::vector<u64> xSlice(3, 0);
    std::vector<u64> ySlice(3, 0);
    OmniPipeXYdata xy{3, 1.0, 2.0, 1, 4}; // xEqRankSize=1
    EXPECT_EQ(CalcOmniPipeDataSlice(xy, sliceCount, xSlice, ySlice), HCCL_E_PARA);
}

// 非法输入：steps 与向量长度不一致应返回错误码
TEST_F(CalcOmniPipeDataSliceTest, StepsMismatchReturnsError)
{
    u64 sliceCount = 1024;
    std::vector<u64> xSlice(2, 0); // steps=3 > size=2
    std::vector<u64> ySlice(2, 0);
    OmniPipeXYdata xy{3, 1.0, 2.0, 4, 4};
    EXPECT_EQ(CalcOmniPipeDataSlice(xy, sliceCount, xSlice, ySlice), HCCL_E_PARA);
}

// ============================================================
// CalcBandwidth2D 集成测试
// ============================================================

class CalcBandwidth2DTest : public ::testing::Test {};

// 两轴都有 rank > 1 时返回值应大于 0
TEST_F(CalcBandwidth2DTest, BothAxesMultiRank_returnsPositive)
{
    u32 steps = 0;
    double scale = 0;
    double bw = CalcBandwidth2D(56.0, 112.0, 4, 4, 5, steps, scale);
    EXPECT_GT(bw, 0.0);
    EXPECT_GE(steps, 2u);
}

// 返回值应 <= max(xB, yB) (等效带宽不超过快轴带宽)
TEST_F(CalcBandwidth2DTest, BandwidthNotExceedFastAxis)
{
    u32 steps = 0;
    double scale = 0;
    double xB = 56.0;
    double yB = 112.0;
    double bw = CalcBandwidth2D(xB, yB, 8, 8, 5, steps, scale);
    // 等效带宽为正且不超过快轴带宽 yB
    EXPECT_GT(bw, 0.0);
    EXPECT_LE(bw, yB + EPS);
}

// 对称拓扑 (xB == yB, xRankSize == yRankSize) 结果合理
TEST_F(CalcBandwidth2DTest, SymmetricTopology_reasonableResult)
{
    u32 steps = 0;
    double scale = 0;
    double bw = CalcBandwidth2D(56.0, 56.0, 8, 8, 5, steps, scale);
    // 带宽比=1, growth=(8-1)/1=7, steps=ceil(log(7+1)/log(7))+1 = 2
    EXPECT_GT(bw, 0.0);
    EXPECT_GE(steps, 2u);
    EXPECT_LE(steps, 5u);
}

// ============================================================
// CalcParallelSliceCount 测试
// 契约（RFC 0003）：childSlice[i] = floor(parentSlice * ratio[i] / sum(ratio))
// 最后一个 Child 承接 parentSlice - 已分配和
// ============================================================

class CalcParallelSliceCountTest : public ::testing::Test {};

// RFC 示例：{2,1} parentSlice=10 → child0=floor(10*2/3)=6, child1=10-6=4
TEST_F(CalcParallelSliceCountTest, RfcExample_2_1_Parent10)
{
    u64 parent = 10;
    u64 sum = 3;
    EXPECT_EQ(CalcParallelSliceCount(parent, 2, sum), 6ULL);          // child0
    EXPECT_EQ(parent - CalcParallelSliceCount(parent, 2, sum), 4ULL); // child1 (余量)
}

// 不整除 {1,1,1} parentSlice=10 → 各 floor(10/3)=3，末子=10-6=4
TEST_F(CalcParallelSliceCountTest, NonDivisible_1_1_1_Parent10)
{
    u64 parent = 10;
    u64 sum = 3;
    u64 c0 = CalcParallelSliceCount(parent, 1, sum);
    u64 c1 = CalcParallelSliceCount(parent, 1, sum);
    u64 last = parent - c0 - c1;
    EXPECT_EQ(c0, 3ULL);
    EXPECT_EQ(c1, 3ULL);
    EXPECT_EQ(last, 4ULL);             // 余量集中末位
    EXPECT_EQ(c0 + c1 + last, parent); // 数据不丢失
}

// 大张量 {1,2,0} parentSlice=10^9：float 会下溢，整数运算正确
TEST_F(CalcParallelSliceCountTest, LargeSlice_1_2_0_Parent1e9_NoUnderflow)
{
    u64 parent = 1000000000ULL;
    u64 sum = 3;
    u64 c0 = CalcParallelSliceCount(parent, 1, sum);
    u64 c1 = CalcParallelSliceCount(parent, 2, sum);
    u64 last = parent - c0 - c1; // 末子 ratio=0，取余量
    EXPECT_EQ(c0, 333333333ULL);
    EXPECT_EQ(c1, 666666666ULL);
    EXPECT_EQ(last, 1ULL); // 余量=1，非 2^64-32
    EXPECT_EQ(c0 + c1 + last, parent);
}

// 平衡比例 {1,1,1} parentSlice=10^9：整数精确，无 float 偏差
TEST_F(CalcParallelSliceCountTest, Balanced_1_1_1_Parent1e9_Exact)
{
    u64 parent = 1000000000ULL;
    u64 sum = 3;
    u64 c0 = CalcParallelSliceCount(parent, 1, sum);
    u64 c1 = CalcParallelSliceCount(parent, 1, sum);
    u64 last = parent - c0 - c1;
    EXPECT_EQ(c0, 333333333ULL);
    EXPECT_EQ(c1, 333333333ULL);
    EXPECT_EQ(last, 333333334ULL);
    EXPECT_EQ(c0 + c1 + last, parent);
}

// 超过 fp24 精度阈值（2^24=16777216），验证不丢精度
TEST_F(CalcParallelSliceCountTest, BeyondFp24Precision)
{
    u64 parent = 16777216ULL + 1ULL; // 2^24+1
    u64 sum = 2;
    u64 c0 = CalcParallelSliceCount(parent, 1, sum);
    u64 last = parent - c0;
    EXPECT_EQ(c0, 8388608ULL); // floor(16777217/2)
    EXPECT_EQ(last, 8388609ULL);
    EXPECT_EQ(c0 + last, parent);
}

// ratio=0 的非末子：该 Child 得 0
TEST_F(CalcParallelSliceCountTest, ZeroRatioGetsZero)
{
    u64 parent = 100;
    u64 sum = 3;
    EXPECT_EQ(CalcParallelSliceCount(parent, 0, sum), 0ULL);
}
