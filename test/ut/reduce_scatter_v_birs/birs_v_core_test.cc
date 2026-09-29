/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "reduce_scatter_v_executor_base.h"
#include "reduce_scatter_v_birs.h"
#include "reduce_scatter_v_birs_inter.h"

namespace {

using ops_hccl::Slice;
using ops_hccl_experimental::ReduceScatterVBIRS;
using ops_hccl_experimental::ReduceScatterVBIRSInter;
using ops_hccl_experimental::ReduceScatterVExecutorBase;

constexpr u64 UT_MIN_SLICE_ALIGN = 16384; // HCCL_MIN_SLICE_ALIGN_910B

class TestableExecutor : public ReduceScatterVExecutorBase {
public:
    using ReduceScatterVExecutorBase::PrepareDataSliceV;
};

class TestableBirs : public ReduceScatterVBIRS {
public:
    using ReduceScatterVBIRS::CalcLocalStrideSize;
    using ReduceScatterVBIRS::CheckScratchCapacity;

    void SetSliceSizes(const std::vector<u64>& sizes)
    {
        slicesDummy_.resize(sizes.size());
        u64 offset = 0;
        for (size_t i = 0; i < sizes.size(); i++) {
            slicesDummy_[i].offset = offset;
            slicesDummy_[i].size = sizes[i];
            offset += sizes[i];
        }
    }

    void SetScratch(u64 size) { scratchMem_ = HcclMem{HCCL_MEM_TYPE_DEVICE, nullptr, size}; }

    u64 GetStride() const { return localStrideSize; }
};

class TestableBirsInter : public ReduceScatterVBIRSInter {
public:
    using ReduceScatterVBIRSInter::CalcLocalStrideSize;
    using ReduceScatterVBIRSInter::CheckScratchCapacity;

    void SetSliceSizes(const std::vector<u64>& sizes)
    {
        slicesDummy_.resize(sizes.size());
        u64 offset = 0;
        for (size_t i = 0; i < sizes.size(); i++) {
            slicesDummy_[i].offset = offset;
            slicesDummy_[i].size = sizes[i];
            offset += sizes[i];
        }
    }

    void SetScratch(u64 size) { scratchMem_ = HcclMem{HCCL_MEM_TYPE_DEVICE, nullptr, size}; }
};

Slice MakeSlice(u64 offset, u64 size)
{
    Slice slice;
    slice.offset = offset;
    slice.size = size;
    return slice;
}

// ---------- PrepareDataSliceV: sendCounts/sendDispls -> slices 的 offset/size 换算 ----------

TEST(ReduceScatterVBirsCore, PrepareDataSliceVNonUniformCounts)
{
    TestableExecutor executor;
    const u64 sendCounts[] = {100, 200, 300, 50};
    const u64 sendDispls[] = {0, 100, 300, 600};
    std::vector<Slice> dataSlice;
    ASSERT_EQ(executor.PrepareDataSliceV(sendCounts, sendDispls, 4, 4, dataSlice), HCCL_SUCCESS);
    ASSERT_EQ(dataSlice.size(), 4U);
    const Slice expected[] = {MakeSlice(0, 400), MakeSlice(400, 800), MakeSlice(1200, 1200), MakeSlice(2400, 200)};
    for (u32 i = 0; i < 4; i++) {
        EXPECT_EQ(dataSlice[i].offset, expected[i].offset) << "slice " << i;
        EXPECT_EQ(dataSlice[i].size, expected[i].size) << "slice " << i;
    }
}

TEST(ReduceScatterVBirsCore, PrepareDataSliceVZeroCountRanks)
{
    TestableExecutor executor;
    const u64 sendCounts[] = {0, 200, 0, 50};
    const u64 sendDispls[] = {0, 0, 200, 200};
    std::vector<Slice> dataSlice;
    ASSERT_EQ(executor.PrepareDataSliceV(sendCounts, sendDispls, 4, 4, dataSlice), HCCL_SUCCESS);
    ASSERT_EQ(dataSlice.size(), 4U);
    EXPECT_EQ(dataSlice[0].size, 0U);
    EXPECT_EQ(dataSlice[1].offset, 0U);
    EXPECT_EQ(dataSlice[1].size, 800U);
    EXPECT_EQ(dataSlice[2].size, 0U);
    EXPECT_EQ(dataSlice[3].offset, 800U);
    EXPECT_EQ(dataSlice[3].size, 200U);
}

TEST(ReduceScatterVBirsCore, PrepareDataSliceVSliceNumZeroRejected)
{
    TestableExecutor executor;
    const u64 sendCounts[] = {100};
    const u64 sendDispls[] = {0};
    std::vector<Slice> dataSlice;
    EXPECT_EQ(executor.PrepareDataSliceV(sendCounts, sendDispls, 4, 0, dataSlice), HCCL_E_PARA);
}

TEST(ReduceScatterVBirsCore, PrepareDataSliceVNullPointerRejected)
{
    TestableExecutor executor;
    const u64 sendCounts[] = {100};
    const u64 sendDispls[] = {0};
    std::vector<Slice> dataSlice;
    EXPECT_NE(executor.PrepareDataSliceV(nullptr, sendDispls, 4, 1, dataSlice), HCCL_SUCCESS);
    EXPECT_NE(executor.PrepareDataSliceV(sendCounts, nullptr, 4, 1, dataSlice), HCCL_SUCCESS);
}

TEST(ReduceScatterVBirsCore, PrepareDataSliceV64BitNoTruncation)
{
    TestableExecutor executor;
    const u64 sendCounts[] = {0x100000000ULL, 1};
    const u64 sendDispls[] = {0, 0x100000000ULL};
    std::vector<Slice> dataSlice;
    ASSERT_EQ(executor.PrepareDataSliceV(sendCounts, sendDispls, 8, 2, dataSlice), HCCL_SUCCESS);
    EXPECT_EQ(dataSlice[0].size, 0x800000000ULL);
    EXPECT_EQ(dataSlice[1].offset, 0x800000000ULL);
}

TEST(ReduceScatterVBirsCore, PrepareDataSliceVResizesExistingVector)
{
    TestableExecutor executor;
    const u64 sendCounts[] = {10, 20};
    const u64 sendDispls[] = {0, 10};
    std::vector<Slice> dataSlice(7, MakeSlice(123, 456));
    ASSERT_EQ(executor.PrepareDataSliceV(sendCounts, sendDispls, 2, 2, dataSlice), HCCL_SUCCESS);
    ASSERT_EQ(dataSlice.size(), 2U);
    EXPECT_EQ(dataSlice[0].size, 20U);
    EXPECT_EQ(dataSlice[1].offset, 20U);
}

// ---------- Intra: CalcLocalStrideSize(RoundUp 16KB) 与 CheckScratchCapacity ----------

TEST(ReduceScatterVBirsCore, IntraStrideRoundsUpMaxSliceTo16K)
{
    TestableBirs birs;
    birs.SetSliceSizes({1000, 300, 16385, 7});
    birs.CalcLocalStrideSize(4);
    EXPECT_EQ(birs.GetStride(), 2 * UT_MIN_SLICE_ALIGN);
}

TEST(ReduceScatterVBirsCore, IntraStrideExactMultipleUnchanged)
{
    TestableBirs birs;
    birs.SetSliceSizes({100, UT_MIN_SLICE_ALIGN, 512});
    birs.CalcLocalStrideSize(3);
    EXPECT_EQ(birs.GetStride(), UT_MIN_SLICE_ALIGN);
}

TEST(ReduceScatterVBirsCore, IntraStrideMaxAtLastRank)
{
    TestableBirs birs;
    birs.SetSliceSizes({100, 200, 300, 5000});
    birs.CalcLocalStrideSize(4);
    EXPECT_EQ(birs.GetStride(), UT_MIN_SLICE_ALIGN);
}

TEST(ReduceScatterVBirsCore, IntraStrideAllZeroSlices)
{
    // RoundUpWithDivisor(0, d) 按约定返回 d: 全零切片(生产流程被入口 all-zero 拦截, 不会到达模板)
    // 的兜底步长为一个对齐单位, 而非 0
    TestableBirs birs;
    birs.SetSliceSizes({0, 0, 0, 0});
    birs.CalcLocalStrideSize(4);
    EXPECT_EQ(birs.GetStride(), UT_MIN_SLICE_ALIGN);
}

TEST(ReduceScatterVBirsCore, IntraScratchCapacityBoundary)
{
    TestableBirs birs;
    birs.SetSliceSizes({5000, 100, 200, 300});
    birs.CalcLocalStrideSize(4);
    ASSERT_EQ(birs.GetStride(), UT_MIN_SLICE_ALIGN);
    birs.SetScratch(4 * UT_MIN_SLICE_ALIGN);
    EXPECT_EQ(birs.CheckScratchCapacity(4), HCCL_SUCCESS);
    birs.SetScratch(4 * UT_MIN_SLICE_ALIGN - 1);
    EXPECT_EQ(birs.CheckScratchCapacity(4), HCCL_E_PARA);
}

TEST(ReduceScatterVBirsCore, IntraScratchCapacityZeroSlicesFallbackStride)
{
    TestableBirs birs;
    birs.SetSliceSizes({0, 0, 0, 0});
    birs.CalcLocalStrideSize(4);
    birs.SetScratch(4 * UT_MIN_SLICE_ALIGN);
    EXPECT_EQ(birs.CheckScratchCapacity(4), HCCL_SUCCESS);
    birs.SetScratch(4 * UT_MIN_SLICE_ALIGN - 1);
    EXPECT_EQ(birs.CheckScratchCapacity(4), HCCL_E_PARA);
}

// ---------- Inter: Prepare 校验 / 不对齐步长 / (rankSize + 2*serverNum) 容量公式 ----------

TEST(ReduceScatterVBirsCore, InterPrepareDivisibility)
{
    TestableBirsInter birsInter;
    EXPECT_EQ(birsInter.Prepare(2, 8), HCCL_SUCCESS);
    EXPECT_EQ(birsInter.Prepare(3, 8), HCCL_E_PARA);
}

TEST(ReduceScatterVBirsCore, InterPrepareZeroServerNumRejected)
{
    TestableBirsInter birsInter;
    EXPECT_EQ(birsInter.Prepare(0, 8), HCCL_E_PARA);
}

TEST(ReduceScatterVBirsCore, InterStrideNotAlignedToMinSlice)
{
    TestableBirsInter birsInter;
    ASSERT_EQ(birsInter.Prepare(2, 8), HCCL_SUCCESS);
    birsInter.SetSliceSizes({1000, 300, 999, 7, 500, 100, 200, 300});
    birsInter.CalcLocalStrideSize(8);
    // demand = (rankSize + 2*serverNum) * stride = (8 + 4) * 1000;
    // 边界命中 12000 即证明 stride 未按 16KB 对齐(否则 demand 为 12*16384)
    birsInter.SetScratch(12 * 1000);
    EXPECT_EQ(birsInter.CheckScratchCapacity(8), HCCL_SUCCESS);
    birsInter.SetScratch(12 * 1000 - 1);
    EXPECT_EQ(birsInter.CheckScratchCapacity(8), HCCL_E_PARA);
}

TEST(ReduceScatterVBirsCore, InterZeroSlicesDemandZero)
{
    TestableBirsInter birsInter;
    ASSERT_EQ(birsInter.Prepare(4, 16), HCCL_SUCCESS);
    std::vector<u64> zeroSizes(16, 0);
    birsInter.SetSliceSizes(zeroSizes);
    birsInter.CalcLocalStrideSize(16);
    birsInter.SetScratch(0);
    EXPECT_EQ(birsInter.CheckScratchCapacity(16), HCCL_SUCCESS);
}

} // namespace
