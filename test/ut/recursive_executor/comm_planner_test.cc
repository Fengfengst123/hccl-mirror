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
#include "mesh_comm_planner.h"
#include "nhr_comm_planner.h"
#include "allgather_nhr_template.h"
#include "data_ops.h"

using namespace ops_hccl;

namespace {
// 测试用公共参数：HCCL_BUFFER 类型，带 ccl/output 指针。
DataParams MakeTestParams(
    HcclDataType dataType, u64 sliceCount, u64 scratchStride, u32 inputRank, u64 tailCount = 0,
    u32 tailRankId = INVALID_VALUE_RANKID, BufferType bufType = BufferType::HCCL_BUFFER)
{
    DataParams params;
    params.dataType = dataType;
    params.sliceCount = sliceCount;
    params.tailCount = tailCount;
    params.scratchStride = scratchStride;
    params.globalTailRankId = tailRankId;
    params.ranksForInputData = {inputRank};
    params.outputBufferType = bufType;
    params.cclBufferPtr = reinterpret_cast<void*>(0x1000);
    params.outputBufferPtr = reinterpret_cast<void*>(0x2000);
    return params;
}

// 测试上下文：聚合 ranks/myRank/ranksForOutputData/txRxSlicesLists 声明，消除重复代码。
struct TestCtx {
    std::vector<u32> ranks;
    u32 myRank{0};
    std::vector<u32> ranksForOutputData;
    std::vector<DataSlicesList> txRxSlicesLists;
};
} // namespace

// ============ RunMeshAllGather 测试 ============

// 基本 4 rank mesh allgather：验证 txRxSlicesLists 的条目数和数据来源
TEST(MeshCommPlannerTest, RunMeshAllGatherBasic4Ranks)
{
    DataParams params = MakeTestParams(HcclDataType::HCCL_DATA_TYPE_INT32, 100, 100 * sizeof(int32_t), 0);
    TestCtx ctx{{0, 1, 2, 3}, 0};

    HcclResult ret = RunMeshAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 3u);
    EXPECT_EQ(ctx.ranksForOutputData.size(), 4u);
}

// 单 rank：不需要 sendRecv
TEST(MeshCommPlannerTest, RunMeshAllGatherSingleRank)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    params.sliceCount = 10;
    params.scratchStride = 10;
    params.ranksForInputData = {0};
    TestCtx ctx{{0}};

    HcclResult ret = RunMeshAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 0u);
    EXPECT_EQ(ctx.ranksForOutputData.size(), 1u);
}

// 8 rank with tail
TEST(MeshCommPlannerTest, RunMeshAllGather8RanksWithTail)
{
    DataParams params = MakeTestParams(HcclDataType::HCCL_DATA_TYPE_FP32, 100, 100 * sizeof(float), 3, 5, 7);
    TestCtx ctx{{0, 1, 2, 3, 4, 5, 6, 7}, 3};

    HcclResult ret = RunMeshAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 7u);
    EXPECT_EQ(ctx.ranksForOutputData.size(), 8u);
}

// directToOutput 路径
TEST(MeshCommPlannerTest, RunMeshAllGatherDirectToOutput)
{
    DataParams params = MakeTestParams(
        HcclDataType::HCCL_DATA_TYPE_INT32, 50, 50 * sizeof(int32_t), 2, 0, INVALID_VALUE_RANKID, BufferType::OUTPUT);
    params.dataStride = 50 * sizeof(int32_t);
    TestCtx ctx{{0, 1, 2, 3}, 2};

    HcclResult ret = RunMeshAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 3u);

    // 验证 ranksForOutputData 布局
    ASSERT_EQ(ctx.ranksForOutputData.size(), 4u);
    EXPECT_EQ(ctx.ranksForOutputData[0], 0u);
    EXPECT_EQ(ctx.ranksForOutputData[1], 1u);
    EXPECT_EQ(ctx.ranksForOutputData[2], 2u);
    EXPECT_EQ(ctx.ranksForOutputData[3], 3u);

    // 验证 dstRankId 顺序（跳过 myRank=2，按 ranks 顺序）
    ASSERT_GE(ctx.txRxSlicesLists.size(), 3u);
    EXPECT_EQ(ctx.txRxSlicesLists[0].dstRankId_, 0u);
    EXPECT_EQ(ctx.txRxSlicesLists[1].dstRankId_, 1u);
    EXPECT_EQ(ctx.txRxSlicesLists[2].dstRankId_, 3u);

    // tx src offset = sliceOffset + myRank * scratchStride = 0 + 2*200 = 400
    const u64 txSrcOffset = 400;
    for (size_t i = 0; i < ctx.txRxSlicesLists.size(); ++i) {
        ASSERT_FALSE(ctx.txRxSlicesLists[i].txSlicesList_.srcSlices_.empty());
        EXPECT_EQ(ctx.txRxSlicesLists[i].txSlicesList_.srcSlices_[0].offset_, txSrcOffset)
            << "tx src offset mismatch at index " << i;
    }

    // rx dst offset = dataOffset + sliceOffset + outIdx * dataStride
    // rank0→outIdx=0→offset=0, rank1→outIdx=1→offset=200, rank3→outIdx=3→offset=600
    ASSERT_FALSE(ctx.txRxSlicesLists[0].rxSlicesList_.dstSlices_.empty());
    EXPECT_EQ(ctx.txRxSlicesLists[0].rxSlicesList_.dstSlices_[0].offset_, 0u);
    ASSERT_FALSE(ctx.txRxSlicesLists[1].rxSlicesList_.dstSlices_.empty());
    EXPECT_EQ(ctx.txRxSlicesLists[1].rxSlicesList_.dstSlices_[0].offset_, 200u);
    ASSERT_FALSE(ctx.txRxSlicesLists[2].rxSlicesList_.dstSlices_.empty());
    EXPECT_EQ(ctx.txRxSlicesLists[2].rxSlicesList_.dstSlices_[0].offset_, 600u);
}

// 空 ranksForInputData 应报错
TEST(MeshCommPlannerTest, RunMeshAllGatherEmptyInputRanks)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    params.sliceCount = 10;
    params.ranksForInputData = {};
    TestCtx ctx{{0, 1}};

    HcclResult ret = RunMeshAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ============ RunNhrAllGather 测试 ============

// 4 rank NHR：验证步数 = log2(4) = 2
TEST(NhrCommPlannerTest, RunNhrAllGather4Ranks)
{
    DataParams params = MakeTestParams(HcclDataType::HCCL_DATA_TYPE_INT32, 100, 100 * sizeof(int32_t), 0);
    TestCtx ctx{{0, 1, 2, 3}};

    HcclResult ret = RunNhrAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 2u);
    EXPECT_EQ(ctx.ranksForOutputData.size(), 4u);
}

// 8 rank NHR：步数 = log2(8) = 3
TEST(NhrCommPlannerTest, RunNhrAllGather8Ranks)
{
    DataParams params = MakeTestParams(HcclDataType::HCCL_DATA_TYPE_FP32, 200, 200 * sizeof(float), 5);
    TestCtx ctx{{0, 1, 2, 3, 4, 5, 6, 7}, 5};

    HcclResult ret = RunNhrAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 3u);
    EXPECT_EQ(ctx.ranksForOutputData.size(), 8u);
}

// 2 rank NHR：步数 = log2(2) = 1
TEST(NhrCommPlannerTest, RunNhrAllGather2Ranks)
{
    DataParams params = MakeTestParams(HcclDataType::HCCL_DATA_TYPE_INT8, 100, 100, 0);
    TestCtx ctx{{0, 1}};

    HcclResult ret = RunNhrAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 1u);
}

// NHR with tail
TEST(NhrCommPlannerTest, RunNhrAllGather4RanksWithTail)
{
    DataParams params = MakeTestParams(HcclDataType::HCCL_DATA_TYPE_INT32, 100, 100 * sizeof(int32_t), 0, 3, 3);
    TestCtx ctx{{0, 1, 2, 3}};

    HcclResult ret = RunNhrAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 2u);
}

// NHR lastStepRxRanks 输出
TEST(NhrCommPlannerTest, RunNhrAllGatherLastStepRxRanks)
{
    DataParams params = MakeTestParams(
        HcclDataType::HCCL_DATA_TYPE_INT32, 100, 100 * sizeof(int32_t), 0, 0, INVALID_VALUE_RANKID, BufferType::OUTPUT);
    params.dataStride = 100 * sizeof(int32_t);
    TestCtx ctx{{0, 1, 2, 3}};
    std::vector<u32> lastStepRxRanks;

    HcclResult ret
        = RunNhrAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists, &lastStepRxRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(lastStepRxRanks.empty());
    // 4 rank、myRank=0、nSteps=2 时，末步 delta=1、recvFromAlgRank=3，
    // i=0: rxAlgRank=3 → connectedInputRanks={0+3}={3}
    // i=1: rxAlgRank=(3+4-2)%4=1 → connectedInputRanks={0+1}={1}
    // 故 lastStepRxRanks 应恰为 {3, 1}
    EXPECT_EQ(lastStepRxRanks.size(), 2u);
    EXPECT_EQ(lastStepRxRanks[0], 3u);
    EXPECT_EQ(lastStepRxRanks[1], 1u);
}

// NHR 单 rank
TEST(NhrCommPlannerTest, RunNhrAllGatherSingleRank)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    params.sliceCount = 10;
    params.ranksForInputData = {0};
    TestCtx ctx{{0}};

    HcclResult ret = RunNhrAllGather(params, ctx.ranks, ctx.myRank, ctx.ranksForOutputData, ctx.txRxSlicesLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx.txRxSlicesLists.size(), 0u);
}

// ============ CanReadLastStepToOutput 测试 ============

TEST(AllGatherNhrTemplateTest, CanReadLastStepToOutputTrue)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    params.sliceCount = 256 * 1024 + 1; // int32 时 > 1MB，满足 DMA 消减阈值
    params.outputBufferType = BufferType::OUTPUT;
    params.enableRemoteMemAccess = false;

    EXPECT_TRUE(AllGatherNhrTemplate::CanReadLastStepToOutput(params));
}

TEST(AllGatherNhrTemplateTest, CanReadLastStepToOutputFalseRemoteAccess)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    params.sliceCount = 256 * 1024;
    params.outputBufferType = BufferType::OUTPUT;
    params.enableRemoteMemAccess = true;

    EXPECT_FALSE(AllGatherNhrTemplate::CanReadLastStepToOutput(params));
}

TEST(AllGatherNhrTemplateTest, CanReadLastStepToOutputFalseHcclBuffer)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    params.sliceCount = 256 * 1024;
    params.outputBufferType = BufferType::HCCL_BUFFER;
    params.enableRemoteMemAccess = false;

    EXPECT_FALSE(AllGatherNhrTemplate::CanReadLastStepToOutput(params));
}

TEST(AllGatherNhrTemplateTest, CanReadLastStepToOutputFalseSmallData)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    params.sliceCount = 100; // 远小于 1MB
    params.outputBufferType = BufferType::OUTPUT;
    params.enableRemoteMemAccess = false;

    EXPECT_FALSE(AllGatherNhrTemplate::CanReadLastStepToOutput(params));
}

// sliceSize 恰等于 1MB 边界：<= 语义下应返回 false
TEST(AllGatherNhrTemplateTest, CanReadLastStepToOutputFalseExactBoundary)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    params.sliceCount = 256 * 1024; // 256K * 4B = 1MB，恰等于 DMA_REDUCTION_MIN_DATA_SIZE
    params.outputBufferType = BufferType::OUTPUT;
    params.enableRemoteMemAccess = false;

    EXPECT_FALSE(AllGatherNhrTemplate::CanReadLastStepToOutput(params));
}
