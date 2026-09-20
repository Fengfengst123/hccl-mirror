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
#include "data_ops.h"

using namespace ops_hccl;

// 测试 GetAlgRank
TEST(DataOpsTest, GetAlgRankNormal)
{
    std::vector<u32> ranks = {10, 20, 30, 40};
    u32 algRank = 0;
    HcclResult ret = GetAlgRank(20, ranks, algRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algRank, 1u);
}

TEST(DataOpsTest, GetAlgRankNotFound)
{
    std::vector<u32> ranks = {10, 20, 30};
    u32 algRank = 0;
    HcclResult ret = GetAlgRank(99, ranks, algRank);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 测试 CheckInputDataRanks
TEST(DataOpsTest, CheckInputDataRanksEmpty)
{
    DataParams params;
    auto ret = CheckInputDataRanks(params, "test");
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST(DataOpsTest, CheckInputDataRanksNonEmpty)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    params.ranksForInputData = {0, 1};
    auto ret = CheckInputDataRanks(params, "test");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试 CalcDataSizeInfo
TEST(DataOpsTest, CalcDataSizeInfoNoTail)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    params.sliceCount = 100;
    params.tailCount = 0;
    auto info = CalcDataSizeInfo(params);
    EXPECT_EQ(info.dataTypeSize, sizeof(int32_t));
    EXPECT_EQ(info.sliceSize, 100u * sizeof(int32_t));
    EXPECT_EQ(info.tailSize, info.sliceSize);
}

TEST(DataOpsTest, CalcDataSizeInfoWithTail)
{
    DataParams params;
    params.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    params.sliceCount = 100;
    params.tailCount = 5;
    auto info = CalcDataSizeInfo(params);
    EXPECT_EQ(info.sliceSize, 100u * sizeof(int32_t));
    EXPECT_EQ(info.tailSize, (100u + 5u) * sizeof(int32_t));
}

// 测试 CalcRankDataSize
TEST(DataOpsTest, CalcRankDataSizeNoTail)
{
    DataSizeInfo info{4, 400, 400};
    EXPECT_EQ(CalcRankDataSize(info, 0, INVALID_VALUE_RANKID), 400u);
    EXPECT_EQ(CalcRankDataSize(info, 5, INVALID_VALUE_RANKID), 400u);
}

TEST(DataOpsTest, CalcRankDataSizeTailRank)
{
    DataSizeInfo info{4, 400, 420};
    u32 tailRank = 3;
    EXPECT_EQ(CalcRankDataSize(info, 3, tailRank), 420u);
    EXPECT_EQ(CalcRankDataSize(info, 0, tailRank), 400u);
}

// 测试 CalcRanksForOutput
TEST(DataOpsTest, CalcRanksForOutput)
{
    std::vector<u32> ranksForInput = {0};
    std::vector<u32> subCommRanks = {0, 1, 2};
    std::vector<u32> output;
    HcclResult ret = CalcRanksForOutput(ranksForInput, subCommRanks, 0, output);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(output.size(), 3u);
    // output should be sorted: {0, 1, 2}
    EXPECT_EQ(output[0], 0u);
    EXPECT_EQ(output[1], 1u);
    EXPECT_EQ(output[2], 2u);
}

TEST(DataOpsTest, CalcRanksForOutputOffset)
{
    std::vector<u32> ranksForInput = {2};
    std::vector<u32> subCommRanks = {1, 2, 3};
    std::vector<u32> output;
    HcclResult ret = CalcRanksForOutput(ranksForInput, subCommRanks, 2, output);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(output.size(), 3u);
    // rankOffset = subCommRank - myRank; output = inputRank + rankOffset
    // subCommRank=1: offset=-1, output=2-1=1
    // subCommRank=2: offset=0, output=2+0=2
    // subCommRank=3: offset=1, output=2+1=3
    // sorted: {1, 2, 3}
    EXPECT_EQ(output[0], 1u);
    EXPECT_EQ(output[1], 2u);
    EXPECT_EQ(output[2], 3u);
}

// 测试 CalcDataSplitByPortGroup
TEST(DataOpsTest, CalcDataSplitByPortGroupSingleChannel)
{
    std::vector<ChannelInfo> channels(1);
    channels[0].portGroupSize = 1;
    std::vector<u64> split, offset;
    HcclResult ret = CalcDataSplitByPortGroup(100, 4, channels, split, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(split.size(), 1u);
    EXPECT_EQ(split[0], 100u);
    EXPECT_EQ(offset[0], 0u);
}

TEST(DataOpsTest, CalcDataSplitByPortGroupTwoChannels)
{
    std::vector<ChannelInfo> channels(2);
    channels[0].portGroupSize = 1;
    channels[1].portGroupSize = 1;
    std::vector<u64> split, offset;
    HcclResult ret = CalcDataSplitByPortGroup(100, 4, channels, split, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(split.size(), 2u);
    EXPECT_EQ(offset[0], 0u);
    EXPECT_EQ(offset[1], split[0]);
    EXPECT_EQ(split[0] + split[1], 100u);
}
