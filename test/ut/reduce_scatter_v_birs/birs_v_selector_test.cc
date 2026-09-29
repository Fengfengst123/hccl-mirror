/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <gtest/gtest.h>
#include "reduce_scatter_v_birs_selector.h"

using ops_hccl::TopoInfo;
using ops_hccl_experimental::BirsVSelectResult;
using ops_hccl_experimental::BirsVSelectResultToCode;
using ops_hccl_experimental::DecideReduceScatterVBirsAlg;

namespace {
TopoInfo MakeTopo(u32 userRankSize, u32 serverNum, HcclDevType deviceType)
{
    TopoInfo topo{};
    topo.userRankSize = userRankSize;
    topo.serverNum = serverNum;
    topo.deviceType = deviceType;
    return topo;
}

const std::string BIRS_V_ALG_NAME = "ReduceScatterVBIRSExecutor";
} // namespace

TEST(ReduceScatterVBirsSelector, ServerNumZeroIsRejectedInsteadOfDivByZero)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 0, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vRejectServerNumZero);
    EXPECT_TRUE(algName.empty());
}

TEST(ReduceScatterVBirsSelector, DualCardSingleServerIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(2, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vRejectRanksPerServerLT4);
    EXPECT_TRUE(algName.empty());
}

TEST(ReduceScatterVBirsSelector, ThreeRanksPerServerIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(6, 2, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vRejectRanksPerServerLT4);
    EXPECT_TRUE(algName.empty());
}

TEST(ReduceScatterVBirsSelector, RankSizeOneIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(1, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vRejectRankSizeOne);
    EXPECT_TRUE(algName.empty());
}

TEST(ReduceScatterVBirsSelector, FourRanksPerServerIsSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 2, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vSelected);
    EXPECT_EQ(algName, BIRS_V_ALG_NAME);
}

TEST(ReduceScatterVBirsSelector, ExactlyFourRanksPerServerBoundarySelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(4, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vSelected);
    EXPECT_EQ(algName, BIRS_V_ALG_NAME);
}

TEST(ReduceScatterVBirsSelector, MoreThanFourRanksSingleServerIsSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(16, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vSelected);
    EXPECT_EQ(algName, BIRS_V_ALG_NAME);
}

TEST(ReduceScatterVBirsSelector, MoreThanFourRanksPerServerIsSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(16, 4, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vSelected);
    EXPECT_EQ(algName, BIRS_V_ALG_NAME);
}

TEST(ReduceScatterVBirsSelector, NonA3DeviceIsNotSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 2, HcclDevType::DEV_TYPE_910B);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vNotSelected);
    EXPECT_TRUE(algName.empty());
}

TEST(ReduceScatterVBirsSelector, OddRankSizeIsNotSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(9, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterVBirsAlg(topo, algName), BirsVSelectResult::vNotSelected);
    EXPECT_TRUE(algName.empty());
}

TEST(ReduceScatterVBirsSelector, SelectResultToCodeMapsRejectsToErrorCodes)
{
    EXPECT_EQ(BirsVSelectResultToCode(BirsVSelectResult::vSelected), HCCL_SUCCESS);
    EXPECT_EQ(BirsVSelectResultToCode(BirsVSelectResult::vNotSelected), HCCL_SUCCESS);
    EXPECT_EQ(BirsVSelectResultToCode(BirsVSelectResult::vRejectRankSizeOne), HCCL_E_INTERNAL);
    EXPECT_EQ(BirsVSelectResultToCode(BirsVSelectResult::vRejectServerNumZero), HCCL_E_PARA);
    EXPECT_EQ(BirsVSelectResultToCode(BirsVSelectResult::vRejectRanksPerServerLT4), HCCL_E_PARA);
}
