/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "testcase_common.h"

using namespace HcclSim;
using namespace ops_hccl;

class ST_ALL_REDUCE_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_INDEPENDENT_OP");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

void RunAllReduceDPUCase(
    const TopoMeta& topoInfo, const u64 dataCount, const HcclDataType dataType, const u32 dataTypeSize,
    const HcclReduceOp reduceOp, u32 rankSize)
{
    RunDpuTest(
        topoInfo, rankSize, dataCount, dataType, dataCount * dataTypeSize, dataCount * dataTypeSize, HcclAllReduce,
        CheckAllReduce, reduceOp);
}

void RunAllReduceDPU1ShotCase(
    const TopoMeta& topoInfo, const u64 dataCount, const HcclDataType dataType, const u32 dataTypeSize,
    const HcclReduceOp reduceOp)
{
    u32 rankSize = CalRankSize1Shot(topoInfo);
    RunAllReduceDPUCase(topoInfo, dataCount, dataType, dataTypeSize, reduceOp, rankSize);
}

void RunAllReduceOmniPipeDPU(const TopoMeta& topoMeta, u64 dataCount, HcclDataType dataType, HcclReduceOp reduceOp)
{
    u32 rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp, rankSize);
}

TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_2x3_fp32_max_1024)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    u64 dataCount = 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceDPU1ShotCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}

TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_2x3_fp32_max_301K)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    u64 dataCount = 301 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceDPU1ShotCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}

TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_4x4_fp32_sum_1024)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    u64 dataCount = 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPU1ShotCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}

TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_4x4_fp32_sum_301K)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    u64 dataCount = 301 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPU1ShotCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}

TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_8x1_fp32_sum_1024)
{
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    u64 dataCount = 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPU1ShotCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}

TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_8x1_fp32_sum_301K)
{
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    u64 dataCount = 301 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPU1ShotCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}

// ========== 正向测试 ==========

// 对称3级: x=4, y=4, z=2, int8
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_4x4x2_int8)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 dataCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}

// 对称3级: x=2, y=2, z=2, bfp16
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_2x2x2_bfp16)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 dataCount = 256;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}

// 对称3级: x=8, y=4, z=2, int32
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_8x4x2_int32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 8);
    u64 dataCount = 2048;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}

// 大数据量: 1M
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_4x4x2_fp32_1m)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 dataCount = 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}

// ========== 边界测试 ==========

// xRankSize=1 (L0退化)
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_1x4x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    u64 dataCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}

// yRankSize=1 (L1退化)
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_4x1x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    u64 dataCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}

// zRankSize=1 (L2退化, 单超节点) -> 不走omnipipe
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_4x4x1_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 4);
    u64 dataCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}

// 最小数据量
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_2x2x2_fp32_count1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 dataCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}

// 大数据量: 25M
TEST_F(ST_ALL_REDUCE_DPU_TEST, omnipipe_dpu_3level_4x4x2_int32_16m)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 dataCount = 25 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceOmniPipeDPU(topoMeta, dataCount, dataType, reduceOp);
}
