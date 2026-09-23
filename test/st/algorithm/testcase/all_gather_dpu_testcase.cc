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

class ST_ALL_GATHER_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

HcclResult RunHcclAllGather(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    (void)op;
    return HcclAllGather(sendBuf, recvBuf, count, dataType, comm, stream);
}

HcclResult RunCheckAllGather(
    HcclSim::AllRankTaskQueues& taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, HcclReduceOp reduceType)
{
    (void)reduceType;
    return CheckAllGather(taskQueues, rankSize, dataType, dataCount);
}

void RunAllGatherDPUA5(const TopoMeta& topoMeta, u64 sendCount, HcclDataType dataType)
{
    // 算子执行参数设置
    u32 rankSize = CalRankSize(topoMeta); // 参与集合通信的卡数(同topoMeta卡数一致)
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    RunDpuTest(
        topoMeta, rankSize, sendCount, dataType, sendCount * dataTypeSize, sendCount * dataTypeSize * rankSize,
        RunHcclAllGather, RunCheckAllGather, HcclReduceOp::HCCL_REDUCE_RESERVED);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_1_fp32)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 1;                                         // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp32)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_210m_fp32)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 210 * 1024 * 1024;                         // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_210m_fp32_2_2)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};                       // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 210 * 1024 * 1024;                         // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int8)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int16)
{
    TopoMeta topoMeta{{{0}, {0}}};                              // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                           // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int32)
{
    TopoMeta topoMeta{{{0}, {0}}};                              // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                           // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int64)
{
    TopoMeta topoMeta{{{0}, {0}}};                              // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                           // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp16)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp64)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP64; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_bfp16)
{
    TopoMeta topoMeta{{{0}, {0}}};                              // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                           // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e5m2)
{
    TopoMeta topoMeta{{{0}, {0}}};                                // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                             // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e4m3)
{
    TopoMeta topoMeta{{{0}, {0}}};                                // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                             // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e8m0)
{
    TopoMeta topoMeta{{{0}, {0}}};                                // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                             // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_hif8)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// asymmetric topology
TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_asymmetric_10m_hif8)
{
    TopoMeta topoMeta{{{0}, {0, 2, 3}}};                       // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// ========== 正向测试 ==========
// 对称3级: x=4, y=4, z=2, int8
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x4x2_int8)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 对称3级: x=2, y=2, z=2, bfp16
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_2x2x2_bfp16)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 sendCount = 256;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 对称3级: x=8, y=4, z=2, int32
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_8x4x2_int32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 8);
    u64 sendCount = 2048;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 大数据量: 1M
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x4x2_fp32_1m)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// ========== 边界测试 ==========
// xRankSize=1 (L0退化)
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_1x4x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// zRankSize=1 (L2退化, 单超节点) -> 不走OmniPipe
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x4x1_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// yRankSize=1 (L1退化)
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x1x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 最小数据量
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_2x2x2_fp32_send1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 sendCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 大数据量: 25M
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x4x2_int32_16m)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 25 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}
