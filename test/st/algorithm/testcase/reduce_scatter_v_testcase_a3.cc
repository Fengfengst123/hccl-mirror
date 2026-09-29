/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "alg_env_config.h"
#include "v_testcase_common.h"

constexpr u32 DATATYPE_SIZE_TABLE_RSV[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(int8_t),
       sizeof(int16_t),
       sizeof(int32_t),
       2,
       sizeof(float),
       sizeof(int64_t),
       sizeof(uint64_t),
       sizeof(uint8_t),
       sizeof(uint16_t),
       sizeof(uint32_t),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

class ST_REDUCE_SCATTER_V_TEST_A2A3 : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

static HcclResult ReduceScatterVDispatchWithOp(
    HcclReduceOp reduceOp, u32 rankId, u64 totalCount, VDataDesTag vDataDes, HcclComm comm, aclrtStream stream)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_RSV[vDataDes.dataType];
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    u64 recvDataCount = vDataDes.counts[rankId];
    u64 sendBufSize = totalCount * dataTypeSize;
    u64 recvBufSize = recvDataCount * dataTypeSize;
    // 零接收 rank: 打桩 aclrtMalloc 拒绝 size=0, 分配单元素哑元缓冲; recvCount 仍按真实值 0 传入
    if (recvBufSize == 0) {
        recvBufSize = dataTypeSize;
    }
    aclError mallocRet = aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
    if (mallocRet != ACL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    mallocRet = aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
    if (mallocRet != ACL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    return HcclReduceScatterV(
        sendBuf, vDataDes.counts.data(), vDataDes.displs.data(), recvBuf, recvDataCount, vDataDes.dataType, reduceOp,
        comm, stream);
}

static HcclResult
ReduceScatterVDispatch(u32 rankId, u64 totalCount, VDataDesTag vDataDes, HcclComm comm, aclrtStream stream)
{
    return ReduceScatterVDispatchWithOp(HcclReduceOp::HCCL_REDUCE_SUM, rankId, totalCount, vDataDes, comm, stream);
}

static HcclResult
ReduceScatterVVerifyWithOp(HcclReduceOp reduceOp, AllRankTaskQueues& taskQueues, u32 rankSize, VDataDesTag vDataDes)
{
    return CheckReduceScatterV(taskQueues, rankSize, reduceOp, vDataDes);
}

static HcclResult ReduceScatterVVerify(AllRankTaskQueues& taskQueues, u32 rankSize, VDataDesTag vDataDes)
{
    return ReduceScatterVVerifyWithOp(HcclReduceOp::HCCL_REDUCE_SUM, taskQueues, rankSize, vDataDes);
}

static auto MakeReduceScatterVDispatch(HcclReduceOp reduceOp)
{
    return [reduceOp](u32 rankId, u64 totalCount, VDataDesTag vDataDes, HcclComm comm, aclrtStream stream) {
        return ReduceScatterVDispatchWithOp(reduceOp, rankId, totalCount, vDataDes, comm, stream);
    };
}

static auto MakeReduceScatterVVerify(HcclReduceOp reduceOp)
{
    return [reduceOp](AllRankTaskQueues& taskQueues, u32 rankSize, VDataDesTag vDataDes) {
        return ReduceScatterVVerifyWithOp(reduceOp, taskQueues, rankSize, vDataDes);
    };
}

static void SetIndependentOpEnv()
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_BIRS_ENABLE", "TRUE", 1);
}

static void RunReduceScatterVMultilevel(
    const TopoMeta& topoInfo, VDataDesTag vDataDes, HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM)
{
    RunVMultilevelTest(
        topoInfo, vDataDes, SetIndependentOpEnv, MakeReduceScatterVDispatch(reduceOp),
        MakeReduceScatterVVerify(reduceOp), HcclDevType::DEV_TYPE_910_93);
}

static VDataDesTag MakeVDataDes(const std::vector<u64>& counts, const std::vector<u64>& displs, HcclDataType dataType)
{
    VDataDesTag vDataDes;
    vDataDes.counts = counts;
    vDataDes.displs = displs;
    vDataDes.dataType = dataType;
    return vDataDes;
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_100_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 0, 0};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_124_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200, 200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600, 800, 1000, 1200, 1400};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_varlen_test)
{
    // 非均匀 counts: 各目标 rank 接收量不同, 覆盖单服务器变体的变长切片与 localStrideSize(max对齐)路径
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 200, 300, 50};
    vDataDes.displs = {0, 100, 300, 600};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_14_16_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    std::vector<u32> args{1, 4, 16};
    for (int i = 0; i < args[0]; i++) {
        SuperPodMeta superPodMeta;
        for (int j = 0; j < args[1]; j++) {
            ServerMeta serverMate;
            for (int k = 0; k < args[2]; k++) {
                serverMate.push_back((unsigned int)k);
            }
            superPodMeta.push_back(serverMate);
        }
        topoMeta.push_back(superPodMeta);
    }

    VDataDesTag vDataDes;

    auto rankSize = 0;
    for (u32 i = 0; i < topoMeta[0].size(); i++)
        rankSize += topoMeta[0][i].size();
    int j = 0;
    for (u32 i = 0; i < rankSize; i++) {
        vDataDes.counts.push_back(200);
        vDataDes.displs.push_back(j);
        j += 200;
    }
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_124_varlen_test)
{
    // 跨服务器 + 非均匀 counts: 覆盖 inter 变体的批量规约与步长 padding 路径
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 200, 150, 250, 80, 120, 300, 160};
    vDataDes.displs = {0, 100, 300, 450, 700, 780, 900, 1200};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_fp16_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({100, 200, 300, 50}, {0, 100, 300, 600}, HcclDataType::HCCL_DATA_TYPE_FP16);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_fp16_max_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({100, 200, 300, 50}, {0, 100, 300, 600}, HcclDataType::HCCL_DATA_TYPE_FP16);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_bf16_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({100, 200, 300, 50}, {0, 100, 300, 600}, HcclDataType::HCCL_DATA_TYPE_BFP16);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_bf16_max_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({100, 200, 300, 50}, {0, 100, 300, 600}, HcclDataType::HCCL_DATA_TYPE_BFP16);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_fp32_sum_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({100, 200, 300, 50}, {0, 100, 300, 600}, HcclDataType::HCCL_DATA_TYPE_FP32);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_fp32_max_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({100, 200, 300, 50}, {0, 100, 300, 600}, HcclDataType::HCCL_DATA_TYPE_FP32);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_fp32_min_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({100, 200, 300, 50}, {0, 100, 300, 600}, HcclDataType::HCCL_DATA_TYPE_FP32);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_MIN);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_fp32_prod_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({100, 200, 300, 50}, {0, 100, 300, 600}, HcclDataType::HCCL_DATA_TYPE_FP32);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_PROD);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_124_fp32_max_varlen_test)
{
    // 跨服务器 + 非均匀 counts + 非 SUM 规约: 覆盖 inter 变体按槽位精确 write-reduce 的 op 透传
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    auto vDataDes = MakeVDataDes(
        {100, 200, 150, 250, 80, 120, 300, 160}, {0, 100, 300, 450, 700, 780, 900, 1200},
        HcclDataType::HCCL_DATA_TYPE_FP32);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_zerocount_varlen_test)
{
    // 非均匀 + 零计数混合: rank0/rank2 接收量为 0, 覆盖空切片路径(零长度搬运不产生输出语义)
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({0, 200, 0, 50}, {0, 0, 200, 200}, HcclDataType::HCCL_DATA_TYPE_INT32);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(ST_REDUCE_SCATTER_V_TEST_A2A3, st_reduce_scatter_v_a3_114_skew_varlen_test)
{
    // 极端不均匀 counts(最大/最小比 4096:1): 覆盖 localStrideSize 按最大切片对齐后的大 padding 路径
    TopoMeta topoMeta{{{0, 1, 2, 3}}};
    auto vDataDes = MakeVDataDes({1, 4096, 2, 1}, {0, 1, 4097, 4099}, HcclDataType::HCCL_DATA_TYPE_INT32);
    RunReduceScatterVMultilevel(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}
