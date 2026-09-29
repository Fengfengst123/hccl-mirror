/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <climits>
#include <string>
#include <vector>
#include "alg_parse.h"

using namespace ops_hccl;

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

// 构造 CostModel：所有算法 count 初始化为 1
static CostModel BuildCostModel(const std::vector<std::string>& algoNames)
{
    CostModel cm;
    cm.count = static_cast<int>(algoNames.size());
    cm.costAlgoParams = new CostAlgoParams[algoNames.size()];
    for (size_t i = 0; i < algoNames.size(); i++) {
        cm.costAlgoParams[i].algName = strdup(algoNames[i].c_str());
        cm.costAlgoParams[i].param = nullptr;
        cm.costAlgoParams[i].count = 1;
    }
    return cm;
}

// 释放 CostModel
static void FreeCostModel(CostModel& cm)
{
    for (int i = 0; i < cm.count; i++) {
        free(const_cast<char*>(cm.costAlgoParams[i].algName));
    }
    delete[] cm.costAlgoParams;
    cm.costAlgoParams = nullptr;
    cm.count = 0;
}

// 按算法名查找 count（0=被排除，1=正常，-1=不存在）
static int GetAlgoCount(const CostModel& cm, const std::string& name)
{
    for (int i = 0; i < cm.count; i++) {
        if (cm.costAlgoParams[i].algName && name == cm.costAlgoParams[i].algName) {
            return cm.costAlgoParams[i].count;
        }
    }
    return -1; // 未找到
}

// 按算法名查找 hcclAlgoPriority（INT_MIN=不存在）
static int GetAlgoPriority(const CostModel& cm, const std::string& name)
{
    for (int i = 0; i < cm.count; i++) {
        if (cm.costAlgoParams[i].algName && name == cm.costAlgoParams[i].algName) {
            return cm.costAlgoParams[i].hcclAlgoPriority;
        }
    }
    return INT_MIN;
}

// 算法名列表（costModel 标准格式：ENGINE_TYPES.second + OP_TYPES.second + EXECUTOR_TYPES.second +
// ALGO_TYPES.second...） 覆盖全部 ALGO_TYPES 值：Mesh, Mesh2Die, MeshOneShot, MeshTwoShot, MeshConcur,
//   MeshMultiLink, MeshChunk, MeshChunkTwoShot, NHR, NHRMultiLink
// 覆盖 EXECUTOR_TYPES 值：Sole, Sequence, Parallel, Concur（PipeLine 在单独测试中覆盖）
static const std::vector<std::string> ALGO_NAMES = {
    "AicpuAllReduceSoleMesh2Die",           // 0  Sole + Mesh2Die
    "AivAllReduceParallelMeshMultiLinkNHR", // 1  Parallel + MeshMultiLink + NHR
    "CcuMSAllReduceSequenceMeshOneShotNHR", // 2  Sequence + MeshOneShot + NHR
    "CcuMSAllReduceSoleNHRMultiLink",
    "AivAllGatherParallelMeshTwoShotMeshChunk",        // 3  Parallel + MeshTwoShot + MeshChunk
    "CcuMSBroadcastConcurMeshChunkNHRMultiLink",       // 4  Concur + MeshChunk + NHRMultiLink
    "AicpuAllToAllSoleMeshConcur",                     // 5  Sole + MeshConcur
    "AicpuAllToAllSoleNHRMultiLink",                   // 5  Sole + MeshConcur
    "DpuAllToAllSequenceMeshMeshNHR",                  // 5  Sole + MeshConcur
    "AivAllToAllVSequenceMeshChunkTwoShotNHRMesh",     // 6  Sequence + MeshChunkTwoShot + NHR + Mesh
    "AivReduceScatterSequenceNHRMultiLinkMeshOneShot", // 7  Sequence + NHRMultiLink + MeshOneShot
    "CcuMSReduceSequenceNHRMeshTwoShot",               // 8  Sequence + NHR + MeshTwoShot
    "CcuMSScatterConcurMeshOneShotMesh",               // 9  Concur + MeshOneShot + Mesh
    "AicpuAllGatherSequenceMeshMeshNHR",               // 10 Sequence + Mesh + Mesh + NHR (matches not(nhr))
    "AicpuAllGatherSequenceMeshNHRNHR",                // 11 Sequence + Mesh + NHR + NHR (excluded by not(nhr))
};

// selector 传入的候选引擎前缀（ENGINE_TYPES.second 格式）
static const std::vector<std::string> ENGINE_TYPES = {"Aicpu", "Aiv", "CcuMS"};

// ---------------------------------------------------------------------------
// 测试 fixture
// ---------------------------------------------------------------------------
class UpdateCostModelTest : public testing::Test {
protected:
    void SetUp() override { cm_ = BuildCostModel(ALGO_NAMES); }
    void TearDown() override { FreeCostModel(cm_); }
    CostModel cm_;
};

// ---------------------------------------------------------------------------
// 测试 1：正向精确匹配——allReduce:sole{mesh2die}
// 匹配 #0 打优先级 1，同 OpType 其他算法不动（不强制过滤）
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, PositiveExactMatch)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:sole{mesh2die}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // 匹配的算法打 hcclAlgoPriority=1，count 不变
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 1);
    EXPECT_EQ(GetAlgoCount(cm_, "AicpuAllReduceSoleMesh2Die"), 1);
    // 同 OpType 下未匹配的算法不受影响（优先级 0，count 保持 1）
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllReduceParallelMeshMultiLinkNHR"), 0);
    EXPECT_EQ(GetAlgoCount(cm_, "AivAllReduceParallelMeshMultiLinkNHR"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSAllReduceSequenceMeshOneShotNHR"), 0);
    // 其他 OpType 的算法不受影响
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllGatherParallelMeshTwoShotMeshChunk"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllToAllSoleMeshConcur"), 0);
}

// ---------------------------------------------------------------------------
// 测试 2：正向多 level 匹配——allReduce:parallel{meshmultilink,nhr}
// 匹配 #1 打优先级 1，其他算法不动
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, PositiveMultiLevelMatch)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:parallel{meshmultilink,nhr}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllReduceParallelMeshMultiLinkNHR"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSAllReduceSequenceMeshOneShotNHR"), 0);
}

// ---------------------------------------------------------------------------
// 测试 3：反向匹配——allReduce:not(sole{mesh2die})
// #0 打优先级 -1（否定层），同 OpType 其他算法不动
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, NegativeExecutorMatch)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:not(sole{mesh2die})"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // #0 打 hcclAlgoPriority=-1，count 不变
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), -1);
    EXPECT_EQ(GetAlgoCount(cm_, "AicpuAllReduceSoleMesh2Die"), 1);
    // isExecNegated 标记 OpType 已匹配，其他算法不受影响
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllReduceParallelMeshMultiLinkNHR"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSAllReduceSequenceMeshOneShotNHR"), 0);
}

// ---------------------------------------------------------------------------
// 测试 4：全局配置（opType 为空）——sole{meshconcur}
// 遍历所有 OpType，命中的打优先级 1，其余不动
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, GlobalConfig)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("sole{meshconcur}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // allToAll 命中 #5 → 优先级 1
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllToAllSoleMeshConcur"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "DpuAllToAllSequenceMeshMeshNHR"), 0);

    // 未命中的 OpType 算法不受影响
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSBroadcastConcurMeshChunkNHRMultiLink"), 0);
}

// ---------------------------------------------------------------------------
// 测试 5：优先级——后面的规则优先级高
// allReduce:sole{mesh2die} 先匹配 #0，但 allReduce:parallel{meshmultilink,nhr} 优先级高匹配 #1
// 同 OpType 只有最后一条生效
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, PriorityOrder)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:sole{mesh2die};allReduce:parallel{meshmultilink,nhr}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // 优先级高的 parallel{meshmultilink,nhr} 命中 #1
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllReduceParallelMeshMultiLinkNHR"), 1);
    // 同 OpType 更早的 sole{mesh2die} 不再生效
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSAllReduceSequenceMeshOneShotNHR"), 0);
}

// ---------------------------------------------------------------------------
// 测试 6：全局否定 + 指定 OpType 启用
// not(nhrmultilink) 全局否定，allReduce:sole{mesh2die} 优先级高
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, GlobalExcludeThenOpTypeEnable)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("not(nhrmultilink);allReduce:sole{mesh2die}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // allReduce 被 sole{mesh2die} 优先命中
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 1);
    // 同 OpType 下其他算法不动
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllReduceParallelMeshMultiLinkNHR"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSAllReduceSequenceMeshOneShotNHR"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSAllReduceSoleNHRMultiLink"), 0);

    // 全局 not(nhrmultilink) 对未认领的 OpType 生效 → allToAll 的 Sole+NHRMultiLink 打 -1
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllToAllSoleMeshConcur"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllToAllSoleNHRMultiLink"), -1);
}

// ---------------------------------------------------------------------------
// 测试 7：前缀匹配（algoList 为空）——allReduce:sole{}
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, PrefixMatch)
{
    std::vector<std::string> names = ALGO_NAMES;
    names.push_back("AivAllReduceSoleMeshConcur"); // 12: sole 前缀，不同引擎
    CostModel cm = BuildCostModel(names);

    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:sole{}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm, ENGINE_TYPES), HCCL_SUCCESS);

    // allReduce + sole 前缀的算法都命中
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReduceSoleMesh2Die"), 1);
    EXPECT_EQ(GetAlgoPriority(cm, "AivAllReduceSoleMeshConcur"), 1);
    // 非 sole 的 allReduce 算法不动
    EXPECT_EQ(GetAlgoPriority(cm, "AivAllReduceParallelMeshMultiLinkNHR"), 0);
    EXPECT_EQ(GetAlgoPriority(cm, "CcuMSAllReduceSequenceMeshOneShotNHR"), 0);

    FreeCostModel(cm);
}

// ---------------------------------------------------------------------------
// 测试 8：多 OpType 匹配，覆盖多种 ALGO_TYPES 和 EXECUTOR_TYPES
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, MultipleOpTypesMatched)
{
    HcclAlgoParser parser;
    std::string config = "allReduce:sole{mesh2die};"
                         "allGather:parallel{meshtwoshot,meshchunk};"
                         "reduceScatter:sequence{nhrmultilink,meshoneshot};"
                         "broadcast:concur{meshchunk,nhrmultilink};"
                         "alltoall:sole{meshconcur};"
                         "alltoallv:sequence{meshchunktwoshot,nhr,mesh};"
                         "scatter:concur{meshoneshot,mesh};"
                         "reduce:sequence{nhr,meshtwoshot}";
    ASSERT_EQ(parser.Parser(config), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // 每个 OpType 都有命中的算法打 1
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllGatherParallelMeshTwoShotMeshChunk"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "AivReduceScatterSequenceNHRMultiLinkMeshOneShot"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSBroadcastConcurMeshChunkNHRMultiLink"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllToAllSoleMeshConcur"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllToAllVSequenceMeshChunkTwoShotNHRMesh"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSScatterConcurMeshOneShotMesh"), 1);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSReduceSequenceNHRMeshTwoShot"), 1);

    // 同 OpType 下未命中的算法不动
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllReduceParallelMeshMultiLinkNHR"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSAllReduceSequenceMeshOneShotNHR"), 0);
}

// ---------------------------------------------------------------------------
// 测试 9：空 parser
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, EmptyParser)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser(""), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 0);
}

// ---------------------------------------------------------------------------
// 测试 10：有效但无匹配的 executorType——pipeline{mesh2die}
// pipeline 在 EXECUTOR_TYPES 中合法，但 costModel 中无对应算法
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, NoMatchExecutorType)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:pipeline{mesh2die}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 0);
}

// ---------------------------------------------------------------------------
// 测试 11：hasNegatedAlgo 模糊匹配——sequence{mesh,not(nhr),nhr}
// #10 命中（level1=Mesh≠NHR）打 1，#11 不命中，#3 同 OpType 未命中不动
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, NegatedAlgoFuzzyMatch)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allGather:sequence{mesh,not(nhr),nhr}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // #10 命中（level1=Mesh，不是 NHR）→ 优先级 1
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllGatherSequenceMeshMeshNHR"), 1);
    // #11 不命中（level1=NHR，被 not 排除）
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllGatherSequenceMeshNHRNHR"), 0);
    // #3 也属于 allGather 但未命中 → 不动
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllGatherParallelMeshTwoShotMeshChunk"), 0);
}

// ---------------------------------------------------------------------------
// 测试 12：hasNegatedAlgo 命中后同 OpType 其他算法不动
// 使用 Mesh2Die 作为 level1 多样化取非匹配
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, NegatedAlgoExcludeOthers)
{
    // 构造有多个 allReduce+sequence 算法的 costModel
    std::vector<std::string> names = ALGO_NAMES;
    names.push_back("AicpuAllReduceSequenceMeshMesh2Die"); // 12: Sequence + Mesh + Mesh2Die (matches not(nhr))
    names.push_back("AivAllReduceSequenceMeshNHR");        // 13: Sequence + Mesh + NHR (excluded by not(nhr))
    CostModel cm = BuildCostModel(names);

    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:sequence{mesh,not(nhr)}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm, ENGINE_TYPES), HCCL_SUCCESS);

    // 命中 Mesh+Mesh2Die（level1=Mesh2Die，不是 NHR）
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReduceSequenceMeshMesh2Die"), 1);
    // 不命中 Mesh+NHR（level1=NHR，被 not 排除）
    EXPECT_EQ(GetAlgoPriority(cm, "AivAllReduceSequenceMeshNHR"), 0);
    // 同 OpType 其他算法不动
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReduceSoleMesh2Die"), 0);

    FreeCostModel(cm);
}

// ---------------------------------------------------------------------------
// Send/Recv 专用 fixture
// 规则 4.6：send/recv 算法不参与 executor 否定排除（不打 -1，保持未配置）
// ---------------------------------------------------------------------------
class UpdateCostModelSendRecvTest : public testing::Test {
protected:
    void SetUp() override
    {
        std::vector<std::string> names = {
            // Send/Recv 算法
            "AicpuSendSoleMesh",          // 0
            "AicpuRecvSoleMesh",          // 1
            "AicpuBatchSendRecvSoleMesh", // 2
            // 非 Send/Recv 对照
            "AicpuAllReduceSoleMesh",        // 3
            "AicpuAllReduceSequenceMeshNHR", // 4
        };
        cm_ = BuildCostModel(names);
    }
    void TearDown() override { FreeCostModel(cm_); }
    CostModel cm_;
};

// ---------------------------------------------------------------------------
// 测试 1：全局 not(sole{}) 前缀匹配——Send/Recv 不打否定标记
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelSendRecvTest, NegatedExecutorPrefixPreservesSendRecv)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("not(sole{})"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // Send/Recv 不参与 executor 否定: 保持未配置(0)且存活
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuSendSoleMesh"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuRecvSoleMesh"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuBatchSendRecvSoleMesh"), 0);
    EXPECT_EQ(GetAlgoCount(cm_, "AicpuSendSoleMesh"), 1);
    // 非 Send/Recv 的 sole 算法打否定标记(-1), 新语义不杀 count
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh"), -1);
    EXPECT_EQ(GetAlgoCount(cm_, "AicpuAllReduceSoleMesh"), 1);
    // 非 sole 算法不受影响
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSequenceMeshNHR"), 0);
}

// ---------------------------------------------------------------------------
// 测试 2：全局 not(sole{mesh}) 精确匹配——Send/Recv 不打否定标记
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelSendRecvTest, NegatedExecutorExactMatchPreservesSendRecv)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("not(sole{mesh})"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // Send/Recv 不参与 executor 否定: 保持未配置(0)且存活
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuSendSoleMesh"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuRecvSoleMesh"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuBatchSendRecvSoleMesh"), 0);
    EXPECT_EQ(GetAlgoCount(cm_, "AicpuSendSoleMesh"), 1);
    // 非 Send/Recv 的 sole{mesh} 算法打否定标记(-1), 新语义不杀 count
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh"), -1);
    EXPECT_EQ(GetAlgoCount(cm_, "AicpuAllReduceSoleMesh"), 1);
    // 非 sole{mesh} 算法不受影响
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSequenceMeshNHR"), 0);
}

// ---------------------------------------------------------------------------
// 测试 3：显式 opType 取反——allreduce:not(sole{mesh}) 不影响 Send/Recv
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelSendRecvTest, ScopedNegationPreservesSendRecv)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:not(sole{mesh})"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // Send/Recv 不受 allReduce 取反影响, 保持未配置(0)
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuSendSoleMesh"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuRecvSoleMesh"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuBatchSendRecvSoleMesh"), 0);
    // allReduce 的 sole{mesh} 打否定标记(-1)
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh"), -1);
    // allReduce 的非 sole{mesh} 不受影响
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSequenceMeshNHR"), 0);
}

// ---------------------------------------------------------------------------
// 测试 13：isExecNegated 找到算法后标记 OpType 已匹配
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, NegatedExecutorMarksOpType)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:not(parallel{meshmultilink,nhr})"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm_, ENGINE_TYPES), HCCL_SUCCESS);

    // #1 打 -1
    EXPECT_EQ(GetAlgoPriority(cm_, "AivAllReduceParallelMeshMultiLinkNHR"), -1);
    // isExecNegated 标记 OpType，其他算法不受影响
    EXPECT_EQ(GetAlgoPriority(cm_, "AicpuAllReduceSoleMesh2Die"), 0);
    EXPECT_EQ(GetAlgoPriority(cm_, "CcuMSAllReduceSequenceMeshOneShotNHR"), 0);
}

// ---------------------------------------------------------------------------
// 测试 14：PipeLine 引擎匹配（覆盖 EXECUTOR_TYPES 中的 pipeline）
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, PipelineMatch)
{
    std::vector<std::string> names = ALGO_NAMES;
    names.push_back("AicpuAllReducePipeLineMesh2Die"); // 12: PipeLine executor
    CostModel cm = BuildCostModel(names);

    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:pipeline{mesh2die}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm, ENGINE_TYPES), HCCL_SUCCESS);

    // pipeline + mesh2die 命中
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReducePipeLineMesh2Die"), 1);
    // 同 OpType 其他算法不动
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReduceSoleMesh2Die"), 0);
    EXPECT_EQ(GetAlgoPriority(cm, "AivAllReduceParallelMeshMultiLinkNHR"), 0);

    FreeCostModel(cm);
}

// ---------------------------------------------------------------------------
// 测试 15：多个 level 同时有 not（覆盖 ALGO_TYPES 多样值）
// allReduce:sequence{mesh2die,not(nhr),not(meshoneshot)}
// ---------------------------------------------------------------------------
TEST_F(UpdateCostModelTest, MultipleNegatedAlgoLevels)
{
    std::vector<std::string> names = ALGO_NAMES;
    names.push_back(
        "AicpuAllReduceSequenceMesh2DieMeshMesh2Die"); // 12: 命中（level1=Mesh≠NHR, level2=Mesh2Die≠MeshOneShot）
    names.push_back("AicpuAllReduceSequenceMesh2DieNHRMesh"); // 13: 不命中（level1=NHR，被 not(nhr) 排除）
    names.push_back(
        "AicpuAllReduceSequenceMesh2DieMeshMeshOneShot"); // 14: 不命中（level2=MeshOneShot，被 not(meshoneshot) 排除）
    CostModel cm = BuildCostModel(names);

    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("allReduce:sequence{mesh2die,not(nhr),not(meshoneshot)}"), HCCL_SUCCESS);

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, cm, ENGINE_TYPES), HCCL_SUCCESS);

    // level1=Mesh(≠NHR), level2=Mesh2Die(≠MeshOneShot) → 命中
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReduceSequenceMesh2DieMeshMesh2Die"), 1);
    // level1=NHR → 被 not(nhr) 排除
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReduceSequenceMesh2DieNHRMesh"), 0);
    // level2=MeshOneShot → 被 not(meshoneshot) 排除
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReduceSequenceMesh2DieMeshMeshOneShot"), 0);
    // 同 OpType 其他算法不动
    EXPECT_EQ(GetAlgoPriority(cm, "AicpuAllReduceSoleMesh2Die"), 0);

    FreeCostModel(cm);
}
