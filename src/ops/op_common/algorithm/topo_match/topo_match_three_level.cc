/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_three_level.h"
#include <algorithm>
#include "log.h"

namespace ops_hccl {

namespace {
    // 校验 level 对称并取维度：GLOBAL 看 instList 是否全等；LOCAL 视为对称
    HcclResult ValidateLevelAndCalcDim(const PhysicalLevelInfo& level, bool& symmetricOut, u32& dim)
    {
        if (level.view == PhysicalLevelView::LOCAL) {
            // LOCAL 仅包含当前实例，无全局分区信息；按当前实例计算维度，本函数不判断跨实例对称性
            dim = static_cast<u32>(level.localRanks.size());
            symmetricOut = true;
            return HcclResult::HCCL_SUCCESS;
        }
        if (!IsInstListSymmetric(level.instSizeListByLayer)) {
            symmetricOut = false;
            return HcclResult::HCCL_SUCCESS;
        }
        symmetricOut = true;
        dim = static_cast<u32>(level.localRanks.size());
        return HcclResult::HCCL_SUCCESS;
    }

    // 计算 ThreeLevel 维度：对称场景维持原逻辑；Layer1 非对称时按 GCD 构造虚拟 POD
    HcclResult CalcDimsAndCheckSymmetry(
        const TopoInfoWithNetLayerDetails& topoInfo, u32 phys0, u32 phys1, u32& d0, u32& d1, u32& d2)
    {
        const auto& physicalLevels = topoInfo.physicalLevels;
        const u32 userRankSize = topoInfo.userRankSize;
        const u32 myRank = topoInfo.userRank;
        u32 level1TotalSize = 0;
        bool sym0 = false;
        bool sym1 = false;
        HcclResult ret = HCCL_SUCCESS;
        ret = ValidateLevelAndCalcDim(physicalLevels[phys0], sym0, d0);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_INFO("[TopoMatchThreeLevel] ValidateLevelAndCalcDim level0 failed: hcclRet -> %d", ret),
            HcclResult::HCCL_E_NOT_SUPPORT);
        ret = ValidateLevelAndCalcDim(physicalLevels[phys1], sym1, level1TotalSize);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_INFO("[TopoMatchThreeLevel] ValidateLevelAndCalcDim level1 failed: hcclRet -> %d", ret),
            HcclResult::HCCL_E_NOT_SUPPORT);
        if (!sym0) {
            HCCL_INFO("[TopoMatchThreeLevel] Rank [%u], asymmetric level0 detected, not support.", myRank);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }

        if (!sym1) {
            const PhysicalLevelInfo& level1 = physicalLevels[phys1];
            // GCD 表示每个虚拟 POD 包含的 rank 数；虚拟 POD 必须由完整 Server 组成，且通信域可被整除
            u32 gcdRankSize = CalcGcd(level1.instSizeListByLayer);
            if (d0 == 0 || gcdRankSize < d0 || gcdRankSize % d0 != 0 || userRankSize % gcdRankSize != 0) {
                HCCL_INFO(
                    "[TopoMatchThreeLevel] Rank [%u], invalid Layer1 GCD split, gcdRankSize[%u], d0[%u], "
                    "userRankSize[%u].",
                    myRank, gcdRankSize, d0, userRankSize);
                return HcclResult::HCCL_E_NOT_SUPPORT;
            }
            level1TotalSize = gcdRankSize;
            HCCL_INFO(
                "[TopoMatchThreeLevel] Rank [%u], Layer1 asymmetric GCD split, gcdRankSize[%u], virtualPodNum[%u].",
                myRank, gcdRankSize, userRankSize / gcdRankSize);
        }

        if (d0 == 0 || level1TotalSize == 0 || level1TotalSize % d0 != 0) {
            HCCL_INFO(
                "[TopoMatchThreeLevel] Rank [%u], level1TotalSize[%u] not divisible by d0[%u].", myRank,
                level1TotalSize, d0);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        d1 = level1TotalSize / d0;
        if (userRankSize % d0 != 0 || (userRankSize / d0) % d1 != 0) {
            HCCL_INFO(
                "[TopoMatchThreeLevel] Rank [%u], userRankSize[%u] not divisible by d0[%u]*d1[%u].", myRank,
                userRankSize, d0, d1);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        d2 = userRankSize / d0 / d1;
        if (d2 == 1) {
            HCCL_INFO("[TopoMatchThreeLevel] Rank [%u], d0=%u, d1=%u, d2=1, not support three level.", myRank, d0, d1);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        return HcclResult::HCCL_SUCCESS;
    }
} // namespace

TopoMatchThreeLevel::TopoMatchThreeLevel() {}
TopoMatchThreeLevel::~TopoMatchThreeLevel() {}

HcclResult TopoMatchThreeLevel::MatchTopo(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    const auto& physicalLevels = topoInfo->physicalLevels;
    u32 myRank = topoInfo->userRank;
    u32 userRankSize = topoInfo->userRankSize;
    if (physicalLevels.empty() || userRankSize == 0 || algAttrs.algoTypes.size() != ALGO_LEVEL_NUM_THREE) {
        HCCL_WARNING(
            "[TopoMatchThreeLevel] Rank [%u], invalid input. "
            "physicalLevels.size[%zu], userRankSize[%u], algoTypes.size[%zu].",
            myRank, physicalLevels.size(), userRankSize, algAttrs.algoTypes.size());
        return HcclResult::HCCL_E_INTERNAL;
    }

    HcclResult ret = HCCL_SUCCESS;
    // 引擎过滤 + 锚点匹配 + 分段 + 最高层校验
    std::vector<u32> effIdx;
    std::vector<u32> pIndices;
    ret = ResolveMapping(physicalLevels, algAttrs, userRankSize, effIdx, pIndices);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_INFO("[TopoMatchThreeLevel] ResolveMapping failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
    u32 phys0 = effIdx[pIndices[0]];
    u32 phys1 = effIdx[pIndices[1]];

    // 对称性判定 + 维度计算（Layer1 非对称按 GCD 划分虚拟 POD）
    u32 d0 = 0;
    u32 d1 = 0;
    u32 d2 = 0;
    ret = CalcDimsAndCheckSymmetry(*topoInfo, phys0, phys1, d0, d1, d2);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_INFO("[TopoMatchThreeLevel] CalcDimsAndCheckSymmetry failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);

    // 构造 infos；level1 代表环须落在 myRank 所在的物理/虚拟 level1 instance 内
    std::vector<u32> group0 = physicalLevels[phys0].localRanks;
    u32 level1Base = (myRank / (d0 * d1)) * (d0 * d1);
    std::vector<u32> group1 = BuildRepresentativeGroup(d0, d1, level1Base + myRank % d0);
    std::vector<u32> group2 = BuildRepresentativeGroup(d0 * d1, d2, myRank % (d0 * d1));
    ret = ValidateGroup(group0, d0, myRank, "level0");
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_WARNING("[TopoMatchThreeLevel] ValidateGroup level0 failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
    ret = ValidateGroup(group1, d1, myRank, "level1");
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_WARNING("[TopoMatchThreeLevel] ValidateGroup level1 failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
    ret = ValidateGroup(group2, d2, myRank, "level2");
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_WARNING("[TopoMatchThreeLevel] ValidateGroup level2 failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
    algHierarchyInfo.infos.resize(ALGO_LEVEL_NUM_THREE);
    for (u32 i = 0; i < ALGO_LEVEL_NUM_THREE; i++) {
        algHierarchyInfo.infos[i].resize(1);
    }
    algHierarchyInfo.infos[0][0] = std::move(group0);
    algHierarchyInfo.infos[1][0] = std::move(group1);
    algHierarchyInfo.infos[ALGO_LEVEL_NUM_TWO][0] = std::move(group2);

    // 填充 physicalIdxForAlgoLevels（二级：MeshConcur 双层，普通单层）
    ret = FillPhysicalIdxForAlgoLevels(
        physicalLevels, effIdx, pIndices, algAttrs.algoTypes, algHierarchyInfo.physicalIdxForAlgoLevels);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_INFO("[TopoMatchThreeLevel] FillPhysicalIdxForAlgoLevels failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
    HCCL_INFO(
        "[TopoMatchThreeLevel] Rank [%u], d0[%u] d1[%u] d2[%u], physicalIdxForAlgoLevels: [%s].", myRank, d0, d1, d2,
        FormatPhysicalIdxForAlgoLevels(algHierarchyInfo.physicalIdxForAlgoLevels).c_str());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
