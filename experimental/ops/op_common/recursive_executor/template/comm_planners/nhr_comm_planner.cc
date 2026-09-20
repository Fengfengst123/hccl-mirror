/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nhr_comm_planner.h"
#include "data_ops.h"
#include "template_utils.h"

#include <algorithm>

namespace ops_hccl {

namespace {

    inline void AddNhrRankDataSlices(
        const DataSizeInfo& sizeInfo, const DataParams& tempAlgParams, const std::vector<u32>& rankIds, u32 tailRankId,
        NhrAllGatherSlicePair& slicePair)
    {
        const u32 dataTypeSize = sizeInfo.dataTypeSize;
        for (u32 rankId : rankIds) {
            const u64 offset = tempAlgParams.sliceOffset + static_cast<u64>(rankId) * tempAlgParams.scratchStride;
            const u64 dataSize = CalcRankDataSize(sizeInfo, rankId, tailRankId);
            slicePair.srcSlices.emplace_back(slicePair.srcPtr, offset, dataSize, dataSize / dataTypeSize);
            slicePair.dstSlices.emplace_back(slicePair.dstPtr, offset, dataSize, dataSize / dataTypeSize);
        }
    }

    // 末步 rx 直写 output：src 远端 ccl，dst 按 output 布局
    inline HcclResult AddNhrRxSlicesToOutput(
        const DataSizeInfo& sizeInfo, const DataParams& tempAlgParams, const std::vector<u32>& rankIds, u32 tailRankId,
        const std::vector<u32>& ranksForOutputData, std::vector<DataSlice>& rxSrc, std::vector<DataSlice>& rxDst)
    {
        const u32 dataTypeSize = sizeInfo.dataTypeSize;
        for (u32 rankId : rankIds) {
            const u64 cclOff = tempAlgParams.sliceOffset + static_cast<u64>(rankId) * tempAlgParams.scratchStride;
            auto it = std::find(ranksForOutputData.begin(), ranksForOutputData.end(), rankId);
            CHK_PRT_RET(
                it == ranksForOutputData.end(),
                HCCL_ERROR("[AddNhrRxSlicesToOutput] rankId[%u] not found in ranksForOutputData.", rankId),
                HCCL_E_PARA);
            const u64 outIdx = static_cast<u64>(std::distance(ranksForOutputData.begin(), it));
            const u64 outOff = tempAlgParams.dataOffset + tempAlgParams.sliceOffset + outIdx * tempAlgParams.dataStride;
            const u64 dataSize = CalcRankDataSize(sizeInfo, rankId, tailRankId);
            rxSrc.emplace_back(nullptr, cclOff, dataSize, dataSize / dataTypeSize);
            rxDst.emplace_back(tempAlgParams.outputBufferPtr, outOff, dataSize, dataSize / dataTypeSize);
        }
        return HCCL_SUCCESS;
    }

    inline NhrStepParams CalcNhrStepParams(u32 step, u32 nSteps, u32 myAlgRank, u32 rankSize)
    {
        NhrStepParams params;
        params.delta = 1u << (nSteps - 1 - step);
        params.sendToAlgRank = (myAlgRank + params.delta) % rankSize;
        params.recvFromAlgRank = (myAlgRank + rankSize - params.delta) % rankSize;
        params.algRankStep = 1u << (nSteps - step);
        params.nSlices = (rankSize - 1 + params.delta) / params.algRankStep;
        return params;
    }

    // 构建 NHR AllGather 单步的 tx/rx DataSlice 列表
    inline HcclResult BuildNhrStepSlices(const NhrStepContext& ctx, NhrStepOutput& out)
    {
        const DataSizeInfo sizeInfo = CalcDataSizeInfo(ctx.tempAlgParams);
        u32 txAlgRank = ctx.myAlgRank;
        u32 rxAlgRank = ctx.params.recvFromAlgRank;
        std::vector<u32> txRankIds;
        std::vector<u32> rxRankIds;
        for (u32 i = 0; i < ctx.params.nSlices; ++i) {
            const long long txConnectedOffset
                = static_cast<long long>(ctx.ranks[txAlgRank]) - static_cast<long long>(ctx.myRank);
            const long long rxConnectedOffset
                = static_cast<long long>(ctx.ranks[rxAlgRank]) - static_cast<long long>(ctx.myRank);
            CHK_RET(GetConnectedInputRanks(
                ctx.ranksForInputData, txConnectedOffset, txRankIds, ctx.tempAlgParams.userRankSize,
                "[RunNhrAllGather]"));
            CHK_RET(GetConnectedInputRanks(
                ctx.ranksForInputData, rxConnectedOffset, rxRankIds, ctx.tempAlgParams.userRankSize,
                "[RunNhrAllGather]"));
            HCCL_DEBUG(
                "[RunNhrAllGather] slice=%u, txAlgRank=%u, txConnectedOffset=%lld, "
                "rxAlgRank=%u, rxConnectedOffset=%lld, txRankNum=%zu, rxRankNum=%zu",
                i, txAlgRank, txConnectedOffset, rxAlgRank, rxConnectedOffset, txRankIds.size(), rxRankIds.size());

            // 末步 rx 直写 output（DMA 消减），非末步 rx 写 ccl buffer
            const bool isLastStep = (ctx.step == ctx.nSteps - 1);
            const bool readLastStepToOutput = (ctx.lastStepRxRanks != nullptr) && isLastStep;
            NhrAllGatherSlicePair txSlicePair{ctx.tempAlgParams.cclBufferPtr, nullptr, out.txSrc, out.txDst};
            AddNhrRankDataSlices(sizeInfo, ctx.tempAlgParams, txRankIds, ctx.tailRankId, txSlicePair);
            if (readLastStepToOutput) {
                CHK_RET(AddNhrRxSlicesToOutput(
                    sizeInfo, ctx.tempAlgParams, rxRankIds, ctx.tailRankId, ctx.ranksForOutputData, out.rxSrc,
                    out.rxDst));
            } else {
                NhrAllGatherSlicePair rxSlicePair{nullptr, ctx.tempAlgParams.cclBufferPtr, out.rxSrc, out.rxDst};
                AddNhrRankDataSlices(sizeInfo, ctx.tempAlgParams, rxRankIds, ctx.tailRankId, rxSlicePair);
            }
            if (isLastStep && ctx.lastStepRxRanks != nullptr) {
                for (u32 rankId : rxRankIds) {
                    ctx.lastStepRxRanks->emplace_back(rankId);
                }
            }
            txAlgRank = (txAlgRank + ctx.rankSize - ctx.params.algRankStep) % ctx.rankSize;
            rxAlgRank = (rxAlgRank + ctx.rankSize - ctx.params.algRankStep) % ctx.rankSize;
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult RunNhrAllGather(
    const DataParams& tempAlgParams, const std::vector<u32>& ranks, u32 myRank, std::vector<u32>& ranksForOutputData,
    std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>* lastStepRxRanks)
{
    txRxSlicesLists.clear();
    if (lastStepRxRanks != nullptr) {
        lastStepRxRanks->clear();
    }
    CHK_RET(CheckInputDataRanks(tempAlgParams, "RunNhrAllGather"));

    CHK_RET(CalcRanksForOutput(tempAlgParams.ranksForInputData, ranks, myRank, ranksForOutputData));
    u32 rankSize = static_cast<u32>(ranks.size());
    if (rankSize <= 1) {
        HCCL_INFO("[RunNhrAllGather] no sendRecv needed, ranksForOutputDataNum=%zu", ranksForOutputData.size());
        return HCCL_SUCCESS;
    }
    // NHR 递归减半要求 rankSize 为 2 的幂
    CHK_PRT_RET(
        (rankSize & (rankSize - 1)) != 0, HCCL_ERROR("[RunNhrAllGather] rankSize[%u] is not a power of 2.", rankSize),
        HCCL_E_PARA);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));

    const DataSizeInfo sizeInfo = CalcDataSizeInfo(tempAlgParams);
    std::vector<u32> ranksForInputData = tempAlgParams.ranksForInputData;
    u32 nSteps = 0;
    // 计算ceil(log2(rankSize))
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_INFO(
        "[RunNhrAllGather] myAlgRank=%u, dataTypeSize=%u, sliceSize=%lu, tailSize=%lu, nSteps=%u", myAlgRank,
        sizeInfo.dataTypeSize, sizeInfo.sliceSize, sizeInfo.tailSize, nSteps);

    const u32 tailRankId = tempAlgParams.globalTailRankId;

    for (u32 step = 0; step < nSteps; ++step) {
        NhrStepParams params = CalcNhrStepParams(step, nSteps, myAlgRank, rankSize);
        HCCL_DEBUG(
            "[RunNhrAllGather] step=%u, delta=%u, sendToAlgRank=%u, sendToRank=%u, "
            "recvFromAlgRank=%u, recvFromRank=%u, algRankStep=%u, nSlices=%u",
            step, params.delta, params.sendToAlgRank, ranks[params.sendToAlgRank], params.recvFromAlgRank,
            ranks[params.recvFromAlgRank], params.algRankStep, params.nSlices);

        NhrStepContext ctx{tempAlgParams, ranks,  ranksForInputData, myRank, myAlgRank,          rankSize,
                           step,          nSteps, tailRankId,        params, ranksForOutputData, lastStepRxRanks};
        NhrStepOutput out;
        CHK_RET(BuildNhrStepSlices(ctx, out));

        txRxSlicesLists.emplace_back(
            SlicesList(std::move(out.txSrc), std::move(out.txDst)),
            SlicesList(std::move(out.rxSrc), std::move(out.rxDst)), ranks[params.recvFromAlgRank],
            ranks[params.sendToAlgRank]);
    }
    HCCL_INFO(
        "[RunNhrAllGather] end: txRxSlicesListNum=%zu, ranksForOutputDataNum=%zu", txRxSlicesLists.size(),
        ranksForOutputData.size());
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
