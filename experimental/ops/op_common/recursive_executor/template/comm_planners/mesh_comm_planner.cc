/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mesh_comm_planner.h"
#include "data_ops.h"
#include "template_utils.h"

#include <algorithm>

namespace ops_hccl {

namespace {

    // 按 ccl buffer 布局（sliceOffset + rankId * stride）追加 src/dst DataSlice
    inline void
    AddRankDataSlices(const MeshSliceContext& sliceInfo, const std::vector<u32>& rankIds, MeshSlicePair& slicePair)
    {
        const u32 dataTypeSize = DATATYPE_SIZE_TABLE[sliceInfo.tempAlgParams.dataType];
        for (u32 rankId : rankIds) {
            const u64 dataSize
                = (sliceInfo.tailSize > 0 && rankId == sliceInfo.tailRankId) ? sliceInfo.tailSize : sliceInfo.sliceSize;
            const u64 dataOffset = sliceInfo.tempAlgParams.sliceOffset + static_cast<u64>(rankId) * sliceInfo.stride;
            HCCL_DEBUG(
                "[RunMeshAllGather] AddRankDataSlices: rankId=%u, sliceOffset=%lu, stride=%lu, "
                "dataOffset=%lu, dataSize=%lu, count=%lu",
                rankId, sliceInfo.tempAlgParams.sliceOffset, sliceInfo.stride, dataOffset, dataSize,
                dataSize / dataTypeSize);
            slicePair.firstSlices.emplace_back(slicePair.firstBufferPtr, dataOffset, dataSize, dataSize / dataTypeSize);
            slicePair.secondSlices.emplace_back(
                slicePair.secondBufferPtr, dataOffset, dataSize, dataSize / dataTypeSize);
        }
    }

    // rx 直写 output：src 按 ccl 布局，dst 按 output 布局
    inline HcclResult AddRankDataSlicesToOutput(
        const MeshSliceContext& sliceInfo, const std::vector<u32>& rankIds, const std::vector<u32>& ranksForOutputData,
        MeshSlicePair& slicePair)
    {
        const u32 dataTypeSize = DATATYPE_SIZE_TABLE[sliceInfo.tempAlgParams.dataType];
        for (u32 rankId : rankIds) {
            const u64 dataSize
                = (sliceInfo.tailSize > 0 && rankId == sliceInfo.tailRankId) ? sliceInfo.tailSize : sliceInfo.sliceSize;
            const u64 srcOffset = sliceInfo.tempAlgParams.sliceOffset + static_cast<u64>(rankId) * sliceInfo.stride;
            auto it = std::find(ranksForOutputData.begin(), ranksForOutputData.end(), rankId);
            CHK_PRT_RET(
                it == ranksForOutputData.end(),
                HCCL_ERROR("[RunMeshAllGather] rankId[%u] not found in ranksForOutputData.", rankId), HCCL_E_PARA);
            const u64 outIdx = static_cast<u64>(std::distance(ranksForOutputData.begin(), it));
            const u64 dstOffset = sliceInfo.tempAlgParams.dataOffset + sliceInfo.tempAlgParams.sliceOffset
                                  + outIdx * sliceInfo.tempAlgParams.dataStride;
            HCCL_DEBUG(
                "[RunMeshAllGather] AddRankDataSlicesToOutput: rankId=%u, srcOffset=%lu, dstOffset=%lu, "
                "dataSize=%lu, count=%lu",
                rankId, srcOffset, dstOffset, dataSize, dataSize / dataTypeSize);
            slicePair.firstSlices.emplace_back(slicePair.firstBufferPtr, srcOffset, dataSize, dataSize / dataTypeSize);
            slicePair.secondSlices.emplace_back(
                slicePair.secondBufferPtr, dstOffset, dataSize, dataSize / dataTypeSize);
        }
        return HCCL_SUCCESS;
    }

    // 准备 Mesh AllGather 常量与参数，填充到 sliceInfo
    inline HcclResult BuildMeshSliceInfo(MeshSliceContext& sliceInfo, const std::vector<u32>& ranks, u32 myRank)
    {
        sliceInfo.rankSize = static_cast<u32>(ranks.size());
        sliceInfo.stride = sliceInfo.tempAlgParams.scratchStride;
        u32 myAlgRank = 0;
        CHK_RET(GetAlgRank(myRank, ranks, myAlgRank));
        sliceInfo.ranksForInputData = sliceInfo.tempAlgParams.ranksForInputData;
        const DataSizeInfo sizeInfo = CalcDataSizeInfo(sliceInfo.tempAlgParams);
        sliceInfo.sliceSize = sizeInfo.sliceSize;
        sliceInfo.tailSize = sizeInfo.tailSize;
        sliceInfo.tailRankId = sliceInfo.tempAlgParams.globalTailRankId;
        sliceInfo.directToOutput = (sliceInfo.tempAlgParams.outputBufferType == BufferType::OUTPUT);
        HCCL_INFO(
            "[RunMeshAllGather] myAlgRank=%u, rankSize=%u, dataTypeSize=%u, sliceSize=%lu, directToOutput=%d",
            myAlgRank, sliceInfo.rankSize, sizeInfo.dataTypeSize, sliceInfo.sliceSize,
            static_cast<int>(sliceInfo.directToOutput));
        return HCCL_SUCCESS;
    }

    // 为单个对端 rank 构建 tx/rx DataSlicesList
    inline HcclResult BuildConnectedPair(
        const MeshSliceContext& sliceInfo, u32 myRank, u32 connectedRank, u32 connectedIdx,
        const std::vector<u32>& ranksForOutputData, std::vector<DataSlicesList>& txRxSlicesLists)
    {
        const long long connectedOffset = static_cast<long long>(connectedRank) - static_cast<long long>(myRank);
        std::vector<u32> connectedInputRanks;
        CHK_RET(GetConnectedInputRanks(
            sliceInfo.ranksForInputData, connectedOffset, connectedInputRanks, sliceInfo.tempAlgParams.userRankSize,
            "[RunMeshAllGather]"));
        HCCL_DEBUG(
            "[RunMeshAllGather] connectedRank=%u, connectedAlgRank=%u, connectedOffset=%lld, "
            "connectedInputRankNum=%zu",
            connectedRank, connectedIdx, connectedOffset, connectedInputRanks.size());
        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        // tx：本端 ccl[myRank slots] -> 对端 ccl
        MeshSlicePair txSlicePair{sliceInfo.tempAlgParams.cclBufferPtr, nullptr, txSrcSlicesAll, txDstSlicesAll};
        AddRankDataSlices(sliceInfo, sliceInfo.ranksForInputData, txSlicePair);
        // rx：directToOutput 时落 output，否则落 ccl buffer
        if (sliceInfo.directToOutput) {
            MeshSlicePair rxSlicePair{nullptr, sliceInfo.tempAlgParams.outputBufferPtr, rxSrcSlicesAll, rxDstSlicesAll};
            CHK_RET(AddRankDataSlicesToOutput(sliceInfo, connectedInputRanks, ranksForOutputData, rxSlicePair));
        } else {
            MeshSlicePair rxSlicePair{nullptr, sliceInfo.tempAlgParams.cclBufferPtr, rxSrcSlicesAll, rxDstSlicesAll};
            AddRankDataSlices(sliceInfo, connectedInputRanks, rxSlicePair);
        }
        txRxSlicesLists.emplace_back(
            SlicesList(std::move(txSrcSlicesAll), std::move(txDstSlicesAll)),
            SlicesList(std::move(rxSrcSlicesAll), std::move(rxDstSlicesAll)), connectedRank, connectedRank);
        HCCL_DEBUG(
            "[RunMeshAllGather] Build DataSlicesList: dataType=%d, sliceSize=%lu, stride=%lu, "
            "connectedRank=%u, txRankNum=%zu, rxRankNum=%zu, txRxSlicesListNum=%zu, directToOutput=%d",
            static_cast<int>(sliceInfo.tempAlgParams.dataType), sliceInfo.sliceSize, sliceInfo.stride, connectedRank,
            sliceInfo.ranksForInputData.size(), connectedInputRanks.size(), txRxSlicesLists.size(),
            static_cast<int>(sliceInfo.directToOutput));
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult RunMeshAllGather(
    const DataParams& tempAlgParams, const std::vector<u32>& ranks, u32 myRank, std::vector<u32>& ranksForOutputData,
    std::vector<DataSlicesList>& txRxSlicesLists)
{
    txRxSlicesLists.clear();

    CHK_RET(CheckInputDataRanks(tempAlgParams, "RunMeshAllGather"));
    CHK_RET(CalcRanksForOutput(tempAlgParams.ranksForInputData, ranks, myRank, ranksForOutputData));
    if (ranks.size() <= 1) {
        HCCL_INFO("[RunMeshAllGather] no sendRecv needed, ranksForOutputDataNum=%zu", ranksForOutputData.size());
        return HCCL_SUCCESS;
    }

    MeshSliceContext sliceInfo{tempAlgParams};
    CHK_RET(BuildMeshSliceInfo(sliceInfo, ranks, myRank));

    for (u32 connectedIdx = 0; connectedIdx < sliceInfo.rankSize; ++connectedIdx) {
        const u32 connectedRank = ranks[connectedIdx];
        if (connectedRank == myRank) {
            continue;
        }
        CHK_RET(
            BuildConnectedPair(sliceInfo, myRank, connectedRank, connectedIdx, ranksForOutputData, txRxSlicesLists));
    }
    HCCL_INFO("[RunMeshAllGather] end: txRxSlicesListNum=%zu", txRxSlicesLists.size());
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
