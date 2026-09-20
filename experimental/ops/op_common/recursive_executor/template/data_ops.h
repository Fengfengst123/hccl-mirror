/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DATA_OPS_H
#define DATA_OPS_H

#include <vector>
#include <algorithm>
#include <limits>
#include "alg_param.h"
#include "data_types.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

inline HcclResult CheckInputDataRanks(const DataParams& tempAlgParams, const char* tag)
{
    CHK_PRT_RET(
        tempAlgParams.ranksForInputData.empty(), HCCL_ERROR("[%s] ranksForInputData is empty.", tag), HCCL_E_PARA);
    CHK_PRT_RET(
        tempAlgParams.dataType >= HcclDataType::HCCL_DATA_TYPE_RESERVED,
        HCCL_ERROR("[%s] dataType[%d] is invalid.", tag, static_cast<int>(tempAlgParams.dataType)), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

// 计算 connectedOffset 后的 input rank 列表（mesh/nhr 公共逻辑）。
// connectedOffset = connectedRank - myRank，将 ranksForInputData 中每个 rank 平移得到对端 rank。
// userRankSize 为全局 rank 总数，用于校验映射结果不越界。
inline HcclResult GetConnectedInputRanks(
    const std::vector<u32>& ranksForInputData, long long connectedOffset, std::vector<u32>& connectedInputRanks,
    u32 userRankSize, const char* tag)
{
    connectedInputRanks.clear();
    connectedInputRanks.reserve(ranksForInputData.size());
    for (u32 rankId : ranksForInputData) {
        const long long connectedRankId = static_cast<long long>(rankId) + connectedOffset;
        CHK_PRT_RET(
            connectedRankId < 0 || connectedRankId >= static_cast<long long>(userRankSize),
            HCCL_ERROR(
                "[%s] connectedRankId[%lld] out of range [0, %u), rankId=%u, connectedOffset=%lld, "
                "ranksForInputData size=%zu.",
                tag, connectedRankId, userRankSize, rankId, connectedOffset, ranksForInputData.size()),
            HCCL_E_PARA);
        HCCL_DEBUG(
            "[%s] GetConnectedInputRanks: rankId=%u, connectedOffset=%lld, connectedRankId=%lld", tag, rankId,
            connectedOffset, connectedRankId);
        connectedInputRanks.emplace_back(static_cast<u32>(connectedRankId));
    }
    return HCCL_SUCCESS;
}

// channel 维度的数据切分信息：split/offset 成组出现，tail 向量为空表示无尾块。
struct ChannelSplitInfo {
    std::vector<u64> dataSplit{};
    std::vector<u64> dataOffset{};
    std::vector<u64> dataSplitTail{};
    std::vector<u64> dataOffsetTail{};

    bool ByChannel() const { return !dataSplit.empty(); }
    bool HasTail() const { return !dataSplitTail.empty(); }
    u64 Split(u32 channelIdx, bool isTailRank) const
    {
        return isTailRank ? dataSplitTail[channelIdx] : dataSplit[channelIdx];
    }
    u64 Offset(u32 channelIdx, bool isTailRank) const
    {
        return isTailRank ? dataOffsetTail[channelIdx] : dataOffset[channelIdx];
    }
};

HcclResult PreCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForInputData,
    u32 channelIdx = 0, const ChannelSplitInfo& channelSplit = {});

HcclResult CalcRanksForOutput(
    const std::vector<u32>& ranksForInputData, const std::vector<u32>& subCommRanks, u32 myRank,
    std::vector<u32>& ranksForOutputData);

// PostCopyData：将 ccl buffer 中的数据搬运到 output buffer。
// ranksForOutputData：需要搬运的 rank 列表，其循环下标 idx 用作 output 槽位索引（非 rankId），
//                     即数据写入位置 = dataOffset + sliceOffset + idx * dataStride。
// skipRanks：需跳过的 rank（已通过其他路径写入 output，如 directToOutput 中 SendAll 直写）。
HcclResult PostCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForOutputData,
    u32 channelIdx = 0, const ChannelSplitInfo& channelSplit = {}, const std::vector<u32>& skipRanks = {});

// 合并直拷：inputBufferPtr → outputBufferPtr，跳过 cclBuffer 中转
HcclResult DirectCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForData,
    u32 channelIdx = 0, const ChannelSplitInfo& channelSplit = {});

struct DataSizeInfo {
    u32 dataTypeSize{0};
    u64 sliceSize{0};
    u64 tailSize{0};
};

DataSizeInfo CalcDataSizeInfo(const DataParams& tempAlgParams);

inline u64 CalcRankDataSize(const DataSizeInfo& sizeInfo, u32 rank, u32 globalTailRankId)
{
    return (sizeInfo.tailSize > 0 && rank == globalTailRankId) ? sizeInfo.tailSize : sizeInfo.sliceSize;
}

HcclResult CalcDataSplitByPortGroup(
    u64 totalDataSize, u32 dataTypeSize, const std::vector<ChannelInfo>& channels, std::vector<u64>& dataSplit,
    std::vector<u64>& dataOffset);

} // namespace ops_hccl

#endif // DATA_OPS_H
