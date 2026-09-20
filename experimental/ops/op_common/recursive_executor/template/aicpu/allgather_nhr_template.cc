/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_nhr_template.h"
#include "channel.h"
#include "comm_planners/nhr_comm_planner.h"
#include "data_transfer.h"
#include "data_ops.h"
#include "template_factory.h"

#include "log.h"
#include <algorithm>

namespace ops_hccl {

REGISTER_RE_TEMPLATE(HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_NHR, AllGatherNhrTemplate)

HcclResult AllGatherNhrTemplate::DoCalcChannelRequest(
    HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& levelChannels)
{
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subcommInfo, levelChannels));
    if (dataSize_ <= SINGLE_CHANNEL_MAX_DATA_SIZE_) {
        DedupChannelsByRemoteRank(levelChannels);
    }
    return HCCL_SUCCESS;
}

u32 AllGatherNhrTemplate::DoCalcThreadNum() const { return channelsPerRank_ * 2; }

u32 AllGatherNhrTemplate::DoCalcNotifyPerThread() const { return 2; }

void AllGatherNhrTemplate::DedupChannelsByRemoteRank(std::vector<HcclChannelDesc>& channels)
{
    std::vector<u32> seenRanks;
    std::vector<HcclChannelDesc> deduped;
    for (const auto& ch : channels) {
        bool dup = false;
        for (u32 r : seenRanks) {
            if (r == ch.remoteRank) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            seenRanks.push_back(ch.remoteRank);
            deduped.push_back(ch);
        }
    }
    channels = std::move(deduped);
}

HcclResult
AllGatherNhrTemplate::RunAlgorithm(std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>& ranksForOutputData)
{
    HCCL_DEBUG("[AllGatherNhrTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].", myRank_, templateRankSize_);

    canParallelPostCopy_ = CanReadLastStepToOutput(tempAlgParams_);

    CHK_RET(RunNhrAllGather(
        tempAlgParams_, ranks_, myRank_, ranksForOutputData, txRxSlicesLists,
        canParallelPostCopy_ ? &lastStepRxRanks_ : nullptr));

    ranksForOutputData_ = ranksForOutputData;

    HCCL_DEBUG(
        "[AllGatherNhrTemplate][RunAlgorithm] end, canParallelPostCopy=%d, lastStepRxRankNum=%zu.",
        static_cast<int>(canParallelPostCopy_), lastStepRxRanks_.size());
    return HCCL_SUCCESS;
}

HcclResult AllGatherNhrTemplate::PreCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_DEBUG("[AllGatherNhrTemplate][PreCopy] start, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AllGatherNhrTemplate][PreCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    const u32 channelNum = static_cast<u32>(channelSplit_.dataSplit.size());
    for (u32 ch = 0; ch < channelNum; ++ch) {
        const ThreadHandle& thread = threads[std::min(ch, static_cast<u32>(threads.size() - 1))];
        CHK_RET(PreCopyData(tempAlgParams_, thread, tempAlgParams_.ranksForInputData, ch, channelSplit_));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNhrTemplate::CopyInputToOutput(const std::vector<ThreadHandle>& threads)
{
    HCCL_DEBUG("[AllGatherNhrTemplate][CopyInputToOutput] start, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AllGatherNhrTemplate][CopyInputToOutput] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    const u32 channelNum = static_cast<u32>(channelSplit_.dataSplit.size());
    for (u32 ch = 0; ch < channelNum; ++ch) {
        const ThreadHandle& thread = threads[std::min(ch, static_cast<u32>(threads.size() - 1))];
        CHK_RET(DirectCopyData(tempAlgParams_, thread, tempAlgParams_.ranksForInputData, ch, channelSplit_));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNhrTemplate::LaunchPostCopy(const std::vector<ThreadHandle>& threads)
{
    const u32 postCopyChannelNum = std::min(static_cast<u32>(channelSplit_.dataSplit.size()), channelsPerRank_);
    for (u32 ch = 0; ch < postCopyChannelNum; ++ch) {
        const u32 postCopyThreadIdx = channelsPerRank_ + ch;
        // 每通道 1:1 sync：threads[ch]（通信线程）notify → threads[postCopyThreadIdx]（PostCopy 线程）wait。
        // notify 排在 threads[ch] 队列中，在该线程之前提交的所有 WRITE 之后才触发，FIFO 保证 happens-before。
        CHK_RET(PreSyncInterThreads(threads[ch], {threads[postCopyThreadIdx]}, {NOTIFY_IDX_POST_COPY}));
        CHK_RET(PostCopyData(
            tempAlgParams_, threads[postCopyThreadIdx], ranksForOutputData_, ch, channelSplit_, lastStepRxRanks_));
    }
    postCopyLaunched_ = true;
    return HCCL_SUCCESS;
}

TransferContext AllGatherNhrTemplate::BuildTransferContext(
    const DataSlicesList& txRxSlicesList, TemplateResource& templateResource, bool isLastStep,
    bool parallelPostCopy) const
{
    TransferContext ctx;
    // 并行 PostCopy 时末步直写 output（省去后续搬运）；否则写 cclBuffer 由 PostCopy 统一搬出
    if (isLastStep && parallelPostCopy) {
        ctx.remoteReadEnabled = true;
        ctx.buffType = BufferType::OUTPUT;
    } else {
        ctx.remoteReadEnabled = tempAlgParams_.enableRemoteMemAccess;
        ctx.buffType = tempAlgParams_.cclBufferType;
    }
    ctx.txRxSlicesList = txRxSlicesList;
    ctx.templateRes = &templateResource;
    ctx.dataType = tempAlgParams_.dataType;
    ctx.reduceOp = tempAlgParams_.reduceOp;
    // NHR 场景各对端共用线程池（threads[channelIdx]），而非按 rankPos*channelsPerRank+channelIdx 分配
    ctx.reuseChannelThreads = true;
    return ctx;
}

bool AllGatherNhrTemplate::CanParallelPostCopy(const TemplateResource& templateResource) const
{
    return canParallelPostCopy_ && !IsPcieProtocol(templateResource.channels);
}

HcclResult AllGatherNhrTemplate::PostCopy(const std::vector<ThreadHandle>& threads)
{
    if (postCopyLaunched_) {
        HCCL_DEBUG("[AllGatherNhrTemplate][PostCopy] skipped, lastStepRxRanks already in output.");
        return HCCL_SUCCESS;
    }

    if (threads.empty()) {
        HCCL_ERROR("[AllGatherNhrTemplate][PostCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    const u32 channelNum = static_cast<u32>(channelSplit_.dataSplit.size());
    for (u32 ch = 0; ch < channelNum; ++ch) {
        const ThreadHandle& thread = threads[std::min(ch, static_cast<u32>(threads.size() - 1))];
        CHK_RET(PostCopyData(tempAlgParams_, thread, ranksForOutputData_, ch, channelSplit_));
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
