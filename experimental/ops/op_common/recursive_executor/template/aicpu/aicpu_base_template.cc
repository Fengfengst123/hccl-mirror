/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_base_template.h"
#include "data_transfer.h"

#include "log.h"

namespace ops_hccl {

void AicpuBaseTemplate::InitKernelRunParams(const DataParams& tempAlgParams)
{
    HCCL_DEBUG("[AicpuBaseTemplate][KernelRun] start, myRank[%u], rankSize[%zu].", myRank_, ranks_.size());
    // reduceOp 清洗下沉到派生模板：基类不再按命令类型清洗 reduceOp，
    // 各命令模板自行在覆写 InitKernelRunParams 时处理
    tempAlgParams_ = tempAlgParams;
    templateRankSize_ = static_cast<u32>(ranks_.size());
}

HcclResult AicpuBaseTemplate::KernelRun(
    const DataParams& tempAlgParams, TemplateResource& templateResource, std::vector<u32>& ranksForOutputData)
{
    InitKernelRunParams(tempAlgParams);

    const u64 sliceCount = tempAlgParams.sliceCount;
    const u64 tailCount = tempAlgParams.tailCount;
    if (sliceCount == 0 && tailCount == 0) {
        HCCL_DEBUG("[AicpuBaseTemplate][KernelRun] sliceCount is 0, no need to do, just success.");
        // 零数据早退时不清空 ranksForOutputData，计算归属后再返回，
        // 避免 SEQUENCE/PARALLEL 下游因空归属报错或产生含空组的父归属
        CHK_RET(CalcRanksForOutput(tempAlgParams.ranksForInputData, ranks_, myRank_, ranksForOutputData));
        ranksForOutputData_ = ranksForOutputData;
        return HCCL_SUCCESS;
    }

    if (templateResource.threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][KernelRun] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PrepareDataSplit(templateResource.channels));

    std::vector<ThreadHandle> subThreads;
    std::vector<u32> notifyIdxMainToSub;
    std::vector<u32> notifyIdxSubToMain;
    PrepareSubThreads(templateResource.threads, subThreads, notifyIdxMainToSub, notifyIdxSubToMain);

    if (templateRankSize_ <= 1) {
        return RunSingleRank(templateResource, subThreads, notifyIdxMainToSub, notifyIdxSubToMain, ranksForOutputData);
    }

    CHK_RET(RunMultiRank(templateResource, subThreads, notifyIdxMainToSub, notifyIdxSubToMain, ranksForOutputData));

    HCCL_DEBUG("[AicpuBaseTemplate][KernelRun] end.");
    return HCCL_SUCCESS;
}

void AicpuBaseTemplate::PrepareSubThreads(
    const std::vector<ThreadHandle>& threads, std::vector<ThreadHandle>& subThreads,
    std::vector<u32>& notifyIdxMainToSub, std::vector<u32>& notifyIdxSubToMain) const
{
    const u32 threadNum = static_cast<u32>(threads.size());
    const bool multiThread = (threadNum > 1);
    if (multiThread) {
        subThreads.assign(threads.begin() + 1, threads.end());
        notifyIdxMainToSub.assign(subThreads.size(), NOTIFY_IDX_PRE_SYNC);
        notifyIdxSubToMain.resize(subThreads.size());
        for (u32 i = 0; i < subThreads.size(); ++i) {
            notifyIdxSubToMain[i] = i;
        }
    }
}

HcclResult AicpuBaseTemplate::RunSingleRank(
    TemplateResource& templateResource, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub, const std::vector<u32>& notifyIdxSubToMain,
    std::vector<u32>& ranksForOutputData)
{
    // RunSingleRank 归属委托 CalcRanksForOutput，而非直接赋值 ranksForInputData
    CHK_RET(CalcRanksForOutput(tempAlgParams_.ranksForInputData, ranks_, myRank_, ranksForOutputData));
    ranksForOutputData_ = ranksForOutputData;

    if (syncAtCopyBoundary_) {
        CHK_RET(PreSyncSubThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    const bool canMergeCopy
        = (tempAlgParams_.inputBufferType == BufferType::INPUT && tempAlgParams_.outputBufferType == BufferType::OUTPUT
           && !tempAlgParams_.enableRemoteMemAccess);
    if (canMergeCopy) {
        CHK_RET(CopyInputToOutput(templateResource.threads));
    } else {
        if (tempAlgParams_.inputBufferType == BufferType::INPUT) {
            CHK_RET(PreCopy(templateResource.threads));
        }
        if (tempAlgParams_.outputBufferType != BufferType::HCCL_BUFFER) {
            CHK_RET(PostCopy(templateResource.threads));
        }
    }

    if (syncAtCopyBoundary_) {
        CHK_RET(PostSyncSubThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::RunMultiRank(
    TemplateResource& templateResource, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub, const std::vector<u32>& notifyIdxSubToMain,
    std::vector<u32>& ranksForOutputData)
{
    if (syncAtCopyBoundary_) {
        CHK_RET(PreSyncSubThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    if (tempAlgParams_.inputBufferType == BufferType::INPUT) {
        CHK_RET(PreCopy(templateResource.threads));
    }

    if (!syncAtCopyBoundary_) {
        CHK_RET(PreSyncSubThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    std::vector<DataSlicesList> txRxSlicesLists;
    CHK_RET(RunAlgorithm(txRxSlicesLists, ranksForOutputData));

    if (!txRxSlicesLists.empty()) {
        CHK_RET(SendAll(txRxSlicesLists, templateResource, templateResource.threads));
    }

    if (!syncAtCopyBoundary_) {
        CHK_RET(PostSyncSubThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    ranksForOutputData_ = ranksForOutputData;

    if (tempAlgParams_.outputBufferType != BufferType::HCCL_BUFFER) {
        CHK_RET(PostCopy(templateResource.threads));
    }

    if (syncAtCopyBoundary_) {
        CHK_RET(PostSyncSubThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::PrepareDataSplit(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_RET(CheckInputDataRanks(tempAlgParams_, "PrepareDataSplit"));
    const DataSizeInfo sizeInfo = CalcDataSizeInfo(tempAlgParams_);

    if (channels.empty() || channels.begin()->second.empty()) {
        channelSplit_.dataSplit.assign(1, sizeInfo.sliceSize);
        channelSplit_.dataOffset.assign(1, 0);
        if (tempAlgParams_.tailCount > 0) {
            channelSplit_.dataSplitTail.assign(1, sizeInfo.tailSize);
            channelSplit_.dataOffsetTail.assign(1, 0);
        } else {
            channelSplit_.dataSplitTail.clear();
            channelSplit_.dataOffsetTail.clear();
        }
        return HCCL_SUCCESS;
    }

    const std::vector<ChannelInfo>& peerChannels = channels.begin()->second;

    CHK_RET(CalcDataSplitByPortGroup(
        sizeInfo.sliceSize, sizeInfo.dataTypeSize, peerChannels, channelSplit_.dataSplit, channelSplit_.dataOffset));

    if (tempAlgParams_.tailCount > 0) {
        CHK_RET(CalcDataSplitByPortGroup(
            sizeInfo.tailSize, sizeInfo.dataTypeSize, peerChannels, channelSplit_.dataSplitTail,
            channelSplit_.dataOffsetTail));
    } else {
        channelSplit_.dataSplitTail.clear();
        channelSplit_.dataOffsetTail.clear();
    }

    HCCL_DEBUG(
        "[AicpuBaseTemplate][PrepareDataSplit] channels=%zu, dataSplit[0]=%llu, dataOffset[0]=%llu.",
        peerChannels.size(), static_cast<unsigned long long>(channelSplit_.dataSplit[0]),
        static_cast<unsigned long long>(channelSplit_.dataOffset[0]));
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::SendAll(
    const std::vector<DataSlicesList>& txRxSlicesLists, TemplateResource& templateResource,
    const std::vector<ThreadHandle>& threads)
{
    if (txRxSlicesLists.empty()) {
        return HCCL_SUCCESS;
    }

    const size_t lastIdx = txRxSlicesLists.size() - 1;
    const bool singleStep = (txRxSlicesLists.size() == 1);
    const bool parallelPostCopy = CanParallelPostCopy(templateResource);

    // 并行 PostCopy：单步场景在通信前先启动后台搬运
    if (parallelPostCopy && singleStep) {
        CHK_RET(LaunchPostCopy(threads));
    }

    for (size_t i = 0; i < txRxSlicesLists.size(); ++i) {
        const bool isLastStep = (i == lastIdx);
        TransferContext ctx = BuildTransferContext(txRxSlicesLists[i], templateResource, isLastStep, parallelPostCopy);
        CHK_RET(DataTransferSend(ctx));

        // 多步并行：倒数第二步通信完成后启动 PostCopy，与末步通信写入并行重叠
        if (parallelPostCopy && !singleStep && i == lastIdx - 1) {
            CHK_RET(LaunchPostCopy(threads));
        }
    }
    return HCCL_SUCCESS;
}

TransferContext AicpuBaseTemplate::BuildTransferContext(
    const DataSlicesList& txRxSlicesList, TemplateResource& templateResource, bool isLastStep,
    bool parallelPostCopy) const
{
    (void)isLastStep;
    (void)parallelPostCopy;
    TransferContext ctx;
    ctx.remoteReadEnabled = tempAlgParams_.enableRemoteMemAccess;
    ctx.buffType = tempAlgParams_.cclBufferType;
    ctx.txRxSlicesList = txRxSlicesList;
    ctx.templateRes = &templateResource;
    ctx.dataType = tempAlgParams_.dataType;
    ctx.reduceOp = tempAlgParams_.reduceOp;
    return ctx;
}

HcclResult AicpuBaseTemplate::PreSyncSubThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub)
{
    if (subThreads.empty()) {
        return HCCL_SUCCESS;
    }
    return ops_hccl::PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub);
}

HcclResult AicpuBaseTemplate::PostSyncSubThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxSubToMain)
{
    if (subThreads.empty()) {
        return HCCL_SUCCESS;
    }
    return ops_hccl::PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain);
}

HcclResult AicpuBaseTemplate::CopyInputToOutput(const std::vector<ThreadHandle>& threads)
{
    HCCL_DEBUG("[AicpuBaseTemplate][CopyInputToOutput] start, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][CopyInputToOutput] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(DirectCopyData(tempAlgParams_, threads[0], tempAlgParams_.ranksForInputData));
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::PreCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_DEBUG("[AicpuBaseTemplate][PreCopy] start, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][PreCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PreCopyData(tempAlgParams_, threads[0], tempAlgParams_.ranksForInputData));
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::PostCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_DEBUG("[AicpuBaseTemplate][PostCopy] start, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][PostCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PostCopyData(tempAlgParams_, threads[0], ranksForOutputData_));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
