/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_executor.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

// 深拷贝 AlgoExecDesc 子树：将 children 中的 shared_ptr 替换为新建节点，断开与注册表的共享
static void DeepCloneChildren(AlgoExecDesc& desc)
{
    for (auto& child : desc.children) {
        auto* subPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&child);
        if (subPtr != nullptr && *subPtr != nullptr) {
            auto clone = std::make_shared<AlgoExecDesc>(**subPtr);
            DeepCloneChildren(*clone);
            *subPtr = clone;
        }
    }
}

OpsExecutor::OpsExecutor(HcclAlgorithm& algo, const OpParam& param) : algo_(algo), rankSize_(0), root_(param.root)
{
    DeepCloneChildren(algo_.algoExecDesc);
    opMode_ = param.opMode;
    dataInfo_.inputPtr = param.inputPtr;
    dataInfo_.inputSize = param.inputSize;
    dataInfo_.outputPtr = param.outputPtr;
    dataInfo_.outputSize = param.outputSize;
    dataInfo_.reduceOp = param.reduceType;
    dataInfo_.dataType = param.DataDes.dataType;
    if (param.DataDes.dataType >= HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[OpsExecutor] dataType=%d out of range [0, %d)", static_cast<int>(param.DataDes.dataType),
            static_cast<int>(HCCL_DATA_TYPE_RESERVED));
        return;
    }
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
}

OpsExecutor::~OpsExecutor() {}

HcclResult OpsExecutor::InitAlgHierarchyInfo(
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    algHierarchyInfo_ = algHierarchyInfo;
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::Orchestrate(const AlgResourceCtxSerializable& resCtx)
{
    CHK_RET(InitRes(resCtx));
    u64 dataCount = 0;
    u64 maxProcCntPerLoop = 0;
    u64 loopTimes = 0;
    u64 dataStride = 0;
    u64 lastProcessCount = 0;
    u64 lastTailCount = 0;
    CHK_RET(PrepareOrchestrate(dataCount, maxProcCntPerLoop, loopTimes, dataStride, lastProcessCount, lastTailCount));
    u64 offsetCount = 0;
    std::vector<AlgoExecDataDesc> reusableChildren;
    for (u64 loopIdx = 0; loopIdx < loopTimes; ++loopIdx) {
        u64 processCount = loopIdx == loopTimes - 1 ? lastProcessCount : maxProcCntPerLoop;
        u64 tailCount = loopIdx == loopTimes - 1 ? lastTailCount : 0;
        AlgoExecDataDesc algoExecDataDesc;
        HCCL_DEBUG(
            "[Orchestrate] myRank_ =%u, loopTimes=%llu, loopIdx=%llu, processCount=%llu, offsetCount=%llu, "
            "tailCount=%llu",
            myRank_, loopTimes, loopIdx, processCount, offsetCount, tailCount);
        const bool isReduceOrAllReduce
            = algo_.hcclCmdType == HcclCMDType::HCCL_CMD_ALLREDUCE || algo_.hcclCmdType == HcclCMDType::HCCL_CMD_REDUCE;
        u64 dataOffset = (algo_.hcclCmdType == HcclCMDType::HCCL_CMD_ALLGATHER || isReduceOrAllReduce) ?
                             offsetCount * dataTypeSize_ :
                             (offsetCount / rankSize_) * dataTypeSize_;
        u64 loopDataStride = isReduceOrAllReduce ? (processCount - tailCount) / rankSize_ * dataTypeSize_ : dataStride;
        InitAlgoExecDataDesc(algoExecDataDesc, dataOffset, processCount - tailCount, tailCount, loopDataStride);
        if (algo_.algoExecDesc.execPolicy == HcclAlgExecPolicy::OMNIPIPE) {
            if (algo_.hcclCmdType != HcclCMDType::HCCL_CMD_ALLREDUCE
                && algo_.hcclCmdType != HcclCMDType::HCCL_CMD_ALLGATHER) {
                HCCL_ERROR(
                    "[OpsExecutor] OMNIPIPE only supports AllReduce/AllGather, but got cmd=%d",
                    static_cast<int>(algo_.hcclCmdType));
                return HCCL_E_INTERNAL;
            }
            DataParams templateDataParams;
            CHK_RET(GenTemplateDataParams(algoExecDataDesc, templateDataParams));
            std::vector<u32> ranksForInputData = {myRank_};
            CHK_RET(PreCopyData(templateDataParams, mainThread_, ranksForInputData));
            algoExecDataDesc.inputBufferType = BufferType::HCCL_BUFFER;
            algoExecDataDesc.outputBufferType = BufferType::HCCL_BUFFER;
            CHK_RET(OrchestrateOmniPipeLoop(algo_.algoExecDesc, algoExecDataDesc));
            std::vector<u32> ranksForOutputData(rankSize_);
            std::iota(ranksForOutputData.begin(), ranksForOutputData.end(), 0);
            CHK_RET(PostCopyData(templateDataParams, mainThread_, ranksForOutputData));
        } else {
            CHK_RET(OrchestrateLoop(algo_.algoExecDesc, algoExecDataDesc, &reusableChildren));
        }
        offsetCount += processCount;
    }
    return HCCL_SUCCESS;
}

u64 OpsExecutor::GetMaxProcCntPerLoop(u64 dataCount)
{
    if (scratchMultiple_ == 0 || dataTypeSize_ == 0) {
        return std::max(dataCount, 1ULL);
    }
    u64 maxByCcl = cclBufferInfo_.size / (static_cast<u64>(scratchMultiple_) * dataTypeSize_);
    if (rankSize_ > 0 && dataCount % rankSize_ != 0
        && (algo_.hcclCmdType == HCCL_CMD_BROADCAST || algo_.hcclCmdType == HCCL_CMD_ALLREDUCE
            || algo_.hcclCmdType == HCCL_CMD_REDUCE)) {
        const u64 maxTailPadding = static_cast<u64>(rankSize_ - 1) * (rankSize_ - 1);
        maxByCcl = (maxByCcl > maxTailPadding) ? (maxByCcl - maxTailPadding) : 1;
    }
    u64 maxByUb = UB_MAX_DATA_SIZE / dataTypeSize_;
    u64 resCount = std::min({dataCount, maxByCcl, maxByUb});
    if (algo_.hcclCmdType != HCCL_CMD_ALLGATHER && algo_.hcclCmdType != HCCL_CMD_ALLREDUCE
        && algo_.hcclCmdType != HCCL_CMD_REDUCE) {
        u64 dataStrideCount = (rankSize_ > 0) ? (dataInfo_.inputSize / rankSize_ / dataTypeSize_) : dataCount;
        resCount = std::min(resCount, dataStrideCount);
        resCount = (resCount / rankSize_) * rankSize_;
    } else if (algo_.hcclCmdType == HCCL_CMD_ALLREDUCE || algo_.hcclCmdType == HCCL_CMD_REDUCE) {
        resCount = (resCount / rankSize_) * rankSize_;
    }
    // 取整后为 0 表示 dataCount < rankSize_，用 min(dataCount, rankSize_) 让 tail 机制单 loop 完整承载
    if (resCount == 0) {
        resCount = std::min(dataCount, static_cast<u64>(rankSize_));
    }
    return std::max(resCount, 1ULL);
}

HcclResult OpsExecutor::PrepareOrchestrate(
    u64& dataCount, u64& maxProcCntPerLoop, u64& loopTimes, u64& dataStride, u64& lastProcessCount, u64& lastTailCount)
{
    if (rankSize_ == 0) {
        HCCL_ERROR("[OpsExecutor] rankSize_ is 0 after InitRes, topoInfo may be invalid");
        return HCCL_E_INTERNAL;
    }
    if (dataTypeSize_ == 0) {
        HCCL_ERROR(
            "[OpsExecutor] dataTypeSize_ is 0, dataType=%d may be invalid", static_cast<int>(dataInfo_.dataType));
        return HCCL_E_PARA;
    }
    dataCount = dataInfo_.inputSize / dataTypeSize_;
    if (dataCount == 0) {
        HCCL_ERROR("[OpsExecutor] dataCount is zero");
        return HCCL_E_PARA;
    }
    dataStride = (dataCount / rankSize_) * dataTypeSize_;
    if (algo_.hcclCmdType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        dataStride = dataInfo_.inputSize;
    }
    maxProcCntPerLoop = GetMaxProcCntPerLoop(dataCount);
    loopTimes = (dataCount + maxProcCntPerLoop - 1) / maxProcCntPerLoop;
    u64 remainder = dataCount % maxProcCntPerLoop;
    lastProcessCount = (remainder == 0) ? maxProcCntPerLoop : remainder;
    if (algo_.hcclCmdType == HcclCMDType::HCCL_CMD_BROADCAST || algo_.hcclCmdType == HcclCMDType::HCCL_CMD_REDUCE
        || algo_.hcclCmdType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        lastTailCount = lastProcessCount % rankSize_;
    } else {
        lastTailCount = 0;
    }
    subCommRoots_.clear();
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::InitRes(const AlgResourceCtxSerializable& resCtx)
{
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    cclBufferInfo_.ptr = resCtx.cclMem.addr;
    cclBufferInfo_.size = resCtx.cclMem.size;
    cclBufferInfo_.bufferType = BufferType::HCCL_BUFFER;
    threads_ = resCtx.threads;
    mainThread_ = threads_.at(0);
    auto topoLevelNum = algHierarchyInfo_.infos.size();
    subThreads_.assign(topoLevelNum, {});
    auto subThreadBegin = threads_.begin();
    auto subThreadEnd = threads_.begin();
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    if (algo_.algoExecDesc.execPolicy == HcclAlgExecPolicy::OMNIPIPE) {
        u32 eqRankSize = 0;
        double eqBw;
        CHK_RET(OmniPipeUpdateEqBWAndReorder(algo_.algoExecDesc, eqRankSize, eqBw));
    }
    channelTable_ = RestoreChannelMap(resCtx);
    AlgResourceRequest resourceRequest;
    CHK_RET(GetRes(resourceRequest));
    if (topoLevelNum == 1) {
        subThreadEnd = subThreadBegin + 1 + maxSlaveThreadNum_.at(0);
        subThreads_.at(0).assign(subThreadBegin, subThreadEnd);
    } else {
        for (size_t subCommIndex = 0; subCommIndex < topoLevelNum; subCommIndex++) {
            subThreadBegin = (subCommIndex == 0 ? subThreadBegin + 1 : subThreadEnd);
            subThreadEnd = subThreadBegin + 1 + maxSlaveThreadNum_.at(subCommIndex);
            subThreads_.at(subCommIndex).assign(subThreadBegin, subThreadEnd);
        }
    }
    cachedTemplateRanks_.clear();
    cachedTemplateResources_.clear();
    cachedTemplateRanks_.reserve(topoLevelNum);
    cachedTemplateResources_.reserve(topoLevelNum);
    for (size_t level = 0; level < topoLevelNum; ++level) {
        cachedTemplateRanks_.emplace_back(algHierarchyInfo_.infos[level].at(0));
        TemplateResource res;
        res.channels = channelTable_.at(level);
        res.threads = subThreads_.at(level);
        cachedTemplateResources_.emplace_back(std::move(res));
    }
    return HCCL_SUCCESS;
}

std::vector<std::map<u32, std::vector<ChannelInfo>>>
OpsExecutor::RestoreChannelMap(const AlgResourceCtxSerializable& resCtx)
{
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo = resCtx.algHierarchyInfo;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> rankIdToChannelInfo(algHierarchyInfo.infos.size());
    for (u32 level = 0; level < algHierarchyInfo.infos.size(); level++) {
        for (auto& channel : resCtx.channels[level]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[level][remoteRank].push_back(channel);
        }
    }
    HCCL_INFO("[RestoreChannelMap] myRank=%u, levelNum=%zu", myRank_, rankIdToChannelInfo.size());
    for (size_t lv = 0; lv < rankIdToChannelInfo.size(); ++lv) {
        u32 totalCh = 0;
        u32 maxChPerRank = 0;
        for (auto& pair : rankIdToChannelInfo[lv]) {
            totalCh += pair.second.size();
            if (pair.second.size() > maxChPerRank) {
                maxChPerRank = pair.second.size();
            }
        }
        HCCL_DEBUG(
            "[RestoreChannelMap] myRank=%u, level=%zu, remoteRankNum=%zu, totalChannel=%u, maxChannelPerRank=%u",
            myRank_, lv, rankIdToChannelInfo[lv].size(), totalCh, maxChPerRank);
    }
    return rankIdToChannelInfo;
}

HcclResult OpsExecutor::PreSyncBySubCommMask(const AlgoExecDesc& execDesc)
{
    if (execDesc.subCommMask == 0) {
        return HCCL_SUCCESS;
    }
    if (algHierarchyInfo_.infos.size() == 1) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> syncInterThreads;
    std::vector<u32> syncNotifyOnAlgoExec;
    auto topoLevelNum = algHierarchyInfo_.infos.size();
    for (int i = 0; i < topoLevelNum; i++) {
        if (execDesc.subCommMask & (1u << i)) {
            syncInterThreads.emplace_back(subThreads_.at(i).at(0));
            if (notifyNumOnSubMainThread_.at(i) == 0) {
                HCCL_ERROR("[OpsExecutor] notifyNumOnSubMainThread_.at(i) is zero");
                return HCCL_E_INTERNAL;
            }
            syncNotifyOnAlgoExec.emplace_back(notifyNumOnSubMainThread_.at(i) - 1);
        }
    }
    return PreSyncInterThreads(mainThread_, syncInterThreads, syncNotifyOnAlgoExec);
}

HcclResult OpsExecutor::PostSyncBySubCommMask(const AlgoExecDesc& execDesc)
{
    if (execDesc.subCommMask == 0) {
        return HCCL_SUCCESS;
    }
    if (algHierarchyInfo_.infos.size() == 1) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> syncInterThreads;
    std::vector<u32> syncNotifyOnMain;
    auto topoLevelNum = algHierarchyInfo_.infos.size();
    for (int i = 0; i < topoLevelNum; i++) {
        if (execDesc.subCommMask & (1u << i)) {
            syncInterThreads.emplace_back(subThreads_.at(i).at(0));
            syncNotifyOnMain.emplace_back(i);
        }
    }
    return PostSyncInterThreads(mainThread_, syncInterThreads, syncNotifyOnMain);
}

HcclResult OpsExecutor::CalcChannelResRecursion(
    HcclComm comm, AlgoExecDesc& algoExecDesc, std::vector<std::vector<HcclChannelDesc>>& requestChannels)
{
    size_t childrenSize = algoExecDesc.children.size();
    for (size_t i = 0; i < childrenSize; ++i) {
        VariantType& v = algoExecDesc.children[i];
        if (const TemplateExecDesc* templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            CHK_RET(CalcTemplateChannelRes(comm, *templateExeDes, requestChannels));
        } else if (auto* hcclAlgorithmPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            CHK_RET(CalcChannelResRecursion(comm, **hcclAlgorithmPtr, requestChannels));
        } else {
            HCCL_ERROR("[CalcChannelResRecursion] unexpected variant type at children index %zu", i);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::GetResRecursion(AlgoExecDesc& algoExecDesc, u32& subCommMask)
{
    size_t childrenSize = algoExecDesc.children.size();
    u32 localSubCommMask = 0;
    for (size_t i = 0; i < childrenSize; ++i) {
        u32 childrenSubCommMask = 0;
        VariantType& v = algoExecDesc.children[i];
        if (TemplateExecDesc* templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            childrenSubCommMask |= (1U << templateExeDes->subCommIndex);
            CHK_RET(GetTemplateRes(*templateExeDes));
        } else if (auto* hcclAlgorithmPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            CHK_RET(GetResRecursion(**hcclAlgorithmPtr, childrenSubCommMask));
        } else {
            HCCL_ERROR("[GetRes] unexpected variant type at children index %zu", i);
            return HCCL_E_INTERNAL;
        }
        localSubCommMask |= childrenSubCommMask;
    }
    algoExecDesc.subCommMask = localSubCommMask;
    subCommMask = localSubCommMask;
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::CalcTemplateChannelRes(
    HcclComm comm, const TemplateExecDesc& templateExeDes, std::vector<std::vector<HcclChannelDesc>>& requestChannels)
{
    int subCommIndex = templateExeDes.subCommIndex;
    std::vector<u32> templateRanks;
    CHK_RET(GetSubCommRanks(subCommIndex, templateRanks));
    auto baseTemplate = GetTemplate(templateExeDes.templateDesc, templateRanks, myRank_);
    BaseTemplate* baseTemplatePtr = baseTemplate.get();
    CHK_PTR_NULL(baseTemplatePtr);
    baseTemplatePtr->SetNetLayer(templateExeDes.netLayer);
    baseTemplatePtr->SetDataSize(dataInfo_.inputSize);
    AlgResourceRequest tempRequest;
    CHK_RET(baseTemplatePtr->CalcRes(comm, algo_.engineType, tempRequest));
    HCCL_DEBUG(
        "[CalcTemplateChannelRes] myRank=%u, subCommIndex=%d, algType=%d, netLayer=%d, "
        "channelNum=%zu, slaveThreadNum=%u, notifyNumOnMainThread=%u, notifyNumPerThreadSize=%zu",
        myRank_, subCommIndex, static_cast<int>(templateExeDes.templateDesc.algType), templateExeDes.netLayer,
        tempRequest.channels.empty() ? 0 : tempRequest.channels.at(0).size(), tempRequest.slaveThreadNum,
        tempRequest.notifyNumOnMainThread, tempRequest.notifyNumPerThread.size());
    requestChannels.at(subCommIndex) = tempRequest.channels.at(0);
    calcResMaxCh_.at(subCommIndex) = baseTemplatePtr->GetChannelsPerRank();
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::SetTemplateChannelConfig(BaseTemplate* tpl, int subCommIndex)
{
    if (tpl == nullptr) {
        HCCL_ERROR("[SetTemplateChannelConfig] template pointer is null.");
        return HCCL_E_PARA;
    }
    if (!channelTable_.empty() && subCommIndex >= 0 && static_cast<size_t>(subCommIndex) < channelTable_.size()) {
        u32 maxCh = 1;
        for (const auto& pair : channelTable_.at(subCommIndex)) {
            if (pair.second.size() > maxCh) {
                maxCh = static_cast<u32>(pair.second.size());
            }
        }
        tpl->SetChannelsPerRank(maxCh);
    } else if (
        !calcResMaxCh_.empty() && subCommIndex >= 0 && static_cast<size_t>(subCommIndex) < calcResMaxCh_.size()) {
        tpl->SetChannelsPerRank(calcResMaxCh_.at(subCommIndex));
    }
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::GetTemplateRes(const TemplateExecDesc& templateExeDes)
{
    int subCommIndex = templateExeDes.subCommIndex;
    std::vector<u32> templateRanks;
    CHK_RET(GetSubCommRanks(subCommIndex, templateRanks));
    auto baseTemplate = GetTemplate(templateExeDes.templateDesc, templateRanks, myRank_);
    BaseTemplate* baseTemplatePtr = baseTemplate.get();
    CHK_PTR_NULL(baseTemplatePtr);
    CHK_RET(SetTemplateChannelConfig(baseTemplatePtr, subCommIndex));
    AlgResourceRequest tempRequest;
    CHK_RET(baseTemplatePtr->GetRes(tempRequest));
    HCCL_DEBUG(
        "[GetTemplateRes] myRank=%u, subCommIndex=%d, algType=%d, "
        "slaveThreadNum=%u, notifyNumOnMainThread=%u, notifyNumPerThreadSize=%zu",
        myRank_, subCommIndex, static_cast<int>(templateExeDes.templateDesc.algType), tempRequest.slaveThreadNum,
        tempRequest.notifyNumOnMainThread, tempRequest.notifyNumPerThread.size());
    maxSlaveThreadNum_.at(subCommIndex) = std::max(maxSlaveThreadNum_.at(subCommIndex), tempRequest.slaveThreadNum);
    maxNotifyNumOnMainThread_.at(subCommIndex)
        = std::max(maxNotifyNumOnMainThread_.at(subCommIndex), tempRequest.notifyNumOnMainThread);
    if (!tempRequest.notifyNumPerThread.empty()) {
        auto it = std::max_element(tempRequest.notifyNumPerThread.begin(), tempRequest.notifyNumPerThread.end());
        maxNotifyNumPerThread_.at(subCommIndex) = std::max(maxNotifyNumPerThread_.at(subCommIndex), *it);
    }
    HCCL_DEBUG(
        "[GetTemplateRes] myRank=%u, subCommIndex=%d, after merge: maxSlaveThreadNum=%u, "
        "maxNotifyNumOnMainThread=%u, maxNotifyNumPerThread=%u",
        myRank_, subCommIndex, maxSlaveThreadNum_.at(subCommIndex), maxNotifyNumOnMainThread_.at(subCommIndex),
        maxNotifyNumPerThread_.at(subCommIndex));
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::CalcRes(HcclComm comm, AlgResourceRequest& resourceRequest)
{
    if (algo_.algoExecDesc.execPolicy == HcclAlgExecPolicy::OMNIPIPE) {
        u32 eqRankSize = 0;
        double eqBw;
        CHK_RET(OmniPipeUpdateEqBWAndReorder(algo_.algoExecDesc, eqRankSize, eqBw));
    }
    auto topoLevelNum = algHierarchyInfo_.infos.size();
    std::vector<std::vector<HcclChannelDesc>> requestChannels(topoLevelNum);
    calcResMaxCh_.assign(topoLevelNum, 1);
    CHK_RET(CalcChannelResRecursion(comm, algo_.algoExecDesc, requestChannels));
    for (size_t subCommIndex = 0; subCommIndex < topoLevelNum; subCommIndex++) {
        resourceRequest.channels.emplace_back(requestChannels.at(subCommIndex));
    }
    CHK_RET(GetRes(resourceRequest));
    u32 totalChannelNum = 0;
    for (size_t i = 0; i < resourceRequest.channels.size(); ++i) {
        u32 chNum = static_cast<u32>(resourceRequest.channels.at(i).size());
        totalChannelNum += chNum;
        HCCL_DEBUG("[CalcRes][Summary] myRank=%u, subCommIndex=%zu, channelNum=%u", myRank_, i, chNum);
    }
    HCCL_INFO(
        "[CalcRes][TotalSummary] myRank=%u, totalSlaveThreadNum=%u, totalChannelNum=%u, "
        "notifyNumOnMainThread=%u, notifyNumPerThreadSize=%zu, channelLevelNum=%zu",
        myRank_, resourceRequest.slaveThreadNum, totalChannelNum, resourceRequest.notifyNumOnMainThread,
        resourceRequest.notifyNumPerThread.size(), resourceRequest.channels.size());
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::GetRes(AlgResourceRequest& resourceRequest)
{
    auto topoLevelNum = algHierarchyInfo_.infos.size();
    maxSlaveThreadNum_.assign(topoLevelNum, 0);
    maxNotifyNumOnMainThread_.assign(topoLevelNum, 0);
    maxNotifyNumPerThread_.assign(topoLevelNum, 0);
    notifyNumOnSubMainThread_.clear();
    u32 rootSubCommMask = 0;
    CHK_RET(GetResRecursion(algo_.algoExecDesc, rootSubCommMask));
    if (topoLevelNum == 1) {
        resourceRequest.notifyNumOnMainThread = maxNotifyNumOnMainThread_.at(0);
    } else {
        resourceRequest.notifyNumOnMainThread = topoLevelNum;
    }
    resourceRequest.slaveThreadNum = 0;
    for (size_t subCommIndex = 0; subCommIndex < topoLevelNum; subCommIndex++) {
        if (topoLevelNum > 1) {
            resourceRequest.slaveThreadNum += maxSlaveThreadNum_.at(subCommIndex) + 1;
            resourceRequest.notifyNumPerThread.emplace_back(maxNotifyNumOnMainThread_.at(subCommIndex) + 1);
            notifyNumOnSubMainThread_.emplace_back(maxNotifyNumOnMainThread_.at(subCommIndex) + 1);
            resourceRequest.notifyNumPerThread.insert(
                resourceRequest.notifyNumPerThread.end(), maxSlaveThreadNum_.at(subCommIndex),
                maxNotifyNumPerThread_.at(subCommIndex));
        } else {
            resourceRequest.slaveThreadNum = maxSlaveThreadNum_.at(subCommIndex);
            resourceRequest.notifyNumPerThread.emplace_back(maxNotifyNumOnMainThread_.at(subCommIndex));
            resourceRequest.notifyNumPerThread.insert(
                resourceRequest.notifyNumPerThread.end(), maxSlaveThreadNum_.at(subCommIndex),
                maxNotifyNumPerThread_.at(subCommIndex));
        }
    }
    scratchMultiple_ = algo_.hcclCmdType == HcclCMDType::HCCL_CMD_ALLGATHER ? rankSize_ : 1;
    HCCL_INFO(
        "[GetRes][Summary] myRank=%u, topoLevelNum=%zu, totalSlaveThreadNum=%u, "
        "notifyNumOnMainThread=%u, notifyNumPerThreadSize=%zu",
        myRank_, topoLevelNum, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread,
        resourceRequest.notifyNumPerThread.size());
    for (size_t i = 0; i < topoLevelNum; ++i) {
        HCCL_DEBUG(
            "[GetRes][Summary] myRank=%u, subCommIndex=%zu, maxSlaveThreadNum=%u, "
            "maxNotifyNumOnMainThread=%u, maxNotifyNumPerThread=%u",
            myRank_, i, maxSlaveThreadNum_.at(i), maxNotifyNumOnMainThread_.at(i), maxNotifyNumPerThread_.at(i));
    }
    return HCCL_SUCCESS;
}

inline void OpsExecutor::InitAlgoExecDataDesc(
    AlgoExecDataDesc& algoExecDataDesc, u64 dataOffset, u64 dataCount, u64 tailCount, u64 dataStride)
{
    algoExecDataDesc.dataOffset = dataOffset;
    algoExecDataDesc.tailCount = tailCount;
    algoExecDataDesc.globalTailRankId = (tailCount > 0 && rankSize_ > 0) ? rankSize_ - 1 : INVALID_VALUE_RANKID;
    algoExecDataDesc.inputBufferType = BufferType::INPUT;
    if (algo_.hcclCmdType == HCCL_CMD_BROADCAST) {
        algoExecDataDesc.outputBufferType = BufferType::INPUT;
    } else if (algo_.hcclCmdType == HCCL_CMD_REDUCE && myRank_ != root_) {
        algoExecDataDesc.outputBufferType = BufferType::HCCL_BUFFER;
    } else {
        algoExecDataDesc.outputBufferType = BufferType::OUTPUT;
    }
    algoExecDataDesc.cclBufferType = BufferType::HCCL_BUFFER;
    if (algo_.hcclCmdType == HCCL_CMD_ALLGATHER) {
        algoExecDataDesc.sliceCount = dataCount;
        algoExecDataDesc.ranksForInputDataGroup.push_back({myRank_});
    } else {
        algoExecDataDesc.sliceCount = dataCount / rankSize_;
        std::vector<u32> ranksForInputData(rankSize_);
        std::iota(ranksForInputData.begin(), ranksForInputData.end(), 0);
        algoExecDataDesc.ranksForInputDataGroup.push_back(std::move(ranksForInputData));
    }
    const bool useTailStride = tailCount > 0
                               && (algo_.hcclCmdType == HCCL_CMD_BROADCAST || algo_.hcclCmdType == HCCL_CMD_ALLREDUCE
                                   || algo_.hcclCmdType == HCCL_CMD_REDUCE);
    const u64 scratchSliceCount = algoExecDataDesc.sliceCount + (useTailStride ? tailCount : 0);
    algoExecDataDesc.scratchStride = scratchSliceCount * dataTypeSize_;
    algoExecDataDesc.dataStride = dataStride;
}

inline HcclResult
OpsExecutor::GenTemplateDataParams(AlgoExecDataDesc& algoExecDataDesc, DataParams& templateDataParams, u32 overrideRoot)
{
    templateDataParams.cclBufferType = algoExecDataDesc.cclBufferType;
    templateDataParams.cclBufferPtr = cclBufferInfo_.ptr;
    templateDataParams.inputBufferType = algoExecDataDesc.inputBufferType;
    if (templateDataParams.inputBufferType == BufferType::INPUT) {
        templateDataParams.inputBufferPtr = dataInfo_.inputPtr;
    } else if (templateDataParams.inputBufferType == BufferType::OUTPUT) {
        templateDataParams.inputBufferPtr = dataInfo_.outputPtr;
    } else if (templateDataParams.inputBufferType == BufferType::HCCL_BUFFER) {
        templateDataParams.inputBufferPtr = cclBufferInfo_.ptr;
    } else {
        HCCL_ERROR("[GenTemplateDataParams] inputBufferType = %d!", templateDataParams.inputBufferType);
        return HCCL_E_INTERNAL;
    }
    templateDataParams.outputBufferType = algoExecDataDesc.outputBufferType;
    if (templateDataParams.outputBufferType == BufferType::INPUT) {
        templateDataParams.outputBufferPtr = dataInfo_.inputPtr;
    } else if (templateDataParams.outputBufferType == BufferType::OUTPUT) {
        templateDataParams.outputBufferPtr = dataInfo_.outputPtr;
    } else if (templateDataParams.outputBufferType == BufferType::HCCL_BUFFER) {
        templateDataParams.outputBufferPtr = cclBufferInfo_.ptr;
    } else {
        HCCL_ERROR("[GenTemplateDataParams] outputBufferType = %d!", templateDataParams.outputBufferType);
        return HCCL_E_INTERNAL;
    }
    templateDataParams.dataType = dataInfo_.dataType;
    templateDataParams.sliceCount = algoExecDataDesc.sliceCount;
    templateDataParams.sliceOffset = algoExecDataDesc.sliceOffset;
    templateDataParams.tailCount = algoExecDataDesc.tailCount;
    templateDataParams.globalTailRankId = algoExecDataDesc.globalTailRankId;
    templateDataParams.dataOffset = algoExecDataDesc.dataOffset;
    templateDataParams.reduceOp = dataInfo_.reduceOp;
    templateDataParams.root = (overrideRoot != INVALID_VALUE_RANKID) ? overrideRoot : root_;
    templateDataParams.enableRemoteMemAccess = opMode_ == OpMode::OFFLOAD;
    templateDataParams.userRankSize = rankSize_;
    templateDataParams.dataStride = algoExecDataDesc.dataStride;
    templateDataParams.scratchStride = algoExecDataDesc.scratchStride;
    if (algoExecDataDesc.ranksForInputDataGroup.size() != 1) {
        HCCL_ERROR(
            "[GenTemplateDataParams] ranksForInputDataGroup size = %zu!",
            algoExecDataDesc.ranksForInputDataGroup.size());
        return HCCL_E_INTERNAL;
    }
    templateDataParams.ranksForInputData = algoExecDataDesc.ranksForInputDataGroup.at(0);
    return HCCL_SUCCESS;
}

inline HcclResult OpsExecutor::UpdateDataSplitParallel(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, u32 childrenId,
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    // 校验 dataSplitRatio 条目数与 children 一致
    if (algoExecDesc.dataSplitRatio.size() != childrenSize) {
        HCCL_ERROR(
            "[UpdateDataSplitParallel] dataSplitRatio size (%zu) != childrenSize (%zu)!",
            algoExecDesc.dataSplitRatio.size(), childrenSize);
        return HCCL_E_PARA;
    }
    u64 dataSplitRatioSum
        = std::accumulate(algoExecDesc.dataSplitRatio.begin(), algoExecDesc.dataSplitRatio.end(), 0ULL);
    if (dataSplitRatioSum == 0) {
        HCCL_ERROR("[UpdateDataSplitParallel] dataSplitRatioSum is 0, all ratios are zero!");
        return HCCL_E_PARA;
    }
    // sliceOffset 累加前一 Child 覆盖范围
    if (childrenId > 0) {
        childrenAlgoExecDataDesc.at(childrenId).sliceOffset
            = childrenAlgoExecDataDesc.at(childrenId - 1).sliceOffset
              + childrenAlgoExecDataDesc.at(childrenId - 1).sliceCount * dataTypeSize_;
    }
    u64 parentSlice = algoExecDataDesc.sliceCount;
    u64 ratio = static_cast<u64>(algoExecDesc.dataSplitRatio.at(childrenId));
    u64 sliceCount;
    if (childrenId == childrenSize - 1) {
        // 最后一个 Child 承接整除余量，保证数据不丢失
        u64 allocated = 0;
        for (size_t i = 0; i < childrenSize - 1; i++) {
            allocated += childrenAlgoExecDataDesc.at(i).sliceCount;
        }
        if (allocated > parentSlice) {
            HCCL_ERROR(
                "[UpdateDataSplitParallel] allocated (%llu) > parentSlice (%llu), slicing overflow!", allocated,
                parentSlice);
            return HCCL_E_INTERNAL;
        }
        sliceCount = parentSlice - allocated;
    } else {
        // 整数运算：childSlice[i] = floor(parentSlice * ratio[i] / sum)
        sliceCount = CalcParallelSliceCount(parentSlice, ratio, dataSplitRatioSum);
    }
    childrenAlgoExecDataDesc.at(childrenId).sliceCount = sliceCount;
    childrenAlgoExecDataDesc.at(childrenId).tailCount
        = (childrenId == childrenSize - 1) ? algoExecDataDesc.tailCount : 0;
    childrenAlgoExecDataDesc.at(childrenId).globalTailRankId
        = (childrenId == childrenSize - 1) ? algoExecDataDesc.globalTailRankId : INVALID_VALUE_RANKID;
    childrenAlgoExecDataDesc.at(childrenId).ranksForInputDataGroup.clear();
    size_t ranksForInputDataGroupSize = algoExecDataDesc.ranksForInputDataGroup.size();
    if (ranksForInputDataGroupSize != 1 && ranksForInputDataGroupSize != childrenSize) {
        HCCL_ERROR(
            "[UpdateDataSplitParallel] ranksForInputDataGroupSize (%zu) matches neither 1 nor childrenSize (%zu)!",
            ranksForInputDataGroupSize, childrenSize);
        return HCCL_E_INTERNAL;
    }
    u32 inputGroupIdx = (ranksForInputDataGroupSize == 1) ? 0 : childrenId;
    childrenAlgoExecDataDesc.at(childrenId)
        .ranksForInputDataGroup.emplace_back(algoExecDataDesc.ranksForInputDataGroup.at(inputGroupIdx));
    return HCCL_SUCCESS;
}

inline void OpsExecutor::UpdateDataSplitSequence(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, u32 childrenId,
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    if (childrenId > 0) {
        childrenAlgoExecDataDesc.at(childrenId).ranksForInputDataGroup
            = childrenAlgoExecDataDesc.at(childrenId - 1).ranksForOutputDataGroup;
        childrenAlgoExecDataDesc.at(childrenId).inputBufferType
            = childrenAlgoExecDataDesc.at(childrenId - 1).outputBufferType;
    }
    childrenAlgoExecDataDesc.at(childrenId).outputBufferType
        = (childrenId == childrenSize - 1) ? algoExecDataDesc.outputBufferType : algoExecDataDesc.cclBufferType;
    return;
}

HcclResult OpsExecutor::MergeChildrenOutput(
    const AlgoExecDesc& algoExecDesc, const std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc,
    AlgoExecDataDesc& algoExecDataDesc)
{
    if (childrenAlgoExecDataDesc.empty()) {
        HCCL_ERROR("[MergeChildrenOutput] childrenAlgoExecDataDesc is empty.");
        return HCCL_E_INTERNAL;
    }
    if (algoExecDesc.execPolicy == HcclAlgExecPolicy::PARALLEL) {
        u64 sliceCount = 0;
        for (const auto& child : childrenAlgoExecDataDesc) {
            sliceCount += child.sliceCount;
        }
        algoExecDataDesc.sliceCount = sliceCount;
        bool allEqual = true;
        const auto& first = childrenAlgoExecDataDesc[0].ranksForOutputDataGroup;
        for (size_t i = 1; i < childrenAlgoExecDataDesc.size(); ++i) {
            if (childrenAlgoExecDataDesc[i].ranksForOutputDataGroup != first) {
                allEqual = false;
                break;
            }
        }
        if (allEqual) {
            algoExecDataDesc.ranksForOutputDataGroup = childrenAlgoExecDataDesc.back().ranksForOutputDataGroup;
        } else {
            algoExecDataDesc.ranksForOutputDataGroup.clear();
            for (const auto& child : childrenAlgoExecDataDesc) {
                if (child.ranksForOutputDataGroup.empty()) {
                    HCCL_ERROR("[MergeChildrenOutput] child.ranksForOutputDataGroup is empty.");
                    return HCCL_E_INTERNAL;
                }
                for (const auto& group : child.ranksForOutputDataGroup) {
                    algoExecDataDesc.ranksForOutputDataGroup.emplace_back(group);
                }
            }
        }
    } else {
        algoExecDataDesc.ranksForOutputDataGroup = childrenAlgoExecDataDesc.back().ranksForOutputDataGroup;
    }
    return HCCL_SUCCESS;
}

HcclResult
OpsExecutor::RunTemplateDesc(TemplateExecDesc* templateExeDes, AlgoExecDataDesc& algoExecDataDesc, u32 overrideRoot)
{
    HCCL_DEBUG(
        "[RunTemplateDesc][myRank_:%d:] templateExeDes: hcclCmdType=%d, algType=%d, subCommIndex=%d", myRank_,
        static_cast<int>(templateExeDes->templateDesc.hcclCmdType),
        static_cast<int>(templateExeDes->templateDesc.algType), templateExeDes->subCommIndex);
    int subCommIndex = templateExeDes->subCommIndex;
    if (subCommIndex < 0 || static_cast<size_t>(subCommIndex) >= cachedTemplateResources_.size()) {
        HCCL_ERROR(
            "[RunTemplateDesc] subCommIndex=%d out of range (cache size=%zu)", subCommIndex,
            cachedTemplateResources_.size());
        return HCCL_E_PARA;
    }
    auto baseTemplate = GetTemplate(templateExeDes->templateDesc, cachedTemplateRanks_.at(subCommIndex), myRank_);
    BaseTemplate* baseTemplatePtr = baseTemplate.get();
    CHK_PTR_NULL(baseTemplatePtr);
    // 运行实例由 GetTemplate 新建，channelsPerRank_ 为默认值 1、dataSize_ 为默认值 0，
    // 需补齐与资源阶段（GetTemplateRes）一致的设置，否则多通道并行 PostCopy 仅覆盖 channel 0 导致数据丢失
    CHK_RET(SetTemplateChannelConfig(baseTemplatePtr, subCommIndex));
    baseTemplatePtr->SetDataSize(dataInfo_.inputSize);
    TemplateResource& templateResource = cachedTemplateResources_.at(subCommIndex);
    DataParams templateDataParams;
    CHK_RET(GenTemplateDataParams(algoExecDataDesc, templateDataParams, overrideRoot));
    HCCL_DEBUG(
        "[RunTemplateDesc][myRank_:%d:] inputBufferType=%d, outputBufferType=%d, cclBufferType=%d, "
        "dataType=%d, dataOffset=%lu, sliceCount=%lu, sliceOffset=%lu, tailCount=%lu, "
        "globalTailRankId=%u, dataStride=%lu, "
        "scratchStride=%lu, reduceOp=%d, root=%u, enableRemoteMemAccess=%d",
        myRank_, static_cast<int>(templateDataParams.inputBufferType),
        static_cast<int>(templateDataParams.outputBufferType), static_cast<int>(templateDataParams.cclBufferType),
        static_cast<int>(templateDataParams.dataType), templateDataParams.dataOffset, templateDataParams.sliceCount,
        templateDataParams.sliceOffset, templateDataParams.tailCount, templateDataParams.globalTailRankId,
        templateDataParams.dataStride, templateDataParams.scratchStride, static_cast<int>(templateDataParams.reduceOp),
        templateDataParams.root, static_cast<int>(templateDataParams.enableRemoteMemAccess));
    algoExecDataDesc.ranksForOutputDataGroup.resize(1);
    CHK_RET(baseTemplatePtr->KernelRun(
        templateDataParams, templateResource, algoExecDataDesc.ranksForOutputDataGroup.at(0)));
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::InitSubCommRoots()
{
    subCommRoots_.clear();
    subCommRoots_.reserve(algHierarchyInfo_.infos.size());
    u64 lowerLevelsRankSize = 1;
    for (u32 level = 0; level < algHierarchyInfo_.infos.size(); ++level) {
        const auto& subCommGroups = algHierarchyInfo_.infos[level];
        CHK_PRT_RET(
            subCommGroups.empty() || subCommGroups.at(0).empty(),
            HCCL_ERROR("[InitSubCommRoots] subComm ranks at level[%u] are empty.", level), HCCL_E_PARA);
        const auto& subCommRanks = subCommGroups.at(0);
        const u32 rootAlgRank = static_cast<u32>((static_cast<u64>(root_) / lowerLevelsRankSize) % subCommRanks.size());
        subCommRoots_.emplace_back(subCommRanks[rootAlgRank]);
        lowerLevelsRankSize *= subCommRanks.size();
    }
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OrchestrateLoop(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, std::vector<AlgoExecDataDesc>* reusableChildren)
{
    size_t childrenSize = algoExecDesc.children.size();
    std::vector<AlgoExecDataDesc> localChildren;
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc
        = (reusableChildren != nullptr) ? *reusableChildren : localChildren;

    InitChildrenDataDesc(algoExecDesc, algoExecDataDesc, childrenAlgoExecDataDesc);

    if (algoExecDesc.execPolicy == HcclAlgExecPolicy::PARALLEL && childrenSize > 1) {
        CHK_RET(PreSyncBySubCommMask(algoExecDesc));
    }

    CHK_RET(ExecChildren(algoExecDesc, algoExecDataDesc, childrenAlgoExecDataDesc));

    CHK_RET(MergeChildrenOutput(algoExecDesc, childrenAlgoExecDataDesc, algoExecDataDesc));
    if (algoExecDesc.execPolicy == HcclAlgExecPolicy::PARALLEL && childrenSize > 1) {
        CHK_RET(PostSyncBySubCommMask(algoExecDesc));
    }
    return HCCL_SUCCESS;
}

void OpsExecutor::InitChildrenDataDesc(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc,
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    if (childrenAlgoExecDataDesc.size() != childrenSize) {
        childrenAlgoExecDataDesc.assign(childrenSize, algoExecDataDesc);
    } else {
        for (size_t i = 0; i < childrenSize; ++i) {
            auto& child = childrenAlgoExecDataDesc.at(i);
            child.dataOffset = algoExecDataDesc.dataOffset;
            child.sliceCount = algoExecDataDesc.sliceCount;
            child.sliceOffset = algoExecDataDesc.sliceOffset;
            child.tailCount = algoExecDataDesc.tailCount;
            child.globalTailRankId = algoExecDataDesc.globalTailRankId;
            child.dataStride = algoExecDataDesc.dataStride;
            child.scratchStride = algoExecDataDesc.scratchStride;
            child.inputBufferType = algoExecDataDesc.inputBufferType;
            child.outputBufferType = algoExecDataDesc.outputBufferType;
            child.cclBufferType = algoExecDataDesc.cclBufferType;
            child.ranksForInputDataGroup.clear();
            child.ranksForOutputDataGroup.clear();
            if (i == 0) {
                child.ranksForInputDataGroup = algoExecDataDesc.ranksForInputDataGroup;
            }
        }
    }
}

HcclResult OpsExecutor::ExecChildren(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc,
    std::vector<AlgoExecDataDesc>& childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    for (size_t i = 0; i < childrenSize; ++i) {
        if (algoExecDesc.execPolicy == HcclAlgExecPolicy::PARALLEL && childrenSize > 1) {
            CHK_RET(UpdateDataSplitParallel(algoExecDesc, algoExecDataDesc, i, childrenAlgoExecDataDesc));
        } else {
            UpdateDataSplitSequence(algoExecDesc, algoExecDataDesc, i, childrenAlgoExecDataDesc);
        }
        VariantType& v = algoExecDesc.children[i];
        if (TemplateExecDesc* templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            if (algoExecDesc.execPolicy == HcclAlgExecPolicy::SEQUENCE && childrenSize > 1) {
                CHK_RET(PreSyncSingleSubComm(templateExeDes->subCommIndex));
            }
            u32 overrideRoot = INVALID_VALUE_RANKID;
            const HcclCMDType cmdType = templateExeDes->templateDesc.hcclCmdType;
            if (cmdType == HcclCMDType::HCCL_CMD_SCATTER || cmdType == HcclCMDType::HCCL_CMD_GATHER) {
                if (subCommRoots_.empty()) {
                    CHK_RET(InitSubCommRoots());
                }
                if (templateExeDes->subCommIndex < 0
                    || static_cast<size_t>(templateExeDes->subCommIndex) >= subCommRoots_.size()) {
                    HCCL_ERROR(
                        "[ExecChildren] subCommIndex=%d out of range (subCommRoots size=%zu)",
                        templateExeDes->subCommIndex, subCommRoots_.size());
                    return HCCL_E_PARA;
                }
                overrideRoot = subCommRoots_[templateExeDes->subCommIndex];
            }
            CHK_RET(RunTemplateDesc(templateExeDes, childrenAlgoExecDataDesc.at(i), overrideRoot));
            if (algoExecDesc.execPolicy == HcclAlgExecPolicy::SEQUENCE && childrenSize > 1) {
                CHK_RET(PostSyncSingleSubComm(templateExeDes->subCommIndex));
            }
        } else if (auto* hcclAlgorithmPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            CHK_RET(OrchestrateLoop(**hcclAlgorithmPtr, childrenAlgoExecDataDesc.at(i), nullptr));
        } else {
            HCCL_ERROR("[ExecChildren] unexpected variant type at children index %zu", i);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OrchestrateOmniPipeLoop(AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc)
{
    if (algoExecDesc.omniPipeXYdata.steps == 0) {
        HCCL_ERROR("[OpsExecutor] OmniPipeXYdata not computed for AlgoExecDesc");
        return HCCL_E_INTERNAL;
    }
    OmniPipeXYdata omniPipeXYdata = algoExecDesc.omniPipeXYdata;
    size_t childrenSize = algoExecDesc.children.size();
    std::vector<std::vector<AlgoExecDataDesc>> childrenExecDataDesc;
    CHK_RET(OmniPipeCalcExecData(algoExecDesc, algoExecDataDesc, omniPipeXYdata, childrenExecDataDesc));
    for (size_t i = 0; i < omniPipeXYdata.steps; i++) {
        CHK_RET(PreSyncBySubCommMask(algoExecDesc));
        for (size_t j = 0; j < childrenSize; j++) {
            VariantType& v = algoExecDesc.children[j];
            if (TemplateExecDesc* templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
                CHK_RET(RunTemplateDesc(templateExeDes, childrenExecDataDesc.at(j).at(i)));
            } else if (auto* hcclAlgorithmPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
                CHK_RET(OrchestrateOmniPipeLoop(**hcclAlgorithmPtr, childrenExecDataDesc.at(j).at(i)));
            } else {
                HCCL_ERROR("[OrchestrateOmniPipe] unexpected variant type at step=%zu, child index=%zu", i, j);
                return HCCL_E_INTERNAL;
            }
        }
        CHK_RET(PostSyncBySubCommMask(algoExecDesc));
    }
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OmniPipeCalcExecData(
    AlgoExecDesc& algoExecDesc, AlgoExecDataDesc& algoExecDataDesc, OmniPipeXYdata& omniPipeXYdata,
    std::vector<std::vector<AlgoExecDataDesc>>& childrenExecDataDesc)
{
    u32 steps = omniPipeXYdata.steps;
    std::vector<AlgoExecDataDesc> xExecDataDesc{steps, algoExecDataDesc};
    std::vector<AlgoExecDataDesc> yExecDataDesc{steps, algoExecDataDesc};
    CHK_RET(OmniPipeUpdateDataSlice(omniPipeXYdata, algoExecDataDesc, xExecDataDesc, yExecDataDesc));
    for (u32 i = 0; i < steps; i++) {
        if (i == steps - 1) {
            std::vector<u32> xRanksForInput;
            CHK_RET(CalcPeerAxisRanksForOutput(
                algoExecDesc, 1, algoExecDataDesc.ranksForInputDataGroup.at(0), xRanksForInput));
            xExecDataDesc.at(i).ranksForInputDataGroup.clear();
            xExecDataDesc.at(i).ranksForInputDataGroup.emplace_back(xRanksForInput);
        }
        if (i > 0) {
            std::vector<u32> yRanksForInput;
            CHK_RET(CalcPeerAxisRanksForOutput(
                algoExecDesc, 0, algoExecDataDesc.ranksForInputDataGroup.at(0), yRanksForInput));
            yExecDataDesc.at(i).ranksForInputDataGroup.clear();
            yExecDataDesc.at(i).ranksForInputDataGroup.emplace_back(yRanksForInput);
        }
    }
    childrenExecDataDesc.emplace_back(xExecDataDesc);
    childrenExecDataDesc.emplace_back(yExecDataDesc);
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OmniPipeUpdateDataSlice(
    OmniPipeXYdata& omniPipeXYdata, const AlgoExecDataDesc& algoExecDataDesc,
    std::vector<AlgoExecDataDesc>& xExecDataDesc, std::vector<AlgoExecDataDesc>& yExecDataDesc)
{
    u32 xRankSize = omniPipeXYdata.xEqRankSize;
    u32 yRankSize = omniPipeXYdata.yEqRankSize;
    if (xRankSize <= 1 || yRankSize <= 1) {
        HCCL_ERROR("[OmniPipeUpdateDataSlice] invalid axis rankSize x=%u y=%u (must > 1)", xRankSize, yRankSize);
        return HCCL_E_INTERNAL;
    }
    u32 steps = omniPipeXYdata.steps;
    u64 sliceCount = algoExecDataDesc.sliceCount;
    std::vector<u64> xSliceCount(steps);
    std::vector<u64> ySliceCount(steps);
    u64 sumXCount = 0;
    u64 sumYCount = 0;
    CHK_RET(CalcOmniPipeDataSlice(omniPipeXYdata, sliceCount, xSliceCount, ySliceCount));
    for (u32 i = 0; i < steps; i++) {
        yExecDataDesc.at(i).sliceCount = i == 0 ? ySliceCount[i] : (ySliceCount[i] / (xRankSize - 1));
        if (i == 0 || i == 1) {
            yExecDataDesc.at(i).sliceOffset = algoExecDataDesc.sliceOffset;
        } else {
            yExecDataDesc.at(i).sliceOffset
                = algoExecDataDesc.sliceOffset + (sumYCount - sliceCount) * dataTypeSize_ / (xRankSize - 1);
        }
        sumYCount += ySliceCount[i];
        xExecDataDesc.at(i).sliceCount = xSliceCount[i];
        if (i == steps - 1) {
            xExecDataDesc.at(i).sliceCount = xSliceCount[i] / (yRankSize - 1);
            xExecDataDesc.at(i).sliceOffset
                = yExecDataDesc.at(i).sliceOffset + yExecDataDesc.at(i).sliceCount * dataTypeSize_;
        } else {
            xExecDataDesc.at(i).sliceOffset = algoExecDataDesc.sliceOffset + sumXCount * dataTypeSize_;
        }
        sumXCount += xSliceCount[i];
        HCCL_DEBUG(
            "[myRank:%u] step= %u, yExecDataDesc.at(i).sliceCount=%llu, yExecDataDesc.at(i).sliceOffset =%llu, "
            "xExecDataDesc.at(i).sliceCount=%llu, xExecDataDesc.at(i).sliceOffset=%llu, xSliceCount[i]=%llu, "
            "ySliceCount[i]=%llu",
            myRank_, i, yExecDataDesc.at(i).sliceCount, yExecDataDesc.at(i).sliceOffset, xExecDataDesc.at(i).sliceCount,
            xExecDataDesc.at(i).sliceOffset, xSliceCount[i], ySliceCount[i]);
    }
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::CalcPeerAxisRanksForOutput(
    const AlgoExecDesc& algoExecDesc, u32 peerChildrenId, std::vector<u32> ranksForInput,
    std::vector<u32>& detaRanksForOutput)
{
    std::vector<u32> subCommRanks;
    std::vector<u32> ranksForOutput;
    const VariantType& v = algoExecDesc.children[peerChildrenId];
    if (const TemplateExecDesc* templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
        CHK_RET(GetSubCommRanks(templateExeDes->subCommIndex, subCommRanks));
        CHK_RET(CalcRanksForOutput(ranksForInput, subCommRanks, myRank_, ranksForOutput));
    } else if (auto* hcclAlgorithmPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
        u32 subCommMask = (*hcclAlgorithmPtr)->subCommMask;
        std::vector<u32> tempRanksForInput = ranksForInput;
        auto topoLevelNum = algHierarchyInfo_.infos.size();
        for (int i = 0; i < topoLevelNum; i++) {
            if (subCommMask & (1u << i)) {
                CHK_RET(GetSubCommRanks(i, subCommRanks));
                CHK_RET(CalcRanksForOutput(tempRanksForInput, subCommRanks, myRank_, ranksForOutput));
                tempRanksForInput = ranksForOutput;
            }
        }
    }
    std::set_difference(
        ranksForOutput.begin(), ranksForOutput.end(), ranksForInput.begin(), ranksForInput.end(),
        std::back_inserter(detaRanksForOutput));
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OmniPipeUpdateEqBWAndReorder(AlgoExecDesc& algoExecDesc, u32& eqRankSize, double& eqBw)
{
    if (algoExecDesc.children.size() != 2) {
        HCCL_ERROR("[CalcEqBW] algoExecDesc children size=%zu, expected 2.", algoExecDesc.children.size());
        return HCCL_E_INTERNAL;
    }
    VariantType& xChild = algoExecDesc.children[0];
    VariantType& yChild = algoExecDesc.children[1];
    u32 xChildRankSize = 0;
    u32 yChildRankSize = 0;
    double xChildEqBw = 0;
    double yChildEqBw = 0;
    CHK_RET(OmniPipeCalcChildEqBW(xChild, xChildRankSize, xChildEqBw));
    CHK_RET(OmniPipeCalcChildEqBW(yChild, yChildRankSize, yChildEqBw));
    if (xChildEqBw > yChildEqBw) {
        std::swap(algoExecDesc.children[0], algoExecDesc.children[1]);
        std::swap(xChildRankSize, yChildRankSize);
        std::swap(xChildEqBw, yChildEqBw);
    }
    eqRankSize = xChildRankSize;
    u32 steps = 1;
    double scale = 1.0;
    eqBw = CalcBandwidth2D(xChildEqBw, yChildEqBw, xChildRankSize, yChildRankSize, OMNI_MAX_STEP_NUM, steps, scale);
    OmniPipeXYdata omniPipeXYdata;
    omniPipeXYdata.xEqRankSize = xChildRankSize;
    omniPipeXYdata.yEqRankSize = yChildRankSize;
    omniPipeXYdata.bandwidthRatio = yChildEqBw / xChildEqBw;
    omniPipeXYdata.steps = steps;
    omniPipeXYdata.scale = scale;
    algoExecDesc.omniPipeXYdata = omniPipeXYdata;
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OmniPipeCalcChildEqBW(VariantType& child, u32& eqRankSize, double& eqBw)
{
    if (TemplateExecDesc* templateExeDes = std::get_if<TemplateExecDesc>(&child)) {
        uint32_t subCommIndex = templateExeDes->subCommIndex;
        if (subCommIndex >= algHierarchyInfo_.infos.size()) {
            HCCL_ERROR("[CalcEqBW]subCommIndex =%d out of range! ", subCommIndex);
            return HCCL_E_INTERNAL;
        }
        eqRankSize = algHierarchyInfo_.infos.at(subCommIndex).at(0).size();
        if (subCommIndex > 0 && eqRankSize <= 1) {
            HCCL_ERROR("[CalcEqBW] layer[%u] eqRankSize=%u ≤ 1, cannot compute CLOS eqBw", subCommIndex, eqRankSize);
            return HCCL_E_INTERNAL;
        }
        eqBw = (subCommIndex == 0) ? OMNI_MESH_BW : OMNI_CLOS_BW / (eqRankSize - 1);
        return HCCL_SUCCESS;
    }
    auto* hcclAlgorithmPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&child);
    if (hcclAlgorithmPtr == nullptr) {
        HCCL_ERROR("[CalcEqBW] child is neither TemplateExecDesc nor shared_ptr<AlgoExecDesc>.");
        return HCCL_E_INTERNAL;
    }
    return OmniPipeUpdateEqBWAndReorder(**hcclAlgorithmPtr, eqRankSize, eqBw);
}

HcclResult OpsExecutor::PreSyncSingleSubComm(u32 subCommIndex)
{
    if (algHierarchyInfo_.infos.size() == 1) {
        return HCCL_SUCCESS;
    }
    ThreadHandle subMain = subThreads_.at(subCommIndex).at(0);
    u32 notifyIdx = notifyNumOnSubMainThread_.at(subCommIndex) - 1;
    return PreSyncInterThreads(mainThread_, {subMain}, {notifyIdx});
}

HcclResult OpsExecutor::PostSyncSingleSubComm(u32 subCommIndex)
{
    if (algHierarchyInfo_.infos.size() == 1) {
        return HCCL_SUCCESS;
    }
    ThreadHandle subMain = subThreads_.at(subCommIndex).at(0);
    return PostSyncInterThreads(mainThread_, {subMain}, {subCommIndex});
}

} // namespace ops_hccl
