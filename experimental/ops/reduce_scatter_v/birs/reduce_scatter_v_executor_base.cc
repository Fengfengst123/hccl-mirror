/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_executor_base.h"

namespace ops_hccl_experimental {
using ops_hccl::ExecMem;
using ops_hccl::HCCL_INTERNODE_MAX_DATA_RATE;
using ops_hccl::RDMA_SEND_MAX_SIZE;
using ops_hccl::SDMA_SEND_MAX_SIZE;

ReduceScatterVExecutorBase::ReduceScatterVExecutorBase() : ExecutorBase() {}

// 执行入口
HcclResult ReduceScatterVExecutorBase::Orchestrate(const OpParam& param, AlgResourceCtx* resCtx)
{
    HcclUs startut = TIME_NOW();
    topoInfo_ = &(resCtx->topoInfo);
    algResource_ = resCtx;
    tag_ = std::string(param.tag);
    algType_ = resCtx->algType;
    unitSize_ = HCCL_SIZE_TABLE[param.vDataDes.dataType];

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    // 做参数的还原
    ThreadHandle* threadHandlePtr
        = reinterpret_cast<ThreadHandle*>(reinterpret_cast<char*>(algResource_) + sizeof(AlgResourceCtx));
    ChannelInfo* channelInfoPtr = reinterpret_cast<ChannelInfo*>(
        reinterpret_cast<char*>(threadHandlePtr) + sizeof(ThreadHandle) * (algResource_->slaveThreadNum + 1));

    HCCL_DEBUG("[ReduceScatterVExecutorBase][Orchestrate] slaveThreadNum[%u]", algResource_->slaveThreadNum);
    for (u32 i = 0; i < algResource_->slaveThreadNum + 1; i++) {
        HCCL_DEBUG("[ReduceScatterVExecutorBase][Orchestrate] threadHandle[%u]=[%llu]", i, threadHandlePtr[i]);
        if (i == 0) {
            thread_ = threadHandlePtr[i];
        } else {
            slaveThreads_.push_back(threadHandlePtr[i]);
        }
    }
    AlgHierarchyInfo& algHierarchyInfo = resCtx->algHierarchyInfo;
    channels_.resize(algHierarchyInfo.levels);
    for (u32 level = 0; level < algHierarchyInfo.levels; level++) {
        u32 curLevelRankSize = algHierarchyInfo.infos[level].localRankSize;
        channels_[level].resize(curLevelRankSize);
        for (u32 rank = 0; rank < curLevelRankSize; rank++) {
            channels_[level][rank] = channelInfoPtr[rank];
        }
        channelInfoPtr += curLevelRankSize;
    }

    HcclResult ret = RunLoop(param);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceScatterVExecutorBase][Orchestrate]errNo[0x%016llx]"
            " ReduceScatterV executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO(
        "[ReduceScatterVExecutorBase][Orchestrate]tag[%s]"
        " ReduceScatterV executor orchestrate success, take time [%lld]us.",
        param.tag, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

bool ReduceScatterVExecutorBase::IsHugeData(u64 curSize) const
{
    bool hugeData = curSize * topoInfo_->userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult ReduceScatterVExecutorBase::AcquireCommIfNeeded(const OpParam& param)
{
    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        int32_t ret = HcommAcquireComm(param.commName);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] [%s] HcommAcquireComm failed ", __func__, param.commName),
            static_cast<HcclResult>(ret));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVExecutorBase::ReleaseCommIfNeeded(const OpParam& param)
{
    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        int32_t ret = HcommReleaseComm(param.commName);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] [%s] HcommReleaseComm failed ", __func__, param.commName),
            static_cast<HcclResult>(ret));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVExecutorBase::RunLoop(const OpParam& param)
{
    // V变体: 本rank接收量为outputSize/unitSize, 总发送量为inputSize
    u64 totalRecvCount = param.outputSize / unitSize_;
    u64 totalRecvSize = param.outputSize;

    u8* curUserInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curUserOutputPtr = static_cast<u8*>(param.outputPtr);
    auto cclInputMem = algResource_->cclInputMem;
    auto cclOutputMem = algResource_->cclOutputMem;
    CHK_PRT_RET(
        (cclInputMem.size == 0), HCCL_ERROR("[ReduceScatterVExecutorBase][RunLoop]cclBuffer size is zero"),
        HCCL_E_PARA);

    CHK_RET(AcquireCommIfNeeded(param));

    u64 curRecvCount = totalRecvCount;
    u64 curRecvSize = totalRecvSize;
    // V变体: 发送总量为inputSize (按各rank的sendCounts/sendDispls布局)
    u64 curSendSize = param.inputSize;

#ifndef AICPU_COMPILE
    if (!IsHugeData(curRecvSize)) {
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
    }
#endif
    // RunLoop 构造 curInputMem 前校验容量
    CHK_PRT_RET(
        curSendSize > cclInputMem.size,
        HCCL_ERROR("[RunLoop]inputSize[%llu] exceeds cclInputMem size[%llu]", curSendSize, cclInputMem.size),
        HCCL_E_PARA);
    // scratch 实际可用区域为整个 cclInputMem(CCL buffer 前半区), 声明大小必须传真实容量而非 curSendSize:
    // 模板以 rankSize*localStrideSize(按 max 切片对齐)为 scratch 需求, 小消息时该需求可大于 inputSize,
    // 若按 curSendSize 声明会使 CheckScratchCapacity 误报容量不足
    HcclMem curInputMem{cclInputMem.type, cclInputMem.addr, cclInputMem.size};
    HcclMem curOutputMem{cclOutputMem.type, cclOutputMem.addr, curRecvSize};

    ExecMem execMem;
    execMem.count = curRecvCount;
    execMem.inputMem = curInputMem;
    execMem.outputMem = curOutputMem;
    execMem.inputPtr = curUserInputPtr;
    execMem.outputPtr = curUserOutputPtr;

    HCCL_DEBUG(
        "[ReduceScatterVExecutorBase][RunLoop] curUserInputPtr[%p], curUserOutputPtr[%p],"
        " curRecvCount[%llu], curRecvSize[%llu], curSendSize[%llu], inputPtr[%p], outputPtr[%p]",
        curUserInputPtr, curUserOutputPtr, curRecvCount, curRecvSize, curSendSize, curInputMem.addr, curOutputMem.addr);

    CHK_RET(KernelRun(param, execMem));

#ifndef AICPU_COMPILE
    if (!IsHugeData(curRecvSize)) {
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    }
#endif
    CHK_RET(ReleaseCommIfNeeded(param));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVExecutorBase::CalcResRequest(
    HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgResourceRequest& resourceRequest, AlgType& algType)
{
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVExecutorBase::PrepareDataSliceV(
    const u64* sendCounts, const u64* sendDispls, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice)
{
    CHK_PRT_RET(
        (sliceNum == 0), HCCL_ERROR("[ReduceScatterVExecutorBase][PrepareDataSliceV]sliceNum is zero."), HCCL_E_PARA);

    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(sendDispls);
    dataSlice.resize(sliceNum);
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sendCounts[i] * unitSize;
        dataSlice[i].offset = sendDispls[i] * unitSize;
        HCCL_DEBUG(
            "[ReduceScatterVExecutorBase][PrepareDataSliceV] slice[%u]: offset[%llu] size[%llu]", i,
            dataSlice[i].offset, dataSlice[i].size);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_experimental
