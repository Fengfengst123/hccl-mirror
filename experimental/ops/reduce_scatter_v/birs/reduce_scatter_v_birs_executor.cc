/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "reduce_scatter_v_birs_executor.h"
#include "topo_experimental.h"

namespace ops_hccl_experimental {
ReduceScatterVBIRSExecutor::ReduceScatterVBIRSExecutor() : ReduceScatterVExecutorBase()
{
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_RING};
}

HcclResult ReduceScatterVBIRSExecutor::CalcResRequest(
    HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgResourceRequest& resourceRequest, AlgType& algType)
{
    if (topoInfo->serverNum == 1) {
        CHK_RET(CalcGeneralTopoInfoForA3(comm, topoInfo, algHierarchyInfo));
    } else {
        CHK_RET(CalcGeneralTopoInfoInterServer(comm, topoInfo, algHierarchyInfo));
    }
    CHK_RET(RefreshAlgType(algType));
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;

    resourceRequest.slaveThreadNum = BIRS_THREAD_NUM;
    for (u32 index = 0; index < BIRS_THREAD_NUM; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = BIRS_THREAD_NUM;

    // level0 channel
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcLevel0ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO(
        "[ReduceScatterVBIRSExecutor][CalcResRequest]slaveThreadNum[%u] notifyNumPerThread[%u]"
        " notifyNumOnMainThread[%u] level0Channels[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread.size(),
        resourceRequest.notifyNumOnMainThread, level0Channels.size());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[ReduceScatterVBIRSExecutor][KernelRun] starts.");

    CHK_RET(KernelRunLevel0(param, execMem));

    HCCL_INFO("reduce scatter V BIRS run success");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSExecutor::SelectAndPrepareBirsVTemplate(
    bool isSingleServer, u32 localRank, u32 localRankSize, std::unique_ptr<AlgTemplateBase>& templatePtr)
{
    TemplateType tplType = isSingleServer ? TEMPLATE_REDUCE_SCATTER_V_BIRS : TEMPLATE_REDUCE_SCATTER_V_BIRS_INTER;
    templatePtr = AlgTemplateRegistry::Instance().GetAlgTemplate(tplType);
    CHK_SMART_PTR_NULL(templatePtr);
    if (isSingleServer) {
        CHK_RET(templatePtr->Prepare(localRank, localRankSize));
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCE_SCATTER_V_BIRS in COMM_LEVEL0", __func__);
    } else {
        CHK_RET(templatePtr->Prepare(topoInfo_->serverNum, topoInfo_->userRankSize));
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCE_SCATTER_V_BIRS_INTER in COMM_LEVEL0", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSExecutor::KernelRunLevel0(const OpParam& param, ExecMem& execMem)
{
    SubCommInfo level0CommInfo;
    CHK_RET(GetSubCommInfo(COMM_LEVEL0, level0CommInfo));
    u32 level0LocalRank = level0CommInfo.localRank;
    u32 level0LocalRankSize = level0CommInfo.localRankSize;
    u32 sliceNum = level0LocalRankSize;

    // V变体: 从param.vDataDes.counts/displs还原各rank的sendCounts与sendDispls
    // 注意: 使用vDataDes(常规字段, sizeof(OpParam)内)而非varData(柔性数组), 因为AICPU_TS
    // kernel下发只拷贝sizeof(OpParam)
    const u64* sendCounts = reinterpret_cast<const u64*>(param.vDataDes.counts);
    const u64* sendDispls = reinterpret_cast<const u64*>(param.vDataDes.displs);

    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareDataSliceV(sendCounts, sendDispls, unitSize_, sliceNum, dataSegsSlice));

    std::unique_ptr<AlgTemplateBase> level0TempAlg = std::make_unique<AlgTemplateBaseExperimental>();
    bool isSingleServer = (topoInfo_->serverNum == 1);
    CHK_RET(SelectAndPrepareBirsVTemplate(isSingleServer, level0LocalRank, level0LocalRankSize, level0TempAlg));

    // V变体: 用户输入区大小为param.inputSize(各rank按sendDispls布局), 输出区为param.outputSize(本rank接收量)
    HcclMem UsrInputMem{HCCL_MEM_TYPE_DEVICE, execMem.inputPtr, param.inputSize};
    HcclMem UsrOutputMem{HCCL_MEM_TYPE_DEVICE, execMem.outputPtr, param.outputSize};

    if (auto exp = dynamic_cast<AlgTemplateBaseExperimental*>(level0TempAlg.get())) {
        // count传本rank接收量(execMem.count), slices传变长切片, 由模板内部按slices_[i].size处理各rank数据量
        CHK_RET(exp->Prepare(
            UsrInputMem, UsrOutputMem, execMem.inputMem, execMem.count, param.vDataDes.dataType, thread_, slaveThreads_,
            param.reduceType, 0, dataSegsSlice, 0, false));

        CHK_RET(exp->RunAsync(level0LocalRank, level0LocalRankSize, channels_[COMM_LEVEL0]));
    } else {
        HCCL_ERROR("[KernelRunLevel0] dynamic_cast to AlgTemplateBaseExperimental failed");
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterVBIRSExecutor", ReduceScatterVBIRS, ReduceScatterVBIRSExecutor);
} // namespace ops_hccl_experimental
