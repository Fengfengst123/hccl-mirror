/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_mesh_template.h"
#include "template_factory.h"

#include "channel.h"
#include "data_transfer.h"
#include "comm_planners/mesh_comm_planner.h"
#include "data_ops.h"

#include "log.h"

namespace ops_hccl {

REGISTER_RE_TEMPLATE(HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, AllGatherMeshTemplate)

HcclResult AllGatherMeshTemplate::DoCalcChannelRequest(
    HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& levelChannels)
{
    return CalcChannelRequestMesh1D(comm, param, topoInfo, subcommInfo, levelChannels);
}

u32 AllGatherMeshTemplate::DoCalcThreadNum() const
{
    const u32 rankSize = static_cast<u32>(ranks_.size());
    u32 threadNum = (rankSize > 1) ? rankSize - 1 : 1;
    return threadNum * channelsPerRank_;
}

u32 AllGatherMeshTemplate::DoCalcNotifyPerThread() const { return 1; }

HcclResult
AllGatherMeshTemplate::RunAlgorithm(std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>& ranksForOutputData)
{
    HCCL_DEBUG("[AllGatherMeshTemplate][RunAlgorithm] start, myRank[%u], rankSize[%u].", myRank_, templateRankSize_);

    CHK_RET(RunMeshAllGather(tempAlgParams_, ranks_, myRank_, ranksForOutputData, txRxSlicesLists));

    HCCL_DEBUG("[AllGatherMeshTemplate][RunAlgorithm] end.");
    return HCCL_SUCCESS;
}

// output 为 userBuffer 时走 READ 方向
static inline bool DirectToOutputMode(const DataParams& params)
{
    return params.outputBufferType == BufferType::OUTPUT;
}

TransferContext AllGatherMeshTemplate::BuildTransferContext(
    const DataSlicesList& txRxSlicesList, TemplateResource& templateResource, bool isLastStep,
    bool parallelPostCopy) const
{
    (void)isLastStep;
    (void)parallelPostCopy;
    if (!DirectToOutputMode(tempAlgParams_)) {
        return AicpuBaseTemplate::BuildTransferContext(txRxSlicesList, templateResource, isLastStep, parallelPostCopy);
    }
    TransferContext ctx;
    // Mesh 场景为 server 内层通信，使用 UB（Unified Bus）链路，
    // 不涉及跨 server 的 PCIE/HD 链路，也无 OFFLOAD 模式约束，
    // 因此无条件使用远端读（READ 方向）直写 output。
    ctx.remoteReadEnabled = true;
    ctx.buffType = BufferType::OUTPUT;
    ctx.txRxSlicesList = txRxSlicesList;
    ctx.templateRes = &templateResource;
    ctx.dataType = tempAlgParams_.dataType;
    ctx.reduceOp = tempAlgParams_.reduceOp;
    return ctx;
}

HcclResult AllGatherMeshTemplate::PostCopy(const std::vector<ThreadHandle>& threads)
{
    // OUTPUT 模式：peer 数据已在 SendAll 中直写 output，这里仅补搬 myRank 的 ccl->output
    if (!DirectToOutputMode(tempAlgParams_)) {
        return AicpuBaseTemplate::PostCopy(threads);
    }

    HCCL_DEBUG("[AllGatherMeshTemplate][PostCopy] directToOutput mode, copy myRank only, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AllGatherMeshTemplate][PostCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    // directToOutput 模式下 peer 数据已在 SendAll 中直写 output，
    // 此处仅补搬 myRank 的 ccl->output。PostCopyData 以循环下标 idx 作为 output 槽位索引，
    // 需传入完整的 ranksForOutputData_ 使 idx 对齐正确槽位，并用 skipRanks 跳过非 myRank 的条目。
    std::vector<u32> skipRanks;
    for (u32 rank : ranksForOutputData_) {
        if (rank != myRank_) {
            skipRanks.push_back(rank);
        }
    }
    CHK_RET(PostCopyData(tempAlgParams_, threads[0], ranksForOutputData_, 0, ChannelSplitInfo{}, skipRanks));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
