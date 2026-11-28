/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "hccl_aiv_utils.h"
#include "aiv/aiv_temp_reduce_scatter_mesh_1D.h"
#include "config_log.h"
#include "cost_model.h"

namespace ops_hccl {

AivTempReduceScatterMesh1D::AivTempReduceScatterMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

AivTempReduceScatterMesh1D::~AivTempReduceScatterMesh1D() {}

bool IsReduceScatterAivWriteEnabled(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo)
{
    // PCIe机型下的单算子场景走写语义，hif8沿用原读语义路径。
    // 严格保序场景同样走写语义：两条写路径的归约顺序均按rank固定的树形折叠，与数据到达顺序无关，
    // 且LocalTreeWrite直写前有逐对端presync、BigDataWrite为非pingpong并以BarrierAll收尾，
    // 跨算子buffer交接安全，满足确定性bit一致要求。
    return IsAivWriteEnabled(param, topoInfo) && param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_HIF8;
}

std::vector<CostModelParam> AivTempReduceScatterMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = 1;
    if (!param.portNum.empty()) {
        portNum = (param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    }
    int kernelNum = 1;
    int taskNum = 5 * (param.rankSize - 1);
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    float B1 = 0.0f;
    float B2 = 0.0f;
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B1);
    } else {
        B1 = 0.0f;
    }
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio, EngineType::AICPU, B2);
    B = B1 + (param.rankSize - 1) * B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AIV, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AIV, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

u64 AivTempReduceScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 2 * tempRankSize_;
    return scratchMultiple;
}

HcclResult AivTempReduceScatterMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMeshClosMultiJetty(comm, param, topoInfo, subCommRanks_, myChannelDescs, true));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UB_MEM) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[AivTempReduceScatterMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("[AivTempReduceScatterMesh1D] Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempReduceScatterMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    numBlocks = numBlocksLimit;
    if (dataSize < REDUCE_SCATTER_SMALL_COUNT_512KB) { // 小数据量，走原来极致低时延的流程
        constexpr uint32_t stepNum = 2;
        if (numBlocks > stepNum * tempRankSize_) {
            numBlocks = stepNum * tempRankSize_;
        }
    }
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[AivTempReduceScatterMesh1D] Actually use core num[%u], limit[%u]", numBlocks, numBlocksLimit);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempReduceScatterMesh1D::CalNumBlocksWrite(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    constexpr u32 TWO_RANK_SIZE = 2;
    constexpr u32 SMALL_RANK_SIZE = 4;
    constexpr u32 MEDIUM_RANK_SIZE = 8;
    constexpr u32 SMALL_CORE_LIMIT = 12;
    constexpr u32 MEDIUM_CORE_LIMIT = 24;
    constexpr u32 LARGE_CORE_LIMIT = 48;
    // 2卡不设额外核数上限；其余按12/24/48三档控核，超过8卡以48核封顶。
    u32 rankCoreLimit = LARGE_CORE_LIMIT;
    if (tempRankSize_ == TWO_RANK_SIZE) {
        rankCoreLimit = numBlocksLimit;
    } else if (tempRankSize_ <= SMALL_RANK_SIZE) {
        rankCoreLimit = SMALL_CORE_LIMIT;
    } else if (tempRankSize_ <= MEDIUM_RANK_SIZE) {
        rankCoreLimit = MEDIUM_CORE_LIMIT;
    }
    numBlocks = std::min(numBlocksLimit, rankCoreLimit);
    // 小数据沿用LocalTreeWrite路径，避免固定多核进入大数据kernel及其全局屏障。
    if (dataSize < REDUCE_SCATTER_SMALL_COUNT_512KB) {
        constexpr u32 SMALL_DATA_CORE_MULTIPLE = 2;
        numBlocks = std::min(numBlocks, SMALL_DATA_CORE_MULTIPLE * tempRankSize_);
    }
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[AivTempReduceScatterMesh1D] Write semantics use core num[%u], limit[%u], rank limit[%u]", numBlocks,
        numBlocksLimit, rankCoreLimit);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempReduceScatterMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempReduceScatterMesh1D] KernelRun start");

    IncSliceId(); // 自动增长sliceId，传入sliceId
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivReduceScatterArgs;
    aivReduceScatterArgs.cmdType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    // 核入口统一为aiv_reduce_scatter_*，读写语义由运行期isWrite分流，funcKey不随语义变化。
    aivReduceScatterArgs.isWrite = tempAlgParams.isWrite;
    aivReduceScatterArgs.input
        = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivReduceScatterArgs.output
        = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivReduceScatterArgs.inBuffType = tempAlgParams.buffInfo.inBuffType;
    aivReduceScatterArgs.outBuffType = tempAlgParams.buffInfo.outBuffType;
    aivReduceScatterArgs.rank = u32(myRank_);
    aivReduceScatterArgs.rankSize = tempRankSize_;
    aivReduceScatterArgs.count = tempAlgParams.sliceSize / HCCL_SIZE_TABLE[dataType_];
    aivReduceScatterArgs.dataType = dataType_;
    aivReduceScatterArgs.op = param.reduceType;
    aivReduceScatterArgs.root = root_;
    aivReduceScatterArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivReduceScatterArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivReduceScatterArgs.stream = param.stream;
    aivReduceScatterArgs.hcclComm = static_cast<HcclComm>(param.hcclComm);
    aivReduceScatterArgs.isOpBase = (param.opMode == OpMode::OPBASE);

    u64 dataSize = tempAlgParams.inputSliceStride;
    if (tempAlgParams.isWrite) {
        CHK_RET(CalNumBlocksWrite(aivReduceScatterArgs.numBlocks, dataSize, param.numBlocksLimit));
    } else {
        CHK_RET(CalNumBlocks(aivReduceScatterArgs.numBlocks, dataSize, param.numBlocksLimit));
    }

    aivReduceScatterArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivReduceScatterArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivReduceScatterArgs.repeatNum = tempAlgParams.repeatNum;
    aivReduceScatterArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivReduceScatterArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivReduceScatterArgs));

    HCCL_INFO("[AivTempReduceScatterMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
