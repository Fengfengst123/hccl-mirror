/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLGATHER_NHR_TEMPLATE_H
#define ALLGATHER_NHR_TEMPLATE_H

#include "aicpu_base_template.h"
#include "data_transfer.h"

namespace ops_hccl {

// AllGather NHR 模板
class AllGatherNhrTemplate : public AicpuBaseTemplate {
public:
    AllGatherNhrTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : AicpuBaseTemplate(myRank, std::move(ranks), std::move(templateDesc))
    {
        syncAtCopyBoundary_ = true;
    }
    ~AllGatherNhrTemplate() = default;

    // 判断末步 rx 是否直写 output（DMA 消减优化）。
    // 策略判断属 Template 层职责，从 nhr_comm_planner 迁入此处。
    static bool CanReadLastStepToOutput(const DataParams& tempAlgParams)
    {
        if (tempAlgParams.enableRemoteMemAccess || tempAlgParams.outputBufferType != BufferType::OUTPUT) {
            return false;
        }
        const DataSizeInfo sizeInfo = CalcDataSizeInfo(tempAlgParams);
        if (sizeInfo.sliceSize <= DMA_REDUCTION_MIN_DATA_SIZE) {
            return false;
        }
        return true;
    }

protected:
    HcclResult DoCalcChannelRequest(
        HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
        const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& levelChannels) override;
    u32 DoCalcThreadNum() const override;
    u32 DoCalcNotifyPerThread() const override;

    HcclResult
    RunAlgorithm(std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>& ranksForOutputData) override;
    HcclResult PreCopy(const std::vector<ThreadHandle>& threads) override;
    HcclResult PostCopy(const std::vector<ThreadHandle>& threads) override;
    HcclResult CopyInputToOutput(const std::vector<ThreadHandle>& threads) override;

    // SendAll hook 覆写
    TransferContext BuildTransferContext(
        const DataSlicesList& txRxSlicesList, TemplateResource& templateResource, bool isLastStep,
        bool parallelPostCopy) const override;
    bool CanParallelPostCopy(const TemplateResource& templateResource) const override;
    HcclResult LaunchPostCopy(const std::vector<ThreadHandle>& threads) override;

private:
    // DMA 消减阈值：sliceSize <= 此值时不启用末步直写 output 优化（比较 sliceSize）
    // 与 SINGLE_CHANNEL_MAX_DATA_SIZE_ 同值但用途不同，独立调优。
    static constexpr u64 DMA_REDUCTION_MIN_DATA_SIZE = 1 * 1024 * 1024; // 1MB
    // 小数据通道去重阈值：dataSize_ <= 此值时去重为单 channel（比较 dataSize_）
    // 与 DMA_REDUCTION_MIN_DATA_SIZE 同值但用途不同，独立调优。
    static constexpr u64 SINGLE_CHANNEL_MAX_DATA_SIZE_ = 1 * 1024 * 1024; // 1MB
    static void DedupChannelsByRemoteRank(std::vector<HcclChannelDesc>& channels);

    bool canParallelPostCopy_{false};
    bool postCopyLaunched_{false};
    std::vector<u32> lastStepRxRanks_{};
};

} // namespace ops_hccl

#endif // ALLGATHER_NHR_TEMPLATE_H
