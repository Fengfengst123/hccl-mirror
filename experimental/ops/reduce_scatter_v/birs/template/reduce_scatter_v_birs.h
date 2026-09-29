/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_V_BIRS_H
#define REDUCE_SCATTER_V_BIRS_H

#include "alg_template_base.h"
#include "alg_data_trans_wrapper.h"
#include "alg_template_base_experimental.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgTemplateBase;
using ops_hccl::AlgTemplateRegistry;
using ops_hccl::ChannelInfo;
using ops_hccl::CUSTOM_TIMEOUT;
using ops_hccl::DefaultTemplateCreator;
using ops_hccl::HCCL_MIN_SLICE_ALIGN_910B;
using ops_hccl::NOTIFY_IDX_ACK;
using ops_hccl::NOTIFY_IDX_DATA_SIGNAL;
using ops_hccl::PostSyncInterThreads;
using ops_hccl::PreSyncInterThreads;
using ops_hccl::RoundUpWithDivisor;
using ops_hccl::Slice;
using ops_hccl::TemplateType;

// ReduceScatterV 的 BIRS 算法模板(变长: 各rank接收量可不同)
// 与均匀 BIRS 的区别: slices_ 由外部传入变长切片(offset/size按sendDispls/sendCounts),
// 每轮传输使用目标rank对应的 slices_[rank].size, scratch统一以 maxSliceSize 向上对齐的 localStrideSize 为步长
// 前置条件: comm 内所有 rank 传入一致的 sendCounts/sendDispls 数组(集合通信语义约定, 与老流程一致)。
// 满足该条件时, SIO 链路两端针对同一目标 rank 的 slices_[target].size 必然相等,
// 每轮 write-reduce 两端长度一致, 对所有规约类型结果正确。
class ReduceScatterVBIRS : public AlgTemplateBaseExperimental {
public:
    explicit ReduceScatterVBIRS();

    ~ReduceScatterVBIRS() override;

    // should be called soon after template ReduceScatterVBIRS instance created
    HcclResult Prepare(u32 interRank, u32 interRankSize) override;

    HcclResult Prepare(
        HcclMem& inputMem, HcclMem& outputMem, HcclMem& scratchMem, const u64 count, const HcclDataType dataType,
        ThreadHandle thread, const std::vector<ThreadHandle>& slaveThreads, const HcclReduceOp reductionOp,
        const u32 root, const std::vector<Slice>& slices, const u64 baseOffset, const bool disableDMAReduce) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels) override;

protected:
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub);
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain);
    HcclResult LocalReduceCCLToCCL(u64 srcOffset, u64 dstOffset, u64 size, ThreadHandle thread);
    // V变体: scratch步长以各rank切片大小的最大值向上对齐, 保证任意rank数据放入一个步长槽
    virtual void CalcLocalStrideSize(const u32 rankSize);
    virtual HcclResult CheckScratchCapacity(u32 rankSize);

    virtual HcclResult Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels);
    HcclResult PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize);
    // 主循环各步骤: localStrideSize 为 scratch 统一步长, 各rank实际传输量取自 slices_[target].size
    HcclResult HCCSProcessMainLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult SIOProcessMainLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult LocalCopyMainLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult FinalStep(const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);

    u32 interRank_;     // comm内的rank排序
    u32 interRankSize_; // 本comm内ranksize总数
    u32 sio_rank;
    std::vector<u32> hccs_ranks;
    std::vector<u32> hccs_neighbour_rank;

    std::vector<ChannelInfo> hccs_links;
    std::vector<ChannelInfo> hccs_links_reversed;
    ChannelInfo sio_link;

    u64 localStrideSize;

    u32 unitSize;
    u32 rankSizeX_ = 2;
    ThreadHandle mainThread;
    std::vector<ThreadHandle> subThreads;
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;

private:
};

} // namespace ops_hccl_experimental
#endif /* * REDUCE_SCATTER_V_BIRS_H */
