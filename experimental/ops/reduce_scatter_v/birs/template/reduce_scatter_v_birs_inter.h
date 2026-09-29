/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_V_BIRS_INTER_H
#define REDUCE_SCATTER_V_BIRS_INTER_H

#include "reduce_scatter_v_birs.h"
#include "alg_template_base_experimental.h"

namespace ops_hccl_experimental {

// ReduceScatterV 的 BIRS 跨服务器(Inter-Server)算法模板
// 注意: scratch 以 localStrideSize*serverNum_ 为块、localStrideSize(各rank切片大小最大值, 不对齐)为槽位步长组织;
//       HCCS WRITE / SIO write-reduce / 树形 local-reduce 均按槽位以实际切片长度(slices_[target].size)精确执行,
//       padding 区不参与任何传输与规约(避免规约未初始化数据导致的不确定结果), 对各规约类型均正确。
//       前置条件与 ReduceScatterVBIRS 相同: 所有 rank 传入一致的 sendCounts/sendDispls 数组,
//       保证链路两端及各块间同一槽位的切片长度一致。当各rank接收量一致(均匀)时与原BIRS数据流等价。
class ReduceScatterVBIRSInter : public ReduceScatterVBIRS {
public:
    explicit ReduceScatterVBIRSInter();

    ~ReduceScatterVBIRSInter() override;

    HcclResult Prepare(u32 serverNum, u32 interRankSize) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels) override;

protected:
    HcclResult LocalCopyPreproc(ThreadHandle& stream, const u32 rank, u64 localStrideSize);
    // V变体: inter 不对齐步长, 保持批量传输结构(各rank接收量一致时与原inter等价)
    void CalcLocalStrideSize(const u32 rankSize) override;
    HcclResult CheckScratchCapacity(u32 rankSize) override;

    HcclResult Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels) override;
    HcclResult HCCSIntraStep(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult SIOIntraStep(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult LocalCopyIntraStep(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult PreprocInterServer(
        const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize, std::vector<ChannelInfo>& channels);
    HcclResult InterServer(
        const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize, std::vector<ChannelInfo>& channels);

private:
    u32 serverNum_;
    u32 intraRankSize_;
    u32 rankSizeX_ = 2;
    std::vector<u32> vec_offsets;
    u64 localStrideSize;
};

} // namespace ops_hccl_experimental
#endif /* * REDUCE_SCATTER_V_BIRS_INTER_H */
