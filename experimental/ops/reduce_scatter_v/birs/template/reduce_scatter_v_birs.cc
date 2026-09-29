/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "reduce_scatter_v_birs.h"
#include "reduce_scatter_v_executor_base.h"
#include <algorithm>

namespace ops_hccl_experimental {
ReduceScatterVBIRS::ReduceScatterVBIRS() : AlgTemplateBaseExperimental() {}

ReduceScatterVBIRS::~ReduceScatterVBIRS() {}

HcclResult ReduceScatterVBIRS::Prepare(u32 interRank, u32 interRankSize)
{
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRS::Prepare(
    HcclMem& inputMem, HcclMem& outputMem, HcclMem& scratchMem, const u64 count, const HcclDataType dataType,
    ThreadHandle thread, const std::vector<ThreadHandle>& slaveThreads, const HcclReduceOp reductionOp, const u32 root,
    const std::vector<Slice>& slices, const u64 baseOffset, const bool disableDMAReduce)
{
    mainThread = thread;
    subThreads = slaveThreads;
    AlgTemplateBase::Prepare(
        inputMem, outputMem, scratchMem, count, dataType, thread, reductionOp, root, slices, baseOffset,
        disableDMAReduce);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRS::LocalReduceCCLToCCL(u64 srcOffset, u64 dstOffset, u64 size, ThreadHandle thread)
{
    void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + srcOffset);
    void* dstSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + dstOffset);
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        thread, dstSlice, srcSlice, size / unitSize, static_cast<HcommDataType>(dataType_),
        static_cast<HcommReduceOp>(reductionOp_))));
    return HCCL_SUCCESS;
}

void ReduceScatterVBIRS::CalcLocalStrideSize(const u32 rankSize)
{
    u64 maxSliceSize = 0;
    for (u32 i = 0; i < rankSize; i++) {
        maxSliceSize = std::max(maxSliceSize, slices_[i].size);
    }
    localStrideSize = RoundUpWithDivisor(maxSliceSize, HCCL_MIN_SLICE_ALIGN_910B);
}

HcclResult ReduceScatterVBIRS::CheckScratchCapacity(u32 rankSize)
{
    u64 scratchDemand = static_cast<u64>(rankSize) * localStrideSize;
    CHK_PRT_RET(
        scratchDemand > scratchMem_.size,
        HCCL_ERROR(
            "[ReduceScatterVBIRS][CheckScratchCapacity]scratch demand[%llu] exceeds scratchMem size[%llu]",
            scratchDemand, scratchMem_.size),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

void ReduceScatterVBIRS::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = BIRS_THREAD_NUM;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void ReduceScatterVBIRS::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = BIRS_THREAD_NUM;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult ReduceScatterVBIRS::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize)
{
    // V变体: 切片由executor按sendCounts/sendDispls提前构造并经Prepare传入, 此处仅作兜底(均匀)
    slices_.resize(rankSize);
    u64 sliceSize = totalCount * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG(" default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRS::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    if (rankSize == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            // V变体: 本rank接收量为 slices_[rank].size
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                thread_, outputMem_.addr, static_cast<u8*>(inputMem_.addr) + slices_[rank].offset,
                slices_[rank].size)));
        }
        return HCCL_SUCCESS;
    }

    if (channels.size() < rankSize) {
        HCCL_ERROR(
            "[ReduceScatterVBIRS][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, channels.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[ReduceScatterVBIRS][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        CHK_RET(PrepareSlicesData(unitSize, count_, rankSize));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRS::HCCSProcessMainLoop(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize)
{
    if (round != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 localOffsetByte = hccs_ranks[round - 1] / rankSizeX_ * localStrideSize;
        u64 remoteOffsetByte = ((rankSize / rankSizeX_) + rank / rankSizeX_) * localStrideSize;
        void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void*>(static_cast<u8*>(hccs_links[round - 1].remoteOutput.addr) + remoteOffsetByte);

        // V变体: 传输量为目标rank(hccs_ranks[round-1])对应的变长切片大小
        u64 curSliceSize = slices_[hccs_ranks[round - 1]].size;
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(subThreads[0], hccs_links[round - 1].handle, dst, src, curSliceSize)));

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRS::SIOProcessMainLoop(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

    // V变体: 本轮处理的target rank及其变长切片
    u32 targetRank = (round != hccs_ranks.size()) ? hccs_neighbour_rank[round] : sio_rank;
    u64 localOffsetByte = slices_[targetRank].offset;
    u64 remoteOffsetByte = (round != hccs_ranks.size()) ? (hccs_ranks[round] / rankSizeX_ * localStrideSize) :
                                                          rank / rankSizeX_ * localStrideSize;
    void* src = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + localOffsetByte);
    void* dst = static_cast<void*>(static_cast<u8*>(sio_link.remoteOutput.addr) + remoteOffsetByte);

    u64 curSliceSize = slices_[targetRank].size;
    CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(
        mainThread, sio_link.handle, dst, src, curSliceSize / unitSize, static_cast<HcommDataType>(dataType_),
        static_cast<HcommReduceOp>(reductionOp_))));

    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRS::LocalCopyMainLoop(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize)
{
    if (round < hccs_ranks.size()) {
        u32 rank_idx = (round < hccs_ranks.size() - 1) ? hccs_ranks[round + 1] : rank;
        void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + slices_[rank_idx].offset);
        void* dstSlice
            = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + rank_idx / rankSizeX_ * localStrideSize);
        // V变体: 拷贝量为rank_idx对应的变长切片大小
        u64 curSliceSize = slices_[rank_idx].size;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, curSliceSize)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRS::FinalStep(const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize)
{
    // V变体: 本rank最终接收量为 slices_[rank].size
    u64 localSliceSize = slices_[rank].size;
    std::vector<u32> vec;
    for (u32 i = 0; i < (rankSize / rankSizeX_); i++) {
        if (i == (rank / rankSizeX_)) {
            vec.push_back((rank / rankSizeX_) * localStrideSize);
        } else {
            vec.push_back(((rankSize / rankSizeX_) + i) * localStrideSize);
        }
    }
    // Tree local reduce
    auto ind = rankSize / rankSizeX_;
    for (u32 stride = 1; stride < ind; stride *= 2) {
        for (u32 i = stride; i < ind; i += stride * 2) {
            CHK_RET(LocalReduceCCLToCCL(vec[i], vec[i - stride], localSliceSize, mainThread));
        }
    }
    // Local copy to output
    void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + vec[0]);
    void* dstSlice = static_cast<void*>(static_cast<u8*>(outputMem_.addr));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, localSliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRS::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    HCCL_INFO(
        "ReduceScatterVBIRS run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.addr, outputMem_.addr, count_);

    hccs_ranks.clear();
    hccs_neighbour_rank.clear();
    hccs_links.clear();

    CHK_RET(Preprocess(rank, rankSize, channels));

    if (rankSize % rankSizeX_ != 0) {
        HCCL_ERROR(
            "[ReduceScatterVBIRS][RunAsync]rankSize[%u] is not evenly divisible by rankSizeX_[%u]", rankSize,
            rankSizeX_);
        return HCCL_E_INTERNAL;
    }

    sio_rank = rank ^ 1;
    sio_link = channels[sio_rank];

    for (u32 i = 1; i < rankSize / rankSizeX_; ++i) {
        u32 current_hccs_rank = (rank + rankSizeX_ * i) % rankSize;
        hccs_ranks.push_back(current_hccs_rank);
        hccs_neighbour_rank.push_back(current_hccs_rank ^ 1);
        hccs_links.push_back(channels[hccs_ranks[i - 1]]);
    }
    hccs_links_reversed.assign(hccs_links.rbegin(), hccs_links.rend());

    CalcLocalStrideSize(rankSize);

    CHK_RET(CheckScratchCapacity(rankSize));

    // MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

    // V变体: 初始拷贝量取 hccs_ranks[0] 对应的变长切片
    if (!hccs_ranks.empty()) {
        u64 initSliceSize = slices_[hccs_ranks[0]].size;
        void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + slices_[hccs_ranks[0]].offset);
        void* dstSlice
            = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + hccs_ranks[0] / rankSizeX_ * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, initSliceSize)));
    }

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        // MainRecordSub + SubWaitMain
        CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

        CHK_RET(HCCSProcessMainLoop(round, rank, rankSize, rankSizeX_, localStrideSize));

        CHK_RET(SIOProcessMainLoop(round, rank, rankSize, rankSizeX_, localStrideSize));

        CHK_RET(LocalCopyMainLoop(round, rank, rankSize, rankSizeX_, localStrideSize));

        // SubRecordMain + MainWaitSub
        CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    }

    // MainRecordSub + SubWaitMain
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

    CHK_RET(FinalStep(rank, rankSize, rankSizeX_, localStrideSize));

    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));

    HCCL_INFO("ReduceScatterVBIRS finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TEMPLATE_REDUCE_SCATTER_V_BIRS, ReduceScatterVBIRS);
} // namespace ops_hccl_experimental
