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
#include "reduce_scatter_v_birs_inter.h"
#include <algorithm>

namespace ops_hccl_experimental {

ReduceScatterVBIRSInter::ReduceScatterVBIRSInter() : ReduceScatterVBIRS() {}

ReduceScatterVBIRSInter::~ReduceScatterVBIRSInter() {}

HcclResult ReduceScatterVBIRSInter::Prepare(u32 serverNum, u32 interRankSize)
{
    serverNum_ = serverNum;
    CHK_PRT_RET(
        serverNum == 0 || interRankSize % serverNum != 0,
        HCCL_ERROR("[Prepare] interRankSize[%u] not divisible by serverNum[%u]", interRankSize, serverNum),
        HCCL_E_PARA);
    intraRankSize_ = interRankSize / serverNum_;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSInter::LocalCopyPreproc(ThreadHandle& thread, const u32 rank, u64 localStrideSize)
{
    void* src;
    void* dst;
    u32 cnt = 0;
    // V变体: 从input按各rank的slices_[rank].offset/size搬运到scratch(步长为localStrideSize)
    for (u32 round = 0; round < intraRankSize_ / 2; round++) {
        for (u32 i = 0; i < serverNum_; i++) {
            u32 rankIdx = 2 * round + i * intraRankSize_ + rank % 2;
            if (slices_[rankIdx].size != 0) {
                src = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + slices_[rankIdx].offset);
                dst = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + cnt * localStrideSize);
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, slices_[rankIdx].size)));
            }
            cnt++;
        }
    }

    for (u32 c = 0; c < serverNum_; c++) {
        u32 rankIdx = hccs_neighbour_rank[0] % intraRankSize_ + c * intraRankSize_;
        if (slices_[rankIdx].size == 0) {
            continue;
        }
        void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + slices_[rankIdx].offset);
        void* dstSlice = static_cast<void*>(
            static_cast<u8*>(scratchMem_.addr) + (intraRankSize_)*localStrideSize * serverNum_ + c * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dstSlice, srcSlice, slices_[rankIdx].size)));
    }

    return HCCL_SUCCESS;
}

void ReduceScatterVBIRSInter::CalcLocalStrideSize(const u32 rankSize)
{
    u64 maxSliceSize = 0;
    for (u32 i = 0; i < rankSize; i++) {
        maxSliceSize = std::max(maxSliceSize, slices_[i].size);
    }
    // 保持不对齐: 与均匀 BIRS inter 一致(reduce_scatter_birs_inter.cc RunAsync 中 localStrideSize = sliceSize
    // 同样未做对齐, 已验证可用)。HCCS WRITE 按字节、SIO write-reduce 按元素搬运, 长度均为 unitSize 整数倍,
    // 无额外对齐约束; 若按 HCCL_MIN_SLICE_ALIGN_910B(16KB) 对齐会成倍放大 scratch 需求,
    // 导致小消息场景 CheckScratchCapacity 误报容量错误
    localStrideSize = maxSliceSize;
}

HcclResult ReduceScatterVBIRSInter::CheckScratchCapacity(u32 rankSize)
{
    u64 scratchDemand = (static_cast<u64>(rankSize) + static_cast<u64>(2) * serverNum_) * localStrideSize;
    CHK_PRT_RET(
        scratchDemand > scratchMem_.size,
        HCCL_ERROR(
            "[ReduceScatterVBIRSInter][CheckScratchCapacity]scratch demand[%llu] exceeds scratchMem size[%llu]",
            scratchDemand, scratchMem_.size),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSInter::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    HCCL_INFO(
        "ReduceScatterVBIRSInter run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.addr, outputMem_.addr, count_);
    return ReduceScatterVBIRS::Preprocess(rank, rankSize, channels);
}

HcclResult ReduceScatterVBIRSInter::HCCSIntraStep(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize)
{
    // V变体: 按槽位精确WRITE(每槽只搬运 slices_[target].size, padding 区不参与传输),
    // 槽位 cnt 对应的目标 rank 为 hccs_ranks[round-1]%intra + cnt*intra
    if (round != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 localOffsetByte = hccs_ranks[round - 1] % intraRankSize_ / rankSizeX_ * localStrideSize * serverNum_;
        u64 remoteOffsetByte
            = ((intraRankSize_ / rankSizeX_) + rank % intraRankSize_ / rankSizeX_) * localStrideSize * serverNum_;
        u32 targetBase = hccs_ranks[round - 1] % intraRankSize_;
        for (u32 cnt = 0; cnt < serverNum_; cnt++) {
            u64 sliceSize = slices_[targetBase + cnt * intraRankSize_].size;
            if (sliceSize == 0) {
                continue;
            }
            void* src
                = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte + cnt * localStrideSize);
            void* dst = static_cast<void*>(
                static_cast<u8*>(hccs_links[round - 1].remoteOutput.addr) + remoteOffsetByte + cnt * localStrideSize);

            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(subThreads[0], hccs_links[round - 1].handle, dst, src, sliceSize)));
        }

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSInter::SIOIntraStep(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize)
{
    // V变体: 按槽位精确write-reduce(每槽只规约 slices_[target].size, padding 区不参与规约,
    // 避免规约未初始化数据); 槽位 cnt 对应的目标 rank 为 targetRank%intra + cnt*intra,
    // 链路两端对同一目标 rank 使用相同切片长度(sendCounts/sendDispls 各rank一致)
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

    u64 localOffsetByte = intraRankSize_ + (round % 2);
    u64 remoteOffsetByte = (round != hccs_ranks.size()) ? ((hccs_ranks[round] % intraRankSize_) / rankSizeX_) :
                                                          (rank % intraRankSize_ / rankSizeX_);
    localOffsetByte *= localStrideSize * serverNum_;
    remoteOffsetByte *= localStrideSize * serverNum_;
    u32 targetRank = (round != hccs_ranks.size()) ? hccs_neighbour_rank[round] : sio_rank;
    for (u32 cnt = 0; cnt < serverNum_; cnt++) {
        u64 sliceSize = slices_[targetRank % intraRankSize_ + cnt * intraRankSize_].size;
        if (sliceSize == 0) {
            continue;
        }
        void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte + cnt * localStrideSize);
        void* dst = static_cast<void*>(
            static_cast<u8*>(sio_link.remoteOutput.addr) + remoteOffsetByte + cnt * localStrideSize);

        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(
            mainThread, sio_link.handle, dst, src, sliceSize / unitSize, static_cast<HcommDataType>(dataType_),
            static_cast<HcommReduceOp>(reductionOp_))));
    }

    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSInter::LocalCopyIntraStep(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize)
{
    // V变体: 从input按各rank的slices_[rank].offset/size搬运到scratch(下一轮SIO用的ping-pong区)
    if (round < hccs_ranks.size()) {
        for (u32 cnt = 0; cnt < serverNum_; cnt++) {
            u32 rankIdx = (round < hccs_ranks.size() - 1) ?
                              ((hccs_neighbour_rank[round + 1]) % intraRankSize_ + cnt * intraRankSize_) :
                              ((sio_rank) % intraRankSize_ + cnt * intraRankSize_);
            if (slices_[rankIdx].size == 0) {
                continue;
            }
            u64 srcOffsetByte = slices_[rankIdx].offset;
            u64 dstOffsetByte
                = (intraRankSize_ + ((round + 1) % 2)) * localStrideSize * serverNum_ + cnt * localStrideSize;

            void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + srcOffsetByte);
            void* dstSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + dstOffsetByte);
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, slices_[rankIdx].size)));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSInter::PreprocInterServer(
    const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize, std::vector<ChannelInfo>& channels)
{
    // V变体: 跨服务器前置阶段, 本rank最终接收量为 slices_[rank].size
    u64 localSliceSize = slices_[rank].size;
    vec_offsets.clear();
    for (u32 i = 0; i < (intraRankSize_ / rankSizeX_); i++) {
        if (i == (rank % intraRankSize_ / rankSizeX_)) {
            vec_offsets.push_back((rank % intraRankSize_ / rankSizeX_) * localStrideSize * serverNum_);
        } else {
            vec_offsets.push_back(((intraRankSize_ / rankSizeX_) + i) * localStrideSize * serverNum_);
        }
    }

    // V变体: 树形规约按槽位精确执行(每槽只规约 slices_[target].size, padding 区不参与规约);
    // 各块槽位 cnt 对应的目标 rank 均为 rank%intra + cnt*intra, 切片长度跨块一致
    auto ind = intraRankSize_ / rankSizeX_;
    for (u32 stride = 1; stride < ind; stride *= 2) {
        for (u32 i = stride; i < ind; i += stride * 2) {
            for (u32 cnt = 0; cnt < serverNum_; cnt++) {
                u64 sliceSize = slices_[rank % intraRankSize_ + cnt * intraRankSize_].size;
                if (sliceSize == 0) {
                    continue;
                }
                CHK_RET(LocalReduceCCLToCCL(
                    vec_offsets[i] + cnt * localStrideSize, vec_offsets[i - stride] + cnt * localStrideSize, sliceSize,
                    mainThread));
            }
        }
    }

    u64 remoteOffsetByte = localStrideSize * serverNum_;
    u64 localOffsetByte = vec_offsets[0] + localStrideSize * (((rank) % rankSize) / intraRankSize_);
    void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
    void* dst = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + remoteOffsetByte);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dst, src, localSliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVBIRSInter::InterServer(
    const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize, std::vector<ChannelInfo>& channels)
{
    // V变体: 跨服务器阶段, 本rank最终接收量为 slices_[rank].size
    u64 localSliceSize = slices_[rank].size;
    for (u32 round = 1; round < serverNum_; round++) {
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            mainThread, channels[((serverNum_ - round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            mainThread, channels[((round)*intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 remoteOffsetByte = localStrideSize * serverNum_ + localStrideSize * round;
        // V变体: 本轮向 peerRank(同intra位置、其他server)转发其目标切片的本server内规约和,
        // 传输长度必须取 slices_[peerRank].size(目标rank的切片长度)而非本rank的 localSliceSize
        u32 peerRank = ((round)*intraRankSize_ + rank) % rankSize;
        u64 peerSliceSize = slices_[peerRank].size;
        u64 localOffsetByte = vec_offsets[0] + localStrideSize * (peerRank / intraRankSize_);
        void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void*>(static_cast<u8*>(channels[peerRank].remoteOutput.addr) + remoteOffsetByte);
        if (peerSliceSize != 0) {
            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(mainThread, channels[peerRank].handle, dst, src, peerSliceSize)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            mainThread, channels[((round)*intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            mainThread, channels[((serverNum_ - round) * intraRankSize_ + rank) % rankSize].handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    }
    u64 ptrSz = (serverNum_ == 1 ? vec_offsets[0] : localStrideSize * serverNum_);
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));
    for (u32 step = 1; step < serverNum_; step *= 2) {
        for (u32 i = 0; i + step < serverNum_; i += 2 * step) {
            CHK_RET(LocalReduceCCLToCCL(
                ptrSz + localStrideSize * (i + step), ptrSz + localStrideSize * i, localSliceSize, mainThread));
        }
    }
    void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + ptrSz);
    void* dstSlice = static_cast<void*>(static_cast<u8*>(outputMem_.addr));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, localSliceSize)));
    return HCCL_SUCCESS;
}

// scatter的入口函数
HcclResult ReduceScatterVBIRSInter::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    hccs_ranks.clear();
    hccs_neighbour_rank.clear();
    hccs_links.clear();

    CHK_RET(Preprocess(rank, rankSize, channels));

    if (intraRankSize_ % rankSizeX_ != 0) {
        HCCL_ERROR(
            "[ReduceScatterVBIRSInter][RunAsync]intraRankSize_[%u] is not evenly divisible by rankSizeX_[%u]",
            intraRankSize_, rankSizeX_);
        return HCCL_E_INTERNAL;
    }

    sio_rank = rank ^ 1;
    sio_link = channels[sio_rank];

    for (u32 i = 1; i < intraRankSize_ / rankSizeX_; ++i) {
        u32 current_hccs_rank
            = (rank % intraRankSize_ + rankSizeX_ * i) % (intraRankSize_) + rank / intraRankSize_ * intraRankSize_;
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

    if (!hccs_neighbour_rank.empty()) {
        CHK_RET(LocalCopyPreproc(mainThread, rank, localStrideSize));
    }

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        // MainRecordSub + SubWaitMain
        CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

        CHK_RET(HCCSIntraStep(round, rank, rankSize, rankSizeX_, localStrideSize));
        CHK_RET(SIOIntraStep(round, rank, rankSize, rankSizeX_, localStrideSize));
        CHK_RET(LocalCopyIntraStep(round, rank, rankSize, rankSizeX_, localStrideSize));

        // SubRecordMain + MainWaitSub
        CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    }

    // MainRecordSub + SubWaitMain
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

    CHK_RET(PreprocInterServer(rank, rankSize, rankSizeX_, localStrideSize, channels));

    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));

    CHK_RET(InterServer(rank, rankSize, rankSizeX_, localStrideSize, channels));

    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    HCCL_INFO("ReduceScatterVBIRSInter finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TEMPLATE_REDUCE_SCATTER_V_BIRS_INTER, ReduceScatterVBIRSInter);
} // namespace ops_hccl_experimental
