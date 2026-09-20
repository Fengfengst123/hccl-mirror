/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_transfer.h"

#include "alg_param.h"
#include "alg_data_trans_wrapper.h"
#include "data_ops.h"
#include "log.h"
#include "hcomm_primitives_dl.h"
#include <map>
#include <vector>

namespace ops_hccl {

// 传输方向契约：
// WRITE 方向：发送方按 tx dst 切片写入对端 remoteCclMem，接收方 rx dst 切片被忽略（纯 ACK 握手）。
// READ 方向：接收方按 rx src/dst 切片从远端 ccl 拉取，发送方 tx 切片被忽略（纯握手）。

// 将 addr_ 为 nullptr 的 slice 修正为 fallbackBase（远端 CCL 内存地址）。
// 使用输出参数避免热路径上的堆分配。
static void FixupSliceAddrs(const std::vector<DataSlice>& slices, void* fallbackBase, std::vector<DataSlice>& result)
{
    result.clear();
    result.reserve(slices.size());
    for (const auto& s : slices) {
        void* addr = (s.addr_ == nullptr) ? fallbackBase : s.addr_;
        result.emplace_back(addr, s.offset_, s.size_, s.count_);
    }
}

// 按 portGroupSize 比例将 SlicesList 切分到指定 channelIdx。
// 对每个 DataSlice 独立计算切分（支持正常块/尾块不同 size）。
static SlicesList
SplitSlicesList(const SlicesList& orig, u32 channelIdx, const std::vector<ChannelInfo>& channels, u32 dataTypeSize)
{
    SlicesList result({}, {});
    // 按 size 缓存切分结果，避免重复计算
    std::map<u64, std::pair<std::vector<u64>, std::vector<u64>>> cache;

    auto getSplit = [&](u64 size) -> const std::pair<std::vector<u64>, std::vector<u64>>& {
        auto it = cache.find(size);
        if (it == cache.end()) {
            std::vector<u64> split, offset;
            CalcDataSplitByPortGroup(size, dataTypeSize, channels, split, offset);
            it = cache.emplace(size, std::make_pair(std::move(split), std::move(offset))).first;
        }
        return it->second;
    };

    for (const auto& slice : orig.srcSlices_) {
        const auto& p = getSplit(slice.size_);
        if (channelIdx < p.first.size() && p.first[channelIdx] > 0) {
            result.srcSlices_.emplace_back(
                slice.addr_, slice.offset_ + p.second[channelIdx], p.first[channelIdx],
                p.first[channelIdx] / dataTypeSize);
        }
    }
    for (const auto& slice : orig.dstSlices_) {
        const auto& p = getSplit(slice.size_);
        if (channelIdx < p.first.size() && p.first[channelIdx] > 0) {
            result.dstSlices_.emplace_back(
                slice.addr_, slice.offset_ + p.second[channelIdx], p.first[channelIdx],
                p.first[channelIdx] / dataTypeSize);
        }
    }
    return result;
}

// 修复远端地址后委托 src batch 函数
static HcclResult SendWrite(const DataInfo& sendInfo, const ThreadHandle& thread, HcclReduceOp reduceOp)
{
    void* remoteBase = sendInfo.channel_.remoteCclMem.addr;
    std::vector<DataSlice> fixedDst;
    FixupSliceAddrs(sendInfo.slices_.dstSlices_, remoteBase, fixedDst);
    SlicesList fixedSlices(sendInfo.slices_.srcSlices_, std::move(fixedDst));
    if (reduceOp == HCCL_REDUCE_RESERVED) {
        DataInfo info(sendInfo.channel_, fixedSlices, sendInfo.dataType_);
        return SendBatchWrite(info, thread);
    }
    DataReduceInfo info(sendInfo.channel_, fixedSlices, sendInfo.dataType_, reduceOp);
    return SendBatchWriteReduce(info, thread);
}

static HcclResult SendRecvWrite(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread, HcclReduceOp reduceOp)
{
    void* remoteBase = sendRecvInfo.sendRecvChannels_.txChannel_.remoteCclMem.addr;
    std::vector<DataSlice> fixedDst;
    FixupSliceAddrs(sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_, remoteBase, fixedDst);
    SlicesList fixedTx(sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_, std::move(fixedDst));
    TxRxSlicesList fixedSlices(fixedTx, sendRecvInfo.sendRecvSlices_.rxSlicesList_);
    if (reduceOp == HCCL_REDUCE_RESERVED) {
        SendRecvInfo info(sendRecvInfo.sendRecvChannels_, fixedSlices, sendRecvInfo.dataType_);
        return SendRecvBatchWrite(info, thread);
    }
    SendRecvReduceInfo info(sendRecvInfo.sendRecvChannels_, fixedSlices, sendRecvInfo.dataType_, reduceOp);
    return SendRecvBatchWriteReduce(info, thread);
}

static HcclResult RecvRead(const DataInfo& recvInfo, const ThreadHandle& thread, HcclReduceOp reduceOp)
{
    void* remoteBase = recvInfo.channel_.remoteCclMem.addr;
    std::vector<DataSlice> fixedSrc;
    FixupSliceAddrs(recvInfo.slices_.srcSlices_, remoteBase, fixedSrc);
    SlicesList fixedSlices(std::move(fixedSrc), recvInfo.slices_.dstSlices_);
    if (reduceOp == HCCL_REDUCE_RESERVED) {
        DataInfo info(recvInfo.channel_, fixedSlices, recvInfo.dataType_);
        return RecvBatchRead(info, thread);
    }
    DataReduceInfo info(recvInfo.channel_, fixedSlices, recvInfo.dataType_, reduceOp);
    return RecvBatchReadReduce(info, thread);
}

static HcclResult SendRecvRead(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread, HcclReduceOp reduceOp)
{
    void* remoteBase = sendRecvInfo.sendRecvChannels_.rxChannel_.remoteCclMem.addr;
    std::vector<DataSlice> fixedSrc;
    FixupSliceAddrs(sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_, remoteBase, fixedSrc);
    SlicesList fixedRx(std::move(fixedSrc), sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_);
    TxRxSlicesList fixedSlices(sendRecvInfo.sendRecvSlices_.txSlicesList_, fixedRx);
    if (reduceOp == HCCL_REDUCE_RESERVED) {
        SendRecvInfo info(sendRecvInfo.sendRecvChannels_, fixedSlices, sendRecvInfo.dataType_);
        return SendRecvBatchRead(info, thread);
    }
    SendRecvReduceInfo info(sendRecvInfo.sendRecvChannels_, fixedSlices, sendRecvInfo.dataType_, reduceOp);
    return SendRecvBatchReadReduce(info, thread);
}

static HcclResult GetTxRxChannels(
    const TransferContext& ctx, bool hasTx, bool hasRx, u32 dstRank, u32 srcRank,
    const std::vector<ChannelInfo>*& txChannels, const std::vector<ChannelInfo>*& rxChannels)
{
    const auto& channels = ctx.templateRes->channels;
    if (hasTx) {
        auto txIt = channels.find(dstRank);
        if (txIt == channels.end() || txIt->second.empty()) {
            HCCL_ERROR("[DataTransfer][Send] tx channel not found, dstRank[%u]", dstRank);
            return HCCL_E_INTERNAL;
        }
        txChannels = &txIt->second;
    }
    if (hasRx) {
        auto rxIt = channels.find(srcRank);
        if (rxIt == channels.end() || rxIt->second.empty()) {
            HCCL_ERROR("[DataTransfer][Send] rx channel not found, srcRank[%u]", srcRank);
            return HCCL_E_INTERNAL;
        }
        rxChannels = &rxIt->second;
    }
    return HCCL_SUCCESS;
}

// 按 dstRank 在 channels map 中的顺序位置取线程索引，
// 若超出 threads 向量范围则回退到线程 0。
ThreadHandle GetThreadForDstRank(const TemplateResource& res, u32 dstRank)
{
    u32 threadIdx = 0;
    auto it = res.channels.find(dstRank);
    if (it != res.channels.end()) {
        threadIdx = static_cast<u32>(std::distance(res.channels.begin(), it));
    }
    if (threadIdx >= res.threads.size()) {
        threadIdx = 0;
    }
    return res.threads[threadIdx];
}

// 通道级线程选择：reuseChannelThreads 用 threads[channelIdx]，
// 否则按 rankPos * channelsPerRank + channelIdx 分配，与单通道路径对齐。
static ThreadHandle GetThreadForChannel(const TransferContext& ctx, u32 channelIdx, u32 channelsPerRank)
{
    const u32 threadNum = static_cast<u32>(ctx.templateRes->threads.size());
    u32 threadIdx;
    if (ctx.reuseChannelThreads) {
        threadIdx = std::min(channelIdx, threadNum - 1);
    } else {
        auto it = ctx.templateRes->channels.find(ctx.txRxSlicesList.dstRankId_);
        u32 rankPos = static_cast<u32>(std::distance(ctx.templateRes->channels.begin(), it));
        threadIdx = std::min(rankPos * channelsPerRank + channelIdx, threadNum - 1);
    }
    return ctx.templateRes->threads[threadIdx];
}

HcclResult DataTransferSend(const TransferContext& ctx)
{
    TransferDirection direction = (ctx.remoteReadEnabled && ctx.buffType == BufferType::OUTPUT) ?
                                      TransferDirection::READ :
                                      TransferDirection::WRITE;

    bool hasTx = !ctx.txRxSlicesList.txSlicesList_.srcSlices_.empty();
    bool hasRx = !ctx.txRxSlicesList.rxSlicesList_.srcSlices_.empty();
    if (!hasTx && !hasRx) {
        return HCCL_SUCCESS;
    }

    u32 dstRank = ctx.txRxSlicesList.dstRankId_;
    u32 srcRank = ctx.txRxSlicesList.srcRankId_;

    const std::vector<ChannelInfo>* txChannels = nullptr;
    const std::vector<ChannelInfo>* rxChannels = nullptr;
    CHK_RET(GetTxRxChannels(ctx, hasTx, hasRx, dstRank, srcRank, txChannels, rxChannels));

    if (ctx.templateRes->threads.empty()) {
        HCCL_ERROR("[DataTransfer][Send] threads is empty");
        return HCCL_E_INTERNAL;
    }

    u32 channelsPerRank = 0;
    if (hasTx && hasRx) {
        channelsPerRank = static_cast<u32>(std::min(txChannels->size(), rxChannels->size()));
    } else if (hasTx) {
        channelsPerRank = static_cast<u32>(txChannels->size());
    } else {
        channelsPerRank = static_cast<u32>(rxChannels->size());
    }
    if (channelsPerRank == 0) {
        HCCL_ERROR("[DataTransfer][Send] channelsPerRank is 0, dstRank[%u], srcRank[%u]", dstRank, srcRank);
        return HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[ctx.dataType];
    const bool needSplit = (channelsPerRank > 1);

    for (u32 ch = 0; ch < channelsPerRank; ++ch) {
        // 单通道直接用原始 slices，多通道按通道切分
        SlicesList txSplit({}, {});
        SlicesList rxSplit({}, {});
        if (hasTx) {
            txSplit = needSplit ? SplitSlicesList(ctx.txRxSlicesList.txSlicesList_, ch, *txChannels, dataTypeSize) :
                                  ctx.txRxSlicesList.txSlicesList_;
        }
        if (hasRx) {
            rxSplit = needSplit ? SplitSlicesList(ctx.txRxSlicesList.rxSlicesList_, ch, *rxChannels, dataTypeSize) :
                                  ctx.txRxSlicesList.rxSlicesList_;
        }

        bool hasTxSplit = !txSplit.srcSlices_.empty();
        bool hasRxSplit = !rxSplit.srcSlices_.empty();
        if (!hasTxSplit && !hasRxSplit) {
            continue;
        }

        const ThreadHandle thread = GetThreadForChannel(ctx, ch, channelsPerRank);
        const ChannelInfo* txCh = hasTx ? &(*txChannels)[ch] : nullptr;
        const ChannelInfo* rxCh = hasRx ? &(*rxChannels)[ch] : nullptr;

        if (hasTxSplit && hasRxSplit && txCh != nullptr && rxCh != nullptr) {
            TxRxChannels channelPair(*txCh, *rxCh);
            TxRxSlicesList slicesList(txSplit, rxSplit);
            SendRecvInfo info(channelPair, slicesList, ctx.dataType);
            if (direction == TransferDirection::READ) {
                CHK_RET(SendRecvRead(info, thread, ctx.reduceOp));
            } else {
                CHK_RET(SendRecvWrite(info, thread, ctx.reduceOp));
            }
        } else if (hasTxSplit && txCh != nullptr) {
            DataInfo info(*txCh, txSplit, ctx.dataType);
            if (direction == TransferDirection::READ) {
                CHK_RET(SendRead(info, thread));
            } else {
                CHK_RET(SendWrite(info, thread, ctx.reduceOp));
            }
        } else if (hasRxSplit && rxCh != nullptr) {
            DataInfo info(*rxCh, rxSplit, ctx.dataType);
            if (direction == TransferDirection::READ) {
                CHK_RET(RecvRead(info, thread, ctx.reduceOp));
            } else {
                CHK_RET(RecvWrite(info, thread));
            }
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
