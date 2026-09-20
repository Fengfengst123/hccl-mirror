/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "data_transfer.h"
#include "template_utils.h"

using namespace ops_hccl;

extern int g_lastCalledFunc;
extern bool g_nullAddrDetected;

// 测试辅助：构造有 remoteCclMem 地址的 ChannelInfo
static ChannelInfo MakeChannel(u32 remoteRank)
{
    ChannelInfo ch;
    ch.isValid = true;
    ch.remoteRank = remoteRank;
    ch.remoteCclMem.addr = reinterpret_cast<void*>(0x1000); // 非 null fallback 基址
    ch.remoteCclMem.size = 4096;
    return ch;
}

// 测试辅助：构造有数据的 DataSlice
static DataSlice MakeSlice(void* addr, u64 size) { return DataSlice(addr, 0, size, size); }

// 测试辅助：构造有 src 切片的 SlicesList
static SlicesList MakeTxSlices(void* srcAddr = reinterpret_cast<void*>(0x2000), void* dstAddr = nullptr)
{
    std::vector<DataSlice> src = {MakeSlice(srcAddr, 128)};
    std::vector<DataSlice> dst = {MakeSlice(dstAddr, 128)};
    return SlicesList(src, dst);
}

// 测试辅助：构造空的 SlicesList（src 为空 → hasTx/hasRx = false）
static SlicesList MakeEmptySlices()
{
    std::vector<DataSlice> empty;
    return SlicesList(empty, empty);
}

// 测试辅助：构造 TransferContext
static TransferContext MakeContext(
    bool hasTx, bool hasRx, bool remoteRead, u32 dstRank = 5, u32 srcRank = 3, const TemplateResource* res = nullptr)
{
    TransferContext ctx;
    ctx.remoteReadEnabled = remoteRead;
    ctx.buffType = BufferType::OUTPUT;
    ctx.dataType = HCCL_DATA_TYPE_INT8;
    ctx.reduceOp = HCCL_REDUCE_RESERVED;
    ctx.templateRes = res;

    SlicesList tx = hasTx ? MakeTxSlices() : MakeEmptySlices();
    SlicesList rx = hasRx ? MakeTxSlices() : MakeEmptySlices();
    ctx.txRxSlicesList = DataSlicesList(tx, rx, srcRank, dstRank);
    return ctx;
}

// 测试辅助：构造有 channels 和 threads 的 TemplateResource
static TemplateResource MakeResource(u32 dstRank, u32 srcRank, bool withThreads = true)
{
    TemplateResource res;
    res.channels[dstRank] = {MakeChannel(dstRank)};
    res.channels[srcRank] = {MakeChannel(srcRank)};
    if (withThreads) {
        res.threads = {100, 200, 300};
    }
    return res;
}

// ============ DataTransferSend：空数据直接返回 ============

TEST(DataTransferTest, BothEmptyReturnsSuccess)
{
    TemplateResource res = MakeResource(5, 3);
    TransferContext ctx = MakeContext(false, false, false, 5, 3, &res);
    g_lastCalledFunc = 0;

    EXPECT_EQ(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 0); // 没有调用任何 wrapper
}

// ============ DataTransferSend：threads 为空报错 ============

TEST(DataTransferTest, EmptyThreadsError)
{
    TemplateResource res = MakeResource(5, 3, false); // 无 threads
    TransferContext ctx = MakeContext(true, false, false, 5, 3, &res);
    g_lastCalledFunc = 0;

    EXPECT_NE(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 0); // 没有调用 wrapper，提前报错
}

// ============ DataTransferSend：channel 缺失报错 ============

TEST(DataTransferTest, TxChannelNotFound)
{
    TemplateResource res; // 空 channels
    res.threads = {100};
    TransferContext ctx = MakeContext(true, false, false, 5, 3, &res);

    EXPECT_NE(DataTransferSend(ctx), HCCL_SUCCESS);
}

TEST(DataTransferTest, RxChannelNotFound)
{
    TemplateResource res; // 空 channels
    res.threads = {100};
    TransferContext ctx = MakeContext(false, true, false, 5, 3, &res);

    EXPECT_NE(DataTransferSend(ctx), HCCL_SUCCESS);
}

// ============ DataTransferSend：hasTx && hasRx，WRITE 方向 ============

TEST(DataTransferTest, TxAndRxWrite)
{
    TemplateResource res = MakeResource(5, 3);
    TransferContext ctx = MakeContext(true, true, false, 5, 3, &res);
    g_lastCalledFunc = 0;

    EXPECT_EQ(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 3); // SendRecvBatchWrite
}

// ============ DataTransferSend：hasTx only，WRITE 方向 ============

TEST(DataTransferTest, TxOnlyWrite)
{
    TemplateResource res = MakeResource(5, 3);
    TransferContext ctx = MakeContext(true, false, false, 5, 3, &res);
    g_lastCalledFunc = 0;

    EXPECT_EQ(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 1); // SendBatchWrite
}

// ============ DataTransferSend：hasRx only，WRITE 方向 ============

TEST(DataTransferTest, RxOnlyWrite)
{
    TemplateResource res = MakeResource(5, 3);
    TransferContext ctx = MakeContext(false, true, false, 5, 3, &res);
    g_lastCalledFunc = 0;

    EXPECT_EQ(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 8); // RecvWrite
}

// ============ DataTransferSend：hasTx && hasRx，READ 方向 ============

TEST(DataTransferTest, TxAndRxRead)
{
    TemplateResource res = MakeResource(5, 3);
    TransferContext ctx = MakeContext(true, true, true, 5, 3, &res);
    g_lastCalledFunc = 0;

    EXPECT_EQ(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 9); // SendRecvBatchRead
}

// ============ DataTransferSend：hasTx only，READ 方向 ============

TEST(DataTransferTest, TxOnlyRead)
{
    TemplateResource res = MakeResource(5, 3);
    TransferContext ctx = MakeContext(true, false, true, 5, 3, &res);
    g_lastCalledFunc = 0;

    EXPECT_EQ(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 7); // SendRead
}

// ============ DataTransferSend：hasRx only，READ 方向 ============

TEST(DataTransferTest, RxOnlyRead)
{
    TemplateResource res = MakeResource(5, 3);
    TransferContext ctx = MakeContext(false, true, true, 5, 3, &res);
    g_lastCalledFunc = 0;

    EXPECT_EQ(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 5); // RecvBatchRead
}

// ============ FixupSliceAddrs：nullptr 修正为 remoteBase ============

TEST(DataTransferTest, FixupSliceAddrsNullptrFixed)
{
    TemplateResource res = MakeResource(5, 3);
    TransferContext ctx;
    ctx.remoteReadEnabled = false;
    ctx.buffType = BufferType::OUTPUT;
    ctx.dataType = HCCL_DATA_TYPE_INT8;
    ctx.reduceOp = HCCL_REDUCE_RESERVED;
    ctx.templateRes = &res;

    // tx 切片 dst addr 为 nullptr，应被 FixupSliceAddrs 修正为 remoteCclMem.addr
    std::vector<DataSlice> txSrc = {MakeSlice(reinterpret_cast<void*>(0x2000), 128)};
    std::vector<DataSlice> txDst = {MakeSlice(nullptr, 128)}; // nullptr dst
    SlicesList tx(txSrc, txDst);
    SlicesList rx = MakeEmptySlices();
    ctx.txRxSlicesList = DataSlicesList(tx, rx, 3, 5);

    g_nullAddrDetected = false;
    g_lastCalledFunc = 0;

    EXPECT_EQ(DataTransferSend(ctx), HCCL_SUCCESS);
    EXPECT_EQ(g_lastCalledFunc, 1);   // SendBatchWrite
    EXPECT_FALSE(g_nullAddrDetected); // FixupSliceAddrs 已修正 nullptr
}

// ============ GetThreadForDstRank ============

TEST(DataTransferTest, GetThreadForDstRankFound)
{
    TemplateResource res;
    res.channels[5] = {MakeChannel(5)};
    res.channels[3] = {MakeChannel(3)};
    res.threads = {100, 200, 300};

    // dstRank=5 是 channels map 的第 1 个元素（distance=0），取 threads[0]
    ThreadHandle t = GetThreadForDstRank(res, 5);
    EXPECT_EQ(t, res.threads[0]);
}

TEST(DataTransferTest, GetThreadForDstRankNotFoundFallsBackToZero)
{
    TemplateResource res;
    res.channels[5] = {MakeChannel(5)};
    res.threads = {100, 200, 300};

    // dstRank=99 不在 channels map，threadIdx=0，取 threads[0]
    ThreadHandle t = GetThreadForDstRank(res, 99);
    EXPECT_EQ(t, res.threads[0]);
}

TEST(DataTransferTest, GetThreadForDstRankExceedsThreadsSize)
{
    TemplateResource res;
    // channels map 有 5 个元素，但 threads 只有 2 个
    res.channels[10] = {MakeChannel(10)};
    res.channels[20] = {MakeChannel(20)};
    res.channels[30] = {MakeChannel(30)};
    res.channels[40] = {MakeChannel(40)};
    res.channels[50] = {MakeChannel(50)};
    res.threads = {100, 200};

    // dstRank=50 是第 5 个元素（distance=4），超出 threads.size()=2，回退到 0
    ThreadHandle t = GetThreadForDstRank(res, 50);
    EXPECT_EQ(t, res.threads[0]);
}
