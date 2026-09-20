/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

// 跟踪最后一次调用的 wrapper 函数，供测试验证分支覆盖
// 0=none, 1=SendBatchWrite, 2=SendBatchWriteReduce,
// 3=SendRecvBatchWrite, 4=SendRecvBatchWriteReduce,
// 5=RecvBatchRead, 6=RecvBatchReadReduce,
// 7=SendRead, 8=RecvWrite,
// 9=SendRecvBatchRead, 10=SendRecvBatchReadReduce
int g_lastCalledFunc = 0;

// 检测 wrapper 收到的 slice 地址是否有 nullptr（FixupSliceAddrs 应已修正）
bool g_nullAddrDetected = false;

static void CheckSlices(const std::vector<DataSlice>& src, const std::vector<DataSlice>& dst)
{
    for (const auto& s : src) {
        if (s.addr_ == nullptr)
            g_nullAddrDetected = true;
    }
    for (const auto& s : dst) {
        if (s.addr_ == nullptr)
            g_nullAddrDetected = true;
    }
}

HcclResult SendBatchWrite(const DataInfo& sendInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 1;
    CheckSlices(sendInfo.slices_.srcSlices_, sendInfo.slices_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult SendBatchWriteReduce(const DataReduceInfo& sendInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 2;
    CheckSlices(sendInfo.slices_.srcSlices_, sendInfo.slices_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchWrite(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 3;
    CheckSlices(
        sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_, sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchWriteReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 4;
    CheckSlices(
        sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_, sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult RecvBatchRead(const DataInfo& recvInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 5;
    CheckSlices(recvInfo.slices_.srcSlices_, recvInfo.slices_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult RecvBatchReadReduce(const DataReduceInfo& recvInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 6;
    CheckSlices(recvInfo.slices_.srcSlices_, recvInfo.slices_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult SendRead(const DataInfo& sendInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 7;
    CheckSlices(sendInfo.slices_.srcSlices_, sendInfo.slices_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult RecvWrite(const DataInfo& recvInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 8;
    CheckSlices(recvInfo.slices_.srcSlices_, recvInfo.slices_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchRead(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 9;
    CheckSlices(
        sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_, sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_);
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchReadReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
{
    g_lastCalledFunc = 10;
    CheckSlices(
        sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_, sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
