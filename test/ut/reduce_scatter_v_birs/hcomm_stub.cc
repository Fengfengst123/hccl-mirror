/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BIRS-V 核心逻辑 UT 的 HCOMM 打桩: 被测函数(PrepareDataSliceV/CalcLocalStrideSize/CheckScratchCapacity)
// 为纯计算逻辑, 不触发数据搬运; 模板/executor 源文件整体编译所需的 Hcomm 符号在此提供空实现。

#include "hcomm_primitives.h"

extern "C" {

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}

int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return 0;
}

int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}

int32_t HcommWriteReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return 0;
}

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)channel;
    (void)remoteNotifyIdx;
    return 0;
}

int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    (void)thread;
    (void)channel;
    (void)localNotifyIdx;
    (void)timeOut;
    return 0;
}

int32_t HcommAcquireComm(const char* commId)
{
    (void)commId;
    return 0;
}

int32_t HcommReleaseComm(const char* commId)
{
    (void)commId;
    return 0;
}

int32_t HcommBatchModeStart(const char* batchTag)
{
    (void)batchTag;
    return 0;
}

int32_t HcommBatchModeEnd(const char* batchTag)
{
    (void)batchTag;
    return 0;
}

} // extern "C"
