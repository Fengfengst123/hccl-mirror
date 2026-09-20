/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RE_DATA_TRANSFER_H
#define RE_DATA_TRANSFER_H

#include "alg_param.h"
#include "data_types.h"

namespace ops_hccl {

struct TransferContext {
    // 控制本次传输方向：true 且 buffType 为 OUTPUT 时用 READ（远端读），
    // 否则用 WRITE。与 DataParams::enableRemoteMemAccess（系统级 OFFLOAD 标志）
    // 语义不同：此处是逐传输的方向开关，可由各模板按场景无条件置 true（如 Mesh），
    // 也可从 DataParams::enableRemoteMemAccess 透传（如基类/NHR 非末步）。
    bool remoteReadEnabled = true;
    BufferType buffType = BufferType::OUTPUT;
    DataSlicesList txRxSlicesList;
    const TemplateResource* templateRes = nullptr;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;

    // NHR 场景下各对端共用同一线程池（threads[channelIdx]），
    // 而非按 rankPos*channelsPerRank+channelIdx 分配。由调用方按场景设置。
    bool reuseChannelThreads = false;
};

HcclResult DataTransferSend(const TransferContext& ctx);

// 按 dstRank 在 channels map 中的顺序位置取线程索引，
// 若超出 threads 向量范围则回退到线程 0。
// 提取为独立函数，供 SendAll 预测末步通信线程，用于 LaunchPostCopy 的同步源。
ThreadHandle GetThreadForDstRank(const TemplateResource& res, u32 dstRank);

} // namespace ops_hccl

#endif // RE_DATA_TRANSFER_H
