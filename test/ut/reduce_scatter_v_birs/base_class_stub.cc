/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 桩: AlgTemplateBase 的构造/析构/虚函数与线程同步原语。
// 真实实现(src/ops/op_common/algorithm/template/alg_template_base.cc)依赖 ExecTimeoutManager 与
// hcomm_primitives_dl 的 dlsym 运行时符号, 不适合 UT 独立编译; 本桩按真实实现的成员初始化语义等价提供,
// 保证被测的 BIRS-V 模板成员(slices_/scratchMem_ 等)行为与生产一致。

#include "alg_template_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

AlgTemplateBase::AlgTemplateBase()
    : slices_(slicesDummy_),
      count_(0),
      dataBytes_(0),
      dataType_(HCCL_DATA_TYPE_RESERVED),
      reductionOp_(HCCL_REDUCE_RESERVED),
      root_(INVALID_VALUE_RANKID),
      baseOffset_(0),
      barrierSwitchOn_(true)
{}

AlgTemplateBase::~AlgTemplateBase() { slices_.clear(); }

HcclResult AlgTemplateBase::Prepare(PrepareData& param)
{
    (void)param;
    return HCCL_E_PARA;
}

HcclResult AlgTemplateBase::Prepare(
    HcclMem& inputMem, HcclMem& outputMem, HcclMem& scratchMem, const u64 count, const HcclDataType dataType,
    ThreadHandle thread, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
    const u64 baseOffset, const bool disableDMAReduce)
{
    inputMem_ = inputMem;
    outputMem_ = outputMem;
    scratchMem_ = scratchMem;
    thread_ = thread;
    count_ = count;
    dataType_ = dataType;
    dataBytes_ = count * DataUnitSize(dataType);
    reductionOp_ = reductionOp;
    root_ = root;
    disableDMAReduce_ = disableDMAReduce;
    baseOffset_ = baseOffset;
    if (slices.size() > 0) {
        slices_.resize(slices.size());
        slices_ = slices;
    }
    return HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::Prepare(u32 interRank, u32 interRankSize)
{
    (void)interRank;
    (void)interRankSize;
    return HCCL_E_PARA;
}

HcclResult AlgTemplateBase::Prepare(
    HcclCollOpInfo* opInfo, const u32 userRank, const std::vector<u32>& ringsOrders,
    const std::vector<Slice>& userMemInputSlices)
{
    (void)opInfo;
    (void)userRank;
    (void)ringsOrders;
    (void)userMemInputSlices;
    return HCCL_E_PARA;
}

HcclResult AlgTemplateBase::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    (void)rank;
    (void)rankSize;
    (void)channels;
    return HCCL_SUCCESS;
}

HcclResult PreSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxMainToSub;
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxSubToMain)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxSubToMain;
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
