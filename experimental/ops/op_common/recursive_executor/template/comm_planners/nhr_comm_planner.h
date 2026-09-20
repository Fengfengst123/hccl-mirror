/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RE_NHR_COMM_PLANNER_H
#define RE_NHR_COMM_PLANNER_H

#include <vector>
#include "alg_param.h"
#include "data_types.h"

namespace ops_hccl {

struct DataParams;

struct NhrAllGatherSlicePair {
    void* srcPtr;
    void* dstPtr;
    std::vector<DataSlice>& srcSlices;
    std::vector<DataSlice>& dstSlices;
};

struct NhrStepParams {
    u32 delta;
    u32 sendToAlgRank;
    u32 recvFromAlgRank;
    u32 algRankStep;
    u32 nSlices;
};

// BuildNhrStepSlices 的输入上下文（消除过多函数参数）
struct NhrStepContext {
    const DataParams& tempAlgParams;
    const std::vector<u32>& ranks;
    const std::vector<u32>& ranksForInputData;
    u32 myRank;
    u32 myAlgRank;
    u32 rankSize;
    u32 step;
    u32 nSteps;
    u32 tailRankId;
    const NhrStepParams& params;
    const std::vector<u32>& ranksForOutputData;
    std::vector<u32>* lastStepRxRanks;
};

// BuildNhrStepSlices 的输出切片
struct NhrStepOutput {
    std::vector<DataSlice> txSrc;
    std::vector<DataSlice> txDst;
    std::vector<DataSlice> rxSrc;
    std::vector<DataSlice> rxDst;
};

// lastStepRxRanks 输出末步 rx 收到的 rankIds（末步才到位，PostCopy 不能提前搬）
HcclResult RunNhrAllGather(
    const DataParams& tempAlgParams, const std::vector<u32>& ranks, u32 myRank, std::vector<u32>& ranksForOutputData,
    std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>* lastStepRxRanks = nullptr);

} // namespace ops_hccl

#endif
