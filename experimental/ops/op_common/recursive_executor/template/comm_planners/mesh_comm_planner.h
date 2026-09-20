/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RE_MESH_COMM_PLANNER_H
#define RE_MESH_COMM_PLANNER_H

#include <vector>
#include "alg_param.h"
#include "data_types.h"

namespace ops_hccl {

struct DataParams;

struct MeshSliceContext {
    const DataParams& tempAlgParams;
    u64 sliceSize{0};
    u64 tailSize{0};
    u64 stride{0};
    u32 tailRankId{INVALID_VALUE_RANKID};
    u32 rankSize{0};
    bool directToOutput{false};
    std::vector<u32> ranksForInputData;

    explicit MeshSliceContext(const DataParams& params) : tempAlgParams(params) {}
};

struct MeshSlicePair {
    void* firstBufferPtr;
    void* secondBufferPtr;
    std::vector<DataSlice>& firstSlices;
    std::vector<DataSlice>& secondSlices;
};

HcclResult RunMeshAllGather(
    const DataParams& tempAlgParams, const std::vector<u32>& ranks, u32 myRank, std::vector<u32>& ranksForOutputData,
    std::vector<DataSlicesList>& txRxSlicesLists);

} // namespace ops_hccl

#endif
