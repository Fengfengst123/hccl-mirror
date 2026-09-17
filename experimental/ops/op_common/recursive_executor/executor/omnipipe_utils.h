/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OMNIPIPE_UTILS_H
#define OMNIPIPE_UTILS_H

#include <cstdint>
#include <vector>
#include "alg_param.h"
#include "algo_desc.h"

namespace ops_hccl {

constexpr uint32_t OMNI_MAX_STEP_NUM = 5;
constexpr double OMNI_MESH_BW = 56;     // Mesh组网每条链路等效带宽56G
constexpr double OMNI_CLOS_BW = 56 * 2; // Clos组网每条链路等效带宽112G

double CalcBandwidth2D(double xB, double yB, u32 xRankSize, u32 yRankSize, u32 maxStepNum, u32& steps, double& scale);
u32 CalcOmniPipeSteps(double bandwidthRatio, u32 xRankSize, u32 maxStep, double& scale);
void CalcOmniPipeDataRatio(
    double bandwidthRatio, u32 xRankSize, u32 yRankSize, u32 steps, double scale, double* xDataSizeRatio,
    double* yDataSizeRatio);
HcclResult CalcOmniPipeDataSlice(
    const OmniPipeXYdata& xy, u64 sliceCount, std::vector<u64>& xSliceCount, std::vector<u64>& ySliceCount);

// PARALLEL 数据切分整数运算：childSlice[i] = floor(parentSlice * ratio / ratioSum)。
// 分步计算避免 u64 乘法溢出：(parentSlice/ratioSum)*ratio + ((parentSlice%ratioSum)*ratio)/ratioSum。
// 调用方需保证 ratioSum > 0；最后一个 Child 的 sliceCount 应由调用方取 parentSlice - 已分配和。
u64 CalcParallelSliceCount(u64 parentSlice, u64 ratio, u64 ratioSum);

} // namespace ops_hccl

#endif
