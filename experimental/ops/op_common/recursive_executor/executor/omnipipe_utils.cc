/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "omnipipe_utils.h"
#include <cmath>
#include <numeric>
#include "log.h"

namespace ops_hccl {

double CalcBandwidth2D(double xB, double yB, u32 xRankSize, u32 yRankSize, u32 maxStepNum, u32& steps, double& scale)
{
    HCCL_DEBUG("[CalcBandwidth2D] start");
    if (yRankSize == 1) {
        HCCL_DEBUG("[CalcBandwidth2D] xB=[%f]", xB);
        steps = 1;
        return xB;
    } else if (xRankSize == 1) {
        HCCL_DEBUG("[CalcBandwidth2D] yB=[%f]", yB);
        steps = 1;
        return yB;
    } else {
        double bandwidthRatio = yB / xB;
        steps = CalcOmniPipeSteps(bandwidthRatio, xRankSize, maxStepNum, scale);
        std::vector<double> xDataSizeRatio(steps);
        std::vector<double> yDataSizeRatio(steps);
        CalcOmniPipeDataRatio(
            bandwidthRatio, xRankSize, yRankSize, steps, scale, xDataSizeRatio.data(), yDataSizeRatio.data());
        double xds = std::accumulate(xDataSizeRatio.begin(), xDataSizeRatio.end(), 0.0);
        HCCL_INFO("[CalcBandwidth2D] Bandwidth2D=[%f]", xB / xds);
        return xB / xds;
    }
}

u32 CalcOmniPipeSteps(double bandwidthRatio, u32 xRankSize, u32 maxStep, double& scale)
{
    HCCL_DEBUG("[CalcOmniPipeSteps] start");
    if (maxStep <= 1) {
        scale = 1.0;
        return 1;
    }
    u32 steps = 1;
    double growth = (xRankSize - 1) / bandwidthRatio;
    double geomSum = 0;
    for (u64 t = 0; t < maxStep - 1; t++) {
        geomSum += std::pow(growth, t);
    }
    scale = bandwidthRatio / geomSum;
    if (xRankSize > bandwidthRatio) {
        if (std::fabs(growth - 1.0) < 1e-4) {
            steps = static_cast<u32>(bandwidthRatio + 1);
        } else {
            steps = static_cast<u32>(std::ceil(std::log((growth - 1) * bandwidthRatio + 1) / std::log(growth))) + 1;
        }
        if (steps <= maxStep) {
            scale = 1.0;
        } else {
            steps = maxStep;
        }
    } else {
        steps = maxStep;
    }
    HCCL_INFO(
        "[CalcOmniPipeSteps] bandwidthRatio=[%f],growth=[%f],step=[%u],scale=[%f]", bandwidthRatio, growth, steps,
        scale);
    return steps;
}

void CalcOmniPipeDataRatio(
    double bandwidthRatio, u32 xRankSize, u32 yRankSize, u32 steps, double scale, double* xDataSizeRatio,
    double* yDataSizeRatio)
{
    HCCL_DEBUG("[CalcOmniPipeDataRatio] start");
    yDataSizeRatio[0] = 1.0;
    xDataSizeRatio[0] = scale / bandwidthRatio;
    double sumXDataSize = xDataSizeRatio[0];
    double sumYDataSize = yDataSizeRatio[0];
    for (u64 index = 1; index < steps - 1; index++) {
        if (index == steps - 2) {
            xDataSizeRatio[index] = 1.0 - sumXDataSize;
            yDataSizeRatio[index] = bandwidthRatio * xDataSizeRatio[index];
        } else {
            yDataSizeRatio[index] = xDataSizeRatio[index - 1] * (xRankSize - 1);
            xDataSizeRatio[index] = yDataSizeRatio[index] / bandwidthRatio;
        }
        sumXDataSize += xDataSizeRatio[index];
        sumYDataSize += yDataSizeRatio[index];
    }
    double remainSliceRatio = 1.0 - (sumYDataSize - 1.0) / (xRankSize - 1);
    xDataSizeRatio[steps - 1]
        = remainSliceRatio * (xRankSize - 1) * (yRankSize - 1) / (xRankSize - 1 + (yRankSize - 1) * bandwidthRatio);
    yDataSizeRatio[steps - 1] = xDataSizeRatio[steps - 1] * bandwidthRatio;
    HCCL_DEBUG("[CalcOmniPipeDataRatio]  end remainSliceRatio =%f", remainSliceRatio);
    return;
}

HcclResult CalcOmniPipeDataSlice(
    const OmniPipeXYdata& xy, u64 sliceCount, std::vector<u64>& xSliceCount, std::vector<u64>& ySliceCount)
{
    HCCL_DEBUG("[CalcOmniPipeDataSlice] start");
    double bandwidthRatio = xy.bandwidthRatio;
    u32 xRankSize = xy.xEqRankSize;
    u32 yRankSize = xy.yEqRankSize;
    u32 steps = xy.steps;
    double scale = xy.scale;
    if (xRankSize <= 1 || yRankSize <= 1) {
        HCCL_ERROR("[CalcOmniPipeDataSlice] xRankSize =%u or yRankSize=%u is invalid", xRankSize, yRankSize);
        return HCCL_E_PARA;
    }
    if (steps < 2 || steps > xSliceCount.size() || steps > ySliceCount.size()) {
        HCCL_ERROR(
            "[CalcOmniPipeDataSlice] steps=%u is invalid (xSliceCount.size=%zu, ySliceCount.size=%zu)", steps,
            xSliceCount.size(), ySliceCount.size());
        return HCCL_E_PARA;
    }
    // 调用 Ratio 版获取递推比例序列（单一事实来源），整数域仅做截断与倍数取整
    std::vector<double> xRatio(steps);
    std::vector<double> yRatio(steps);
    CalcOmniPipeDataRatio(bandwidthRatio, xRankSize, yRankSize, steps, scale, xRatio.data(), yRatio.data());
    xSliceCount.at(0) = static_cast<u64>(xRatio[0] * sliceCount);
    ySliceCount.at(0) = sliceCount;
    u64 sumXData = xSliceCount[0];
    u64 sumYData = ySliceCount[0];
    for (u64 index = 1; index < steps - 1; index++) {
        if (index == steps - 2) {
            // 特例步：x 取整数余量精确收口，y 向下取整到 (xRankSize-1) 倍数
            xSliceCount[index] = sliceCount - sumXData;
            ySliceCount[index]
                = static_cast<u64>(bandwidthRatio * xSliceCount[index] / (xRankSize - 1)) * (xRankSize - 1);
        } else {
            // 一般步：从 Ratio 版取比例乘 sliceCount 落到整数域，y 向下取整到 (xRankSize-1) 倍数
            xSliceCount[index] = static_cast<u64>(xRatio[index] * sliceCount);
            ySliceCount[index] = static_cast<u64>(yRatio[index] * sliceCount / (xRankSize - 1)) * (xRankSize - 1);
        }
        sumXData += xSliceCount[index];
        sumYData += ySliceCount[index];
    }
    // 末步：整数域 remain 公式 + 倍数取整（精确余量收口）
    u64 remainSlice = sliceCount - (sumYData - sliceCount) / (xRankSize - 1);
    xSliceCount[steps - 1]
        = remainSlice * (xRankSize - 1) * (yRankSize - 1) / (xRankSize - 1 + (yRankSize - 1) * bandwidthRatio);
    xSliceCount[steps - 1] = xSliceCount[steps - 1] / (yRankSize - 1) * (yRankSize - 1);
    ySliceCount[steps - 1] = (remainSlice - xSliceCount[steps - 1] / (yRankSize - 1)) * (xRankSize - 1);
    HCCL_DEBUG("[CalcOmniPipeDataSlice] end remainSlice =%llu", remainSlice);
    return HCCL_SUCCESS;
}

u64 CalcParallelSliceCount(u64 parentSlice, u64 ratio, u64 ratioSum)
{
    // 整数运算：floor(parentSlice * ratio / ratioSum)
    // 分步避免 u64 乘法溢出：(parentSlice/ratioSum)*ratio + ((parentSlice%ratioSum)*ratio)/ratioSum
    // 溢出安全性：base*ratio <= parentSlice（因 ratio <= ratioSum）；
    // remainder*ratio < ratioSum^2 <= (2^32)^2 = 2^64（ratioSum 为 u32 比例之和，实际远小于 2^32）
    u64 base = parentSlice / ratioSum;
    u64 remainder = parentSlice % ratioSum;
    return base * ratio + (remainder * ratio) / ratioSum;
}

} // namespace ops_hccl
