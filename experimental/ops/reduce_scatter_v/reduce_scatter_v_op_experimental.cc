/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_op_experimental.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include "op_common_experimental.h"
#include "param_check.h"
#include "reduce_scatter_v.h"
#include "load_kernel.h"
#include "hcomm_dlsym.h"
#include "hcomm_host_profiling_dl.h"

using namespace std;

extern "C" HcclResult HcclReduceScatterVInner(
    void* sendBuf, const void* sendCounts, const void* sendDispls, void* recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream);

namespace ops_hccl_experimental {
using ops_hccl::CheckCount;
using ops_hccl::CheckDataTypeRSV;
using ops_hccl::CheckReduceOp;
using ops_hccl::CheckReduceScatterVInputParam;
using ops_hccl::COMM_INDENTIFIER_MAX_LENGTH;
using ops_hccl::DATATYPE_SIZE_TABLE;
using ops_hccl::HcclCheckTag;
using ops_hccl::HcomCheckUserRank;
using ops_hccl::InitEnvConfig;
using ops_hccl::LoadAICPUKernel;
using ops_hccl::LogHcclExit;
using ops_hccl::OpMode;
using ops_hccl::PrepareReduceScatterVParam;
using ops_hccl::ReduceScatterVEntryLog;

HcclResult ReduceScatterVExperimental(
    void* sendBuf, const void* sendCounts, const void* sendDispls, void* recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    CHK_RET(InitEnvConfig());

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    if (rankSize == 1 || !MatchBIRSV()) {
        return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }
    // BIRS-V 仅适用于 A3(DEV_TYPE_910_93) 且 rank 数为偶数(与 DecideReduceScatterVBirsAlg 的 vNotSelected
    // 判定一致); 不满足时同样回退老流程, 保证开启 HCCL_BIRS_ENABLE 后非目标拓扑的行为与未开启时一致
    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    CHK_RET(HcclGetDeviceType(deviceType));
    if (rankSize % 2 != 0 || deviceType != HcclDevType::DEV_TYPE_910_93) {
        HCCL_INFO(
            "[ReduceScatterVExperimental] BIRS-V not applicable, deviceType[%d] rankSize[%u], fallback to inner flow",
            static_cast<int>(deviceType), rankSize);
        return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    }
    // 校验sendCounts全部为0的情况
    const u64* sendCountsAddr = reinterpret_cast<const u64*>(sendCounts);
    CHK_RET(CheckReduceScatterVInputParam(comm, sendBuf, recvBuf, recvCount, sendCounts, sendDispls, stream));
    CHK_PRT_RET(
        std::all_of(
            sendCountsAddr, sendCountsAddr + rankSize,
            [](auto count) {
                return count == 0;
            }),
        HCCL_WARNING("input all %u elements in sendCounts are 0, return success", rankSize), HCCL_SUCCESS);
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET(HcomCheckUserRank(rankSize, userRank));
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataTypeRSV(dataType));
    CHK_RET(CheckReduceOp(dataType, op));

    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "ReduceScatterV_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));

    /* 接口交互信息日志 */
    CHK_RET(ReduceScatterVEntryLog(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, stream, tag, rankSize,
        "HcclReduceScatterV"));

    // 申请OpParam参数结构体内存(包含变长varData区)
    u64 varMemSize = 2 * rankSize * sizeof(u64);
    void* paramMem = malloc(sizeof(OpParam) + varMemSize);
    if (!paramMem) {
        HCCL_ERROR("[ReduceScatterVExperimental] malloc OpParam failed!");
        return HCCL_E_INTERNAL;
    }
    auto deleter = [](OpParam* p) {
        if (p) {
            p->~OpParam();
            free(p);
        }
    };
    OpParam* tmpParamPtr = new (paramMem) OpParam();
    std::unique_ptr<OpParam, decltype(deleter)> paramPtr(tmpParamPtr, deleter);
    OpParam& param = *paramPtr;

    CHK_RET(ReduceScatterVOutPlaceCustom(
        param, sendBuf, sendDispls, sendCounts, recvBuf, recvCount, dataType, op, comm, stream, rankSize));

    CHK_RET(LogHcclExit("HcclReduceScatterV", tag.c_str(), startut));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVOutPlaceCustom(
    OpParam& param, void* sendBuf, const void* sendDispls, const void* sendCounts, void* recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream, u32 userRankSize)
{
    HCCL_INFO("Start to execute ReduceScatterVOutPlaceCustom");
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const std::string tag = "ReduceScatterV_" + std::string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    CHK_RET(PrepareReduceScatterVParam(
        sendBuf, sendDispls, sendCounts, recvBuf, recvCount, dataType, op, comm, stream, tag, OpMode::OPBASE,
        userRankSize, 2 * userRankSize * sizeof(u64), param));
    // host(COMM_ENGINE_CPU_TS)编排路径由 KernelRunLevel0 直接读取 param.vDataDes.counts/displs(host指针);
    // 在 experimental 侧补齐 displs, 不改动公共 PrepareReduceScatterVParam 的 production 路径。
    // AICPU_TS 路径下 PrepareReduceScatterVVarData 会用 device 指针覆写 counts/displs, 此赋值无副作用。
    param.vDataDes.displs = const_cast<void*>(static_cast<const void*>(sendDispls));

    if (IsAiCpuMode(param.deviceType, userRankSize)) {
        HCCL_DEBUG("is aicpu mode");
        CHK_RET(LoadAICPUKernel());
        param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    } else {
        HCCL_DEBUG("is host mode");
        param.engine = CommEngine::COMM_ENGINE_CPU_TS;
    }

    uint64_t beginTime = 0;
    if (HcommIsProfilingSupported()) {
        beginTime = HcommGetProfilingSysCycleTime();
    }

    CHK_RET(ProcessA3(comm, param, beginTime));

    HCCL_INFO("Execute ReduceScatterVOutPlaceCustom success.");
    return HCCL_SUCCESS;
}

bool MatchBIRSV()
{
    const char* val = std::getenv("HCCL_BIRS_ENABLE");
    if (val == nullptr)
        return false;
    std::string str = val;
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "true") {
        return true;
    }
    if (lower == "false") {
        return false;
    }
    HCCL_WARNING(
        "[MatchBIRSV] HCCL_BIRS_ENABLE[%s] is invalid, expected TRUE/FALSE, fallback to disabled", str.c_str());
    return false;
}

} // namespace ops_hccl_experimental
