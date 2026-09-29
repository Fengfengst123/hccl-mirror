/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_ENGINE
#define HCCLV2_COLL_ALG_SELECTOR_ENGINE

#include <string>
#include <vector>

#include "alg_param.h"
#include "cost_model.h"
#include "cost_table.h"
#include "log.h"
#include "hccl_res.h"

namespace ops_hccl {

class SelectorEngine {
public:
    static SelectorEngine* Global();

    HcclResult Run(HcclComm comm, OpParam& param, TopoInfoWithNetLayerDetails* topoInfo, std::string& algName);

    // 新选择器算子白名单: 本迭代仅支持 AllReduce/ReduceScatter/AllGather
    static bool IsOpSupported(HcclCMDType opType);

private:
    SelectorEngine() = default;

    HcclResult InitCostModel(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, OpParam& param, CostModel*& cm);

    // 调 tuner 改 cost: Enrich 填 3D 名 + 插件改 cost(未加载或空表跳过, 含 AllToAll(V/VC) dataType 特判)
    HcclResult TunerEnrichCostTable(HcclComm comm, CostTable& ct, const OpParam& param);

    HcclResult SelectMinCost(const CostTable& ct, OpParam& param, std::string& algName);

    // AIV_ONLY 模式下选不到 AIV 算法时打印 ERROR，包含当前拓扑条件
    static void LogAivOnlyNotMatch(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo);

    static std::string QueryTemplateInfo(const std::string& algName);

    static std::string QueryExecutorName(const std::string& algName);

    static void
    LogSelectedAlgo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const std::string& algName);
};

} // namespace ops_hccl

#endif
