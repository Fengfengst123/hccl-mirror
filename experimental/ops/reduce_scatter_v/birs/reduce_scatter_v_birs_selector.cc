/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_scatter_v_birs_selector.h"

namespace ops_hccl_experimental {
using ops_hccl::TopoInfo;

BirsVSelectResult DecideReduceScatterVBirsAlg(const TopoInfo& topoInfo, std::string& algName)
{
    if (topoInfo.userRankSize == 1) {
        return BirsVSelectResult::vRejectRankSizeOne;
    }
    if (topoInfo.deviceType != HcclDevType::DEV_TYPE_910_93 || (topoInfo.userRankSize % 2 != 0)) {
        return BirsVSelectResult::vNotSelected;
    }
    if (topoInfo.serverNum == 0) {
        return BirsVSelectResult::vRejectServerNumZero;
    }
    if (topoInfo.userRankSize / topoInfo.serverNum < 4) {
        return BirsVSelectResult::vRejectRanksPerServerLT4;
    }
    algName = "ReduceScatterVBIRSExecutor";
    return BirsVSelectResult::vSelected;
}

HcclResult BirsVSelectResultToCode(BirsVSelectResult result)
{
    switch (result) {
        case BirsVSelectResult::vSelected:
        case BirsVSelectResult::vNotSelected:
            return HCCL_SUCCESS;
        case BirsVSelectResult::vRejectRankSizeOne:
            return HCCL_E_INTERNAL;
        case BirsVSelectResult::vRejectServerNumZero:
        case BirsVSelectResult::vRejectRanksPerServerLT4:
            return HCCL_E_PARA;
    }
    return HCCL_E_INTERNAL;
}

} // namespace ops_hccl_experimental
