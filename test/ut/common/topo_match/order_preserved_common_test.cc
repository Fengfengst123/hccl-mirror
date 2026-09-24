/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "order_preserved_common.h"

namespace {
int g_hcommVersion = CANN_VERSION(9, 2, 0, 2);
}

extern "C" int GetHcommVersion(void) { return g_hcommVersion; }

namespace ops_hccl {
namespace {
    u8 g_environmentLevel = static_cast<u8>(DeterministicEnableLevel::DETERMINISTIC_DISABLE);
    uint32_t g_deterministicLevel = static_cast<uint32_t>(DeterministicEnableLevel::DETERMINISTIC_DISABLE);
    HcclComm g_queriedComm = nullptr;
} // namespace

const u8& GetExternalInputHcclDeterministic() { return g_environmentLevel; }

DlHcommFunction::~DlHcommFunction() = default;

DlHcommFunction& DlHcommFunction::GetInstance()
{
    static DlHcommFunction instance;
    return instance;
}

namespace {

    void SetCommDeterministicLevel(uint32_t deterministicLevel)
    {
        g_hcommVersion = CANN_VERSION(9, 2, 0, 2);
        g_deterministicLevel = deterministicLevel;
        DlHcommFunction::GetInstance().dlHcclConfigGetInfo
            = [](HcclComm comm, HcclConfigType, uint32_t, void* info) -> HcclResult {
            g_queriedComm = comm;
            *static_cast<uint32_t*>(info) = g_deterministicLevel;
            return HCCL_SUCCESS;
        };
    }

    OpParam MakeStrictSupportedParam()
    {
        OpParam param{};
        param.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
        param.reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
        return param;
    }

    TEST(OrderPreservedCommonTest, UsesDeterministicLevelFromComm)
    {
        OpParam param = MakeStrictSupportedParam();
        int commToken = 0;
        param.hcclComm = &commToken;

        SetCommDeterministicLevel(static_cast<uint32_t>(DeterministicEnableLevel::DETERMINISTIC_STRICT));
        EXPECT_TRUE(IsNeedStrictModeForOrderPreserved(param, 4));
        EXPECT_EQ(g_queriedComm, param.hcclComm);

        SetCommDeterministicLevel(static_cast<uint32_t>(DeterministicEnableLevel::DETERMINISTIC_ENABLE));
        EXPECT_FALSE(IsNeedStrictModeForOrderPreserved(param, 4));

        SetCommDeterministicLevel(static_cast<uint32_t>(DeterministicEnableLevel::DETERMINISTIC_DISABLE));
        EXPECT_FALSE(IsNeedStrictModeForOrderPreserved(param, 4));
    }

    TEST(OrderPreservedCommonTest, KeepsExistingStrictEligibilityConditions)
    {
        OpParam param = MakeStrictSupportedParam();
        SetCommDeterministicLevel(static_cast<uint32_t>(DeterministicEnableLevel::DETERMINISTIC_STRICT));

        EXPECT_FALSE(IsNeedStrictModeForOrderPreserved(param, MIN_STRICT_RANK_NUM_ORDER_PRESERVED));

        param.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
        EXPECT_FALSE(IsNeedStrictModeForOrderPreserved(param, 4));

        param.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
        param.reduceType = HcclReduceOp::HCCL_REDUCE_MAX;
        EXPECT_FALSE(IsNeedStrictModeForOrderPreserved(param, 4));
    }

} // namespace
} // namespace ops_hccl
