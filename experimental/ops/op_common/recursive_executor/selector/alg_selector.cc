/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_selector.h"
#include "log.h"

namespace ops_hccl {

// 校验 AlgoExecDesc 树结构：OMNIPIPE 的 Child 仅允许叶子或 OMNIPIPE 子树
static HcclResult ValidateAlgoExecDesc(const AlgoExecDesc& desc)
{
    for (size_t i = 0; i < desc.children.size(); i++) {
        if (std::get_if<TemplateExecDesc>(&desc.children[i])) {
            continue; // 叶子节点，合法
        }
        auto* subPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&desc.children[i]);
        if (subPtr == nullptr || *subPtr == nullptr) {
            HCCL_ERROR("[ValidateAlgoExecDesc] child[%zu] is null subtree", i);
            return HCCL_E_PARA;
        }
        // OMNIPIPE 节点的子树 Child 必须同为 OMNIPIPE
        if (desc.execPolicy == HcclAlgExecPolicy::OMNIPIPE && (*subPtr)->execPolicy != HcclAlgExecPolicy::OMNIPIPE) {
            HCCL_ERROR(
                "[ValidateAlgoExecDesc] OMNIPIPE child[%zu] is subtree with execPolicy=%d, "
                "only OMNIPIPE subtrees are allowed under OMNIPIPE",
                i, static_cast<int>((*subPtr)->execPolicy));
            return HCCL_E_PARA;
        }
        CHK_RET(ValidateAlgoExecDesc(**subPtr));
    }
    return HCCL_SUCCESS;
}

AlgSelector& AlgSelector::Instance()
{
    static AlgSelector instance;
    return instance;
}

HcclResult AlgSelector::Register(const std::string& algName, HcclAlgorithm algo)
{
    HcclResult ret = ValidateAlgoExecDesc(algo.algoExecDesc);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AlgSelector] Register '%s' failed: invalid AlgoExecDesc tree", algName.c_str());
        return ret;
    }
    std::lock_guard<std::mutex> lock(mu_);
    algMap_[algName] = std::move(algo);
    return HCCL_SUCCESS;
}

bool AlgSelector::GetAlgorithm(const std::string& algName, HcclAlgorithm& algo) const
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = algMap_.find(algName);
    if (it == algMap_.end()) {
        return false;
    }
    algo = it->second;
    return true;
}

} // namespace ops_hccl
