/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base_v2.h"
#include "aiv_reduce_scatter_mesh_1d.h"
#include "aiv_reduce_scatter_mesh_1d_bigdata.h"
#include "aiv_reduce_scatter_local_tree.h"
#include "aiv_reduce_scatter_local_tree_corectrl.h"

template <typename T>
__aicore__ inline void AivReduceScatterV2SuperKernelDispatch(SUPERKERNEL_ARGS_DEF)
{
    // 仅读取分发所需字段，不做完整Init：避免GetTag重复自增破坏相邻算子flag半区交替
    __gm__ AivSuperKernelArgs* args = reinterpret_cast<__gm__ AivSuperKernelArgs*>(hiddenInput);
    if (args->numBlocks > 2 * args->rankSize) {
        AivReduceScatterV2Mesh1DBigDataSuperKernel<T>(SUPERKERNEL_ARGS_CALL);
    } else if (args->numBlocks >= args->rankSize) {
        AivReduceScatterV2LocalTreeSuperKernel<T>(SUPERKERNEL_ARGS_CALL);
    } else {
        AivReduceScatterV2LocalTreeCoreCtrlSuperKernel<T>(SUPERKERNEL_ARGS_CALL);
    }
}

__aicore__ inline void sk_rs_superkernel_dispatch(SUPERKERNEL_ARGS_DEF)
{
#ifdef HCCL_DTYPE_INT8
    AivReduceScatterV2SuperKernelDispatch<int8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT16
    AivReduceScatterV2SuperKernelDispatch<int16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT32
    AivReduceScatterV2SuperKernelDispatch<int32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP16
    AivReduceScatterV2SuperKernelDispatch<half>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP32
    AivReduceScatterV2SuperKernelDispatch<float>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_BFP16
    AivReduceScatterV2SuperKernelDispatch<bfloat16_t>(SUPERKERNEL_ARGS_CALL);
#else
#endif
}

extern "C" __aicore__ void sk_reducescatter_mesh_1d(SUPERKERNEL_LITE_ARGS_DEF)
{
    AIV_INFO_HINT;
    SUPERKERNEL_LITE_ARGS_EXTRACT;
    return sk_rs_superkernel_dispatch(SUPERKERNEL_ARGS_CALL);
}
