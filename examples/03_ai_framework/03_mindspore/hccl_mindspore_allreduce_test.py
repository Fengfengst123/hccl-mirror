#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.communication import get_group_size, get_rank, init


def main():
    # 图模式下设备由启动器（如msrun）通过环境变量注入，此处不显式指定device_id
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    # 初始化通信域，MindSpore 底层通过 HCCL 完成集合通信
    init()
    rank = get_rank()
    world_size = get_group_size()

    # 构造输入数据，shape为(world_size,)，值为0~N-1
    input_tensor = Tensor(np.arange(world_size, dtype=np.float32))
    print("[Rank %d] Input: %s" % (rank, input_tensor))

    # 调用 AllReduce 集合通信算子，HCCL 后端执行求和
    output = ops.AllReduce()(input_tensor)
    print("[Rank %d] Output: %s" % (rank, output))

    # 期望输出：每个元素为所有 rank 对应位置数据之和，即 world_size * (0~N-1)
    expected = np.arange(world_size, dtype=np.float32) * world_size
    if not np.allclose(output.asnumpy(), expected):
        raise RuntimeError(
            "[Rank %d] AllReduce result check failed, expected: %s, actual: %s"
            % (rank, expected, output.asnumpy())
        )
    print("[Rank %d] AllReduce test passed." % rank)


if __name__ == "__main__":
    main()
