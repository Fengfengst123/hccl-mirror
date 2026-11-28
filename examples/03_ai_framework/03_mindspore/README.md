# 使用 MindSpore 执行 AllReduce 操作

## 样例介绍

本样例展示如何使用 MindSpore 接口执行 AllReduce 操作，包含以下功能点：

- 通过 `mindspore.set_context()` 接口指定运行设备与模式。
- 通过 `mindspore.communication.init()` 接口初始化通信域（底层由 HCCL 承载集合通信）。
- 通过 `ops.AllReduce()` 算子执行 AllReduce 操作。

## 环境准备

### 环境要求

本样例支持以下产品，组网为单机N卡（N>=2）：

- Atlas A3系列产品

本样例依赖以下 Python 包：

- `mindspore`：华为开源的全场景AI框架，其分布式通信能力在昇腾硬件上由 HCCL 承载。

### 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh
```

## 执行样例

本样例每个进程对应一张卡，推荐使用 MindSpore 启动器 `msrun` 拉起，以 8 卡为例：

```bash
msrun --worker_num=8 hccl_mindspore_allreduce_test.py
```

设备号由启动器自动注入。单进程调试请同样使用启动器拉起（直接运行脚本会因缺少 `RANK_ID` 等必备环境变量在 `init()` 处报错）：

```bash
msrun --worker_num=1 --local_worker_num=1 hccl_mindspore_allreduce_test.py
```

脚本内置期望值校验，所有进程输出 `AllReduce test passed` 即为通过。

> 注意：可通过设置 `HCCL_OP_EXPANSION_MODE` 环境变量配置通信算子的展开模式，不同产品型号支持的范围可参考[环境变量列表](https://hiascend.com/document/redirect/CannCommunityEnvRef)中该环境变量的使用方法。
>
> ```bash
> # 设置通信算子的展开模式为AI CPU通信引擎
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## 结果示例

每个 rank 的数据初始化为 0~7，经过 AllReduce 操作后，每个 rank 的结果是所有 rank 对应位置数据的和（8 个 rank 的数据相加）。

```
[Rank 0] Input: [0. 1. 2. 3. 4. 5. 6. 7.]
[Rank 1] Input: [0. 1. 2. 3. 4. 5. 6. 7.]
...
[Rank 0] Output: [ 0.  8. 16. 24. 32. 40. 48. 56.]
[Rank 1] Output: [ 0.  8. 16. 24. 32. 40. 48. 56.]
...
```
