# 主流框架集成

## 场景说明

HCCL在系统中的位置如下图所示。

![HCCL在系统中位置](figures/hccl_location.png)

AI框架主要有三种编程执行形态，单算子模式、图模式（Ascend IR）和图捕获模式（aclgraph），因此HCCL也提供了对应的工作方式。

- 单算子模式与图捕获模式（aclgraph）下，AI框架直接调用HCCL的C语言接口，下发通信算子到加速引擎执行，关于HCCL通信算子API的详细介绍可参见[通信算子](../api_ref/comm_op_interface/README.md)。
- 图模式（Ascend IR）下，AI框架使用Ascend算子IR将模型的计算过程构造成一张图，通过Graph Engine（简称GE）将图中的通信算子下发给加速引擎执行，关于图模式的详细介绍，可参见《[图开发指南](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/graph_dev/README.md)》，Ascend IR的定义可参见《[算子库接口参考](https://hiascend.com/document/redirect/CannCommunityOplist)》中的“Ascend IR算子规格说明”。

针对PyTorch和MindSpore框架，HCCL的调用已集成到TorchNPU和MindSpore框架代码中，开发者指定使用HCCL作为分布式后端，直接使用框架原生通信API，即可实现分布式能力，详细使用方法可参见《[TorchNPU 产品文档](https://hiascend.com/document/redirect/pytorchuserguide)》和[MindSpore官网](https://www.mindspore.cn/)。

针对TensorFlow框架，HCCL通过TensorFlow适配插件TF Adapter对接TensorFlow框架，详细使用方法可参见《[TensorFlow 模型迁移指南](https://hiascend.com/document/redirect/tfcommunityguide)》。

针对vLLM、SGLang等推理引擎，其在昇腾上的分布式推理（张量并行TP、专家并行EP、数据并行DP、上下文并行CP）同样以HCCL作为通信底座，但集成适配由引擎侧的适配工程承担（如《[vllm-ascend 文档](https://vllm-ascend.readthedocs.io/en/latest/)》），无需开发者直接对接HCCL。以vllm-ascend为例，其对HCCL的消费方式包括：通过TorchNPU以`backend="hccl"`使用`torch.distributed`；通过ctypes直接调用`libhccl.so`的C接口实现流感知的精细控制；MoE场景通过MC2（Kernel Fusion Custom）融合通信算子框架执行AlltoAll。KV Cache跨节点传输（PD分离场景）由昇腾单边通信库HIXL（Huawei Xfer Library，参见《[HIXL 仓库介绍](https://gitcode.com/cann/hixl)》）承担，与HCCL集合通信分工互补。推理部署中常用的HCCL环境变量（如`HCCL_BUFFSIZE`、`HCCL_OP_EXPANSION_MODE`）可参见[环境变量参考](./hccl_env/README.md)。

## 示例代码

- [PyTorch框架调用](../../../examples/03_ai_framework/01_pytorch)
- [TensorFlow框架调用](../../../examples/03_ai_framework/02_tensorflow)
- [MindSpore框架调用](../../../examples/03_ai_framework/03_mindspore)
