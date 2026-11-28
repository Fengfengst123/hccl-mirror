# Mainstream Framework Integration

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:30:00.213Z pushedAt=2026-11-12T08:53:57.216Z nl=2 -->

## Scenario Description

The position of HCCL in the system is shown in the following figure.

![Position of HCCL in the System](figures/hccl_location.png)

AI frameworks primarily have three programming execution modes: single-operator mode, graph mode (Ascend IR), and graph capture mode (aclgraph). Accordingly, HCCL provides corresponding working modes for each.

- In single-operator mode and graph capture mode (aclgraph), the AI framework directly calls the C language APIs of HCCL to dispatch communication operators to the acceleration engine for execution. For details about HCCL communication operator APIs, see [Communication Operators](../api_ref/comm_op_interface/README.md).
- In graph mode (Ascend IR), the AI framework uses Ascend operator IR to construct the computation process of a model into a graph, and dispatches communication operators in the graph to the acceleration engine for execution through Graph Engine (GE). For details about graph mode, see *[Graph Development Guide](https://gitcode.com/cann/ge/blob/master/docs/en/user_guides/graph_dev/README.md)*. For the definition of Ascend IR, see "Ascend IR Operator Specification" in *[Operator Library Interface Reference](https://hiascend.com/en/document/redirect/CannCommunityOplist)*.

For PyTorch and MindSpore frameworks, HCCL invocation has been integrated into the TorchNPU and MindSpore framework code. Developers can specify HCCL as the distributed backend and directly use the framework's native communication APIs to implement distributed capabilities. For detailed usage, see *[TorchNPU Product Documentation](https://hiascend.com/en/document/redirect/pytorchuserguide)* and the [MindSpore official website](https://www.mindspore.cn/en).

For the TensorFlow framework, HCCL interfaces with TensorFlow through the TensorFlow adapter plugin TF Adapter. For detailed usage, see *[TensorFlow Model Migration Guide](https://hiascend.com/en/document/redirect/tfcommunityguide)*.

For inference engines such as vLLM and SGLang, distributed inference on Ascend (tensor parallelism [TP], expert parallelism [EP], data parallelism [DP], and context parallelism [CP]) also uses HCCL as the communication backbone, but the integration is handled by engine-side adaptation projects (such as *[vllm-ascend documentation](https://vllm-ascend.readthedocs.io/en/latest/)*), so developers do not need to interface with HCCL directly. Taking vllm-ascend as an example, HCCL is consumed in three ways: through TorchNPU with `backend="hccl"` via `torch.distributed`; through ctypes direct calls to the C APIs of `libhccl.so` for stream-aware fine-grained control; and through the MC2 (Kernel Fusion Custom) custom operator framework for AlltoAll in MoE scenarios. Cross-node KV Cache transfer (PD disaggregation) is handled by the unilateral communication library HIXL (Huawei Xfer Library; see the [HIXL repository](https://gitcode.com/cann/hixl)), which complements HCCL collective communication. Common HCCL environment variables for inference deployment (such as `HCCL_BUFFSIZE` and `HCCL_OP_EXPANSION_MODE`) are described in the [environment variable reference](./hccl_env/README.md).

## Sample Code

- [PyTorch Framework Invocation](../../../examples/03_ai_framework/01_pytorch)
- [TensorFlow Framework Invocation](../../../examples/03_ai_framework/02_tensorflow)
- [MindSpore Framework Invocation](../../../examples/03_ai_framework/03_mindspore)
