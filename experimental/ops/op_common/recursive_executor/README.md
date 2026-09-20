# Recursive Executor

简体中文 | [English](./README_en.md)

> 试验性递归执行器：以算法描述树 + 通用解释器替代硬编码调度链，为 HCCL 集合通信算子提供统一、可组合、可递归的执行框架。

本模块是 HCCL 架构重构 RFC（[issue #2](https://gitcode.com/luyang20/hccl/issues/2)）的参考实现，置于 `experimental/` 下，不编入商用版本，不保证兼容性。

---

## 目录

- [动机](#动机)
- [设计](#设计)
- [用法](#用法)
- [现状](#现状)
- [限制](#限制)

---

## 动机

### 当前架构的问题

HCCL 现有算子实现存在以下结构性问题：

1. **调度逻辑硬编码**：每个算子的多层级通信流程（如 AllGather 的 server→super-pod→cross-super-pod）以串行代码写死在算子内部，无法复用，新增算子需重写整条链路。
2. **算法与执行耦合**：算法选择、数据切分、传输执行混在同一函数中，难以单独测试或替换。
3. **新算子成本高**：新增一个算子需理解全链路代码，复制粘贴大量样板逻辑，维护负担重。
4. **多层级拓扑支持碎片化**：不同层级的通信原语分散在多处，缺乏统一编排。
5. **流水线重叠难以实现**：数据搬运与计算的重叠（overlap）需要在算子层手写复杂状态机，难以泛化。
6. **可测试性差**：调度逻辑嵌入算子内部，无法对单层通信进行独立单元测试。

### 驱动因素：四级拓扑

昇腾大规模训练集群的网络拓扑天然分四级：

| 层级     | 拓扑类型            | 说明                                 |
| ------ | --------------- | ---------------------------------- |
| layer0 | 服务器内互联          | 服务器内节点间通信                           |
| layer1 | 跨服务器互联           | 服务器间节点通信                            |
| layer2 | 跨 super-pod 互联    | super-pod 间节点通信                     |
| layer3 | 跨 super-pod 互联    | 跨 super-pod 节点通信                     |

当前架构为每级拓扑写独立调度代码，无法组合。递归执行器以**统一算法树**描述多级拓扑流程，将每级通信封装为 Template 叶节点，通过递归编排自动组合。

### 目标

- **统一调度**：用一棵算法描述树表达任意多层级通信流程
- **可组合**：算法节点可嵌套子算法，实现层级递归
- **可扩展**：新算子只需组装算法树 + 实现 Template，无需改动执行器
- **可测试**：每个 Template/CommPlanner 可独立测试
- **流水线友好**：执行策略（SEQUENCE/PARALLEL/OMNIPIPE）作为算法描述的一部分，执行器统一处理

---

## 设计

### 总体架构

```text
┌─────────────────────────────────────────────────────┐
│                  AlgSelector (注册表)                 │
│  算子 → HcclAlgorithm (静态算法描述：算法树 + 参数映射)     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│                  OpsExecutor (通用解释器)             │
│  递归遍历算法树 → 按策略编排 → 调用 Template 执行       │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Template (单层执行单元)                   │
│  PreCopy → RunAlgorithm(CommPlanner) → SendAll → PostCopy │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              CommPlanner (通信计划生成器)                │
│  计算通信对端、数据切片、传输方向 → 调用 HCOMM 传输      │
└─────────────────────────────────────────────────────┘
```

四层职责分离：

| 层次 | 组件                     | 职责                                              |
| -- | ---------------------- | ----------------------------------------------- |
| L1 | AlgSelector / HcclAlgorithm | 算法静态描述：执行策略树 + 数据参数映射                           |
| L2 | OpsExecutor            | 通用递归解释器：遍历算法树，按策略编排子节点                          |
| L3 | Template               | 单层执行框架：PreCopy → CommPlanner → SendAll → PostCopy |
| L4 | CommPlanner              | 通信计划：计算 peers、数据切片、传输方向                         |

### 核心数据结构

#### AlgoExecDesc — 递归算法树节点

```cpp
// inc/algo_desc.h

enum class HcclAlgExecPolicy {
    SEQUENCE,   // 子节点串行执行，前序输出 = 后序输入
    PARALLEL,   // 子节点并行执行，数据切分后各管一片
    OMNIPIPE,   // 流水线重叠：边收边算边发
};

// 叶节点：指向一个 Template 执行描述
struct TemplateExecDesc {
    TemplateDesc templateDesc;  // Template 标识 + 参数
    int subCommIndex = 0;       // 子通信域索引
    int netLayer = -1;          // 网络层索引，-1=遍历所有层取首个匹配
};

// 递归节点：自身也是一棵算法树
struct AlgoExecDesc;
using VariantType = std::variant<TemplateExecDesc, std::shared_ptr<AlgoExecDesc>>;

struct AlgoExecDesc {
    HcclAlgExecPolicy execPolicy = HcclAlgExecPolicy::SEQUENCE;
    std::vector<VariantType> children;
    std::vector<u32> dataSplitRatio;   // 并行数据切分比例
    u32 subCommMask{0};               // 子通信域位掩码，用于同步
    OmniPipeXYdata omniPipeXYdata;    // OmniPipe 流水线参数
};
```

`VariantType` 是 `TemplateExecDesc`（叶）或 `std::shared_ptr<AlgoExecDesc>`（子树）的变体（`std::variant`），支持任意深度的递归嵌套。

#### DataParams — 统一数据参数

```cpp
// inc/data_types.h

struct DataParams {
    void* inputBufferPtr = nullptr;   // 输入数据指针
    void* outputBufferPtr = nullptr;  // 输出数据指针
    void* cclBufferPtr = nullptr;     // CCL 中间缓冲区
    BufferType inputBufferType = BufferType::INPUT;
    BufferType outputBufferType = BufferType::OUTPUT;
    BufferType cclBufferType = BufferType::HCCL_BUFFER;
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    u64 dataOffset{0};       // Loop 在用户内存中的起始偏移
    u64 sliceCount{0};       // 当前处理的切片元素数
    u64 sliceOffset{0};      // Parallel 子片偏移
    u64 tailCount{0};        // 末尾不整除的余量
    u32 globalTailRankId{INVALID_VALUE_RANKID};  // 末尾 rank ID
    u64 dataStride{0};       // 用户内存相邻 Slot 间距
    u64 scratchStride{0};    // CCL Buffer 相邻 Slot 间距
    HcclReduceOp reduceOp{HCCL_REDUCE_RESERVED};
    u32 root{INVALID_VALUE_RANKID};
    bool enableRemoteMemAccess{false};  // OFFLOAD 模式标志
    u32 userRankSize{0};               // 全局 rank 总数，用于校验映射越界
    std::vector<u32> ranksForInputData;  // 当前 Buffer 各 Slot 的归属 rank
    // ...
};
```

统一内存模型：`Input → CCL Buffer → ... → CCL Buffer → Output`

每个 Template 从 `DataParams` 取输入、写输出到 `CCL Buffer`，下一个 Template 从同一 `CCL Buffer` 读输入，形成流水线。

### 执行策略

| 策略           | 语义      | 数据流                                                      |
| ------------ | ------- | -------------------------------------------------------- |
| **SEQUENCE** | 子节点串行执行 | 前序子节点的 `ranksForOutputData` = 后序子节点的 `ranksForInputData` |
| **PARALLEL** | 子节点并行执行 | 数据按切分策略分配给各子节点独立处理                                       |
| **OMNIPIPE** | 流水线重叠   | 边收边算边发，最小化 Bubble 开销；子树须同为 OMNIPIPE 策略，恰好 2 个子节点，支持递归嵌套   |

`ranksForInputData` / `ranksForOutputData` 是连接各阶段的数据所有权契约，确保 SEQUENCE 下前后阶段数据一致。

### 算法组装示例

以 AllGather 四级拓扑为例，组装为一棵扁平 SEQUENCE 树（见 `experimental/ops/all_gather/all_gather.cc`）：

```text
AlgoExecDesc(execPolicy=SEQUENCE)
├── TemplateExecDesc(Mesh, subCommIndex=3)  // layer3: 跨 super-pod
├── TemplateExecDesc(NHR,  subCommIndex=2)  // layer2: 跨 super-pod NHR
├── TemplateExecDesc(NHR,  subCommIndex=1)  // layer1: 跨服务器 NHR
└── TemplateExecDesc(Mesh, subCommIndex=0)  // layer0: server 内 Mesh
```

### OpsExecutor 递归编排

```cpp
// executor/ops_executor.cc
// 注意：以下为伪代码，仅示意递归编排逻辑，非真实签名

void OpsExecutor::OrchestrateLoop(const AlgoExecDesc& hcclAlgorithm, const DataParams& params) {
    switch (hcclAlgorithm.policy) {
        case SEQUENCE:
            // 串行执行各子节点，传递 ranksForOutputData → ranksForInputData
            for (auto& child : hcclAlgorithm.children) {
                if (child.isTemplate()) {
                    ExecuteTemplate(child.asTemplate(), params);
                } else {
                    OrchestrateLoop(child.asHcclAlgorithm(), params);  // 递归
                }
            }
            break;
        case PARALLEL:
            // 数据切分后并行执行
            break;
        case OMNIPIPE:
            // 流水线重叠编排：子树须同为 OMNIPIPE，恰好 2 个子节点，
            // 基于 OmniPipeXYdata 计算步数与数据切片，递归编排子树
            break;
    }
}
```

### Template / CommPlanner 分离

**Template** 负责执行框架，固定流程为：

```text
PreCopy → RunAlgorithm(CommPlanner) → SendAll → PostCopy
```

- **PreCopy**：将输入数据搬到 CCL Buffer
- **RunAlgorithm**：调用 CommPlanner 执行实际通信
- **SendAll**：将 CCL Buffer 数据分发到各 rank
- **PostCopy**：将 CCL Buffer 数据搬到输出

**CommPlanner** 负责通信计划生成：计算通信对端（peers）、数据切片（slices）、传输方向（TransferDirection），然后调用 HCOMM 传输接口。

这种分离使 Template 只关心执行框架，CommPlanner 只关心通信细节，两者可独立开发和测试。

### 四级拓扑匹配

`TopoMatchFourLevel`（`topo/topo_match_four_level.h`）实现四级对称拓扑匹配：

- 检测当前 rank 在四级拓扑中的位置
- 为每级选择对应的 Template（具体通信原语由实际拓扑决定）
- 当前实现要求拓扑对称（各 rank 视角一致）

---

## 用法

### 编译

本模块作为 `RecursiveExecutor` OBJECT 库编入主 HCCL 构建，由顶层 CMakeLists 控制：

```bash
# 主仓构建即可编译本模块
bash build.sh --pkg
```

CMake 配置（`CMakeLists.txt`）要点：

- 源文件列表 `RE_CORE_SRC` 包含 14 个 `.cc` 文件
- 以 OBJECT 库形式链接到 `libhccl.so`
- 定义编译宏 `ENABLE_EXPERIMENTAL`
- include 路径：`src/`、`inc/`、CANN 安装目录
- 从 `cann_version.h` 读取 CANN 版本号

### 运行期开关

本特性通过运行期开关 `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR=true` 控制是否生效（见 [experimental/README.md](../../../README.md#5-运行期开关)）。

开关函数 `IsRecursiveExecutorEnabled()` 定义在 `executor/adaptor_executor.cc` 中。开关位于注册阶段而非调用阶段：`REGISTER_ALG` 宏在静态初始化时检查该开关，开关关闭时不注册算法与执行器，selector 自然无法选中该算法，不会进入执行路径。

- 常量值默认 `false`（编译期关闭），需修改 `IsRecursiveExecutorEnabled()` 中的 `constexpr bool recursiveExecutorEnabled` 为 `true` 后重新编译，方可通过环境变量在运行期开启。
- 运行期启用方式：设置环境变量 `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR=true`。

### 注册新算法

通过 `REGISTER_ALG` 宏注册算法到 `AlgSelector` 单例：

```cpp
// experimental/ops/all_gather/all_gather.cc — 现有注册示例

static HcclAlgorithm MakeAicpuAllGatherSequenceMeshNHRNHRMesh()
{
    HcclAlgorithm algo;
    algo.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    algo.engineType  = HcclAlgEngineType::COMM_ENGINE_AICPU;
    algo.topoMatch   = std::make_shared<TopoMatchFourLevel>();
    algo.algoExecDesc = MakeAllGather4LevelAlgoExecDesc();
    algo.algName     = "AicpuAllGatherSequenceMeshNHRNHRMesh";
    return algo;
}

REGISTER_ALG(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherSequenceMeshNHRNHRMesh,
    MakeAicpuAllGatherSequenceMeshNHRNHRMesh());
```

`MakeAllGather4LevelAlgoExecDesc()` 负责组装算法树（`AlgoExecDesc`），`REGISTER_ALG` 一步完成算法入 `AlgSelector` 和执行器入 `CollAlgExecRegistryV2`，两表以同一算法名关联。

### 添加新 Template

1. 在 `template/aicpu/` 下新建 Template 类，继承 `AicpuBaseTemplate`
2. 实现 `RunAlgorithm`（纯虚），按需重写 `PreCopy`、`PostCopy` 方法，以及 `SendAll` 定制化 hook（`BuildTransferContext`/`CanParallelPostCopy`/`LaunchPostCopy`）
3. 用 `REGISTER_RE_TEMPLATE` 宏注册 Template 类到工厂表

```cpp
// template/aicpu/allgather_mesh_template.cc

REGISTER_RE_TEMPLATE(
    HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, AllGatherMeshTemplate)
```

`REGISTER_RE_TEMPLATE` 将 Template 类注册进 `TemplateRegistry` 单例，框架通过 `GetTemplate()` 按 `TemplateDesc` 查表创建实例，新增 Template 时调用此宏即可，无需修改 `GetTemplate` 本身。

### 添加新 CommPlanner

1. 在 `template/comm_planners/` 下新建 CommPlanner 实现文件
2. 实现 `RunMeshXxx()` / `RunNhrXxx()` 等函数，计算 peers + 数据切片 + 传输方向
3. 在对应 Template 的 `RunAlgorithm()` 中调用

现有 CommPlanner：

| CommPlanner          | 文件                              | 功能                                |
| ------------------ | ------------------------------- | --------------------------------- |
| `RunMeshAllGather` | `comm_planners/mesh_comm_planner.cc` | Mesh 拓扑 AllGather                 |
| `RunNhrAllGather`  | `comm_planners/nhr_comm_planner.cc`  | NHR 拓扑 AllGather（递归减半算法） |

### 添加新算子

1. 在 `experimental/ops/<op>/` 下新建 `<op_name>.cc`（如 `all_gather.cc`）
2. 组装算法树 `AlgoExecDesc`，设置执行策略和数据参数映射
3. 用 `REGISTER_ALG` 注册
4. （可选）添加新 Template / CommPlanner

---

## 现状

### 实现进度：Phase 1 / 3

RFC 规划分三阶段推进，当前处于第一阶段：

| 阶段      | 目标                                   | 状态    |
| ------- | ------------------------------------ | ----- |
| Phase 1 | 核心框架 + AllGather + AICPU 引擎 + 四级对称拓扑 + SEQUENCE/PARALLEL/OMNIPIPE 三策略 | ✅ 已实现 |
| Phase 2 | 多算子覆盖 + 多引擎（AIV/CCU）   | ⏳ 规划中 |
| Phase 3 | 非对称拓扑 + 生产集成          | ⏳ 规划中 |

### 已实现

- **核心框架**：`AlgoExecDesc` 递归算法树、`OpsExecutor` 通用递归解释器、`DataParams` 统一数据模型
- **执行策略**：
  - **SEQUENCE**：子节点串行执行，前序输出 = 后序输入
  - **PARALLEL**：子节点并行执行，按 `dataSplitRatio` 数据切分，支持前后子通信域同步
  - **OMNIPIPE**：流水线重叠编排，基于 `OmniPipeXYdata` 计算步数与数据切片，支持二维带宽建模
- **算法注册**：`AlgSelector` 单例 + `REGISTER_ALG` 宏（`experimental/ops/all_gather/all_gather.cc` 已实例化注册 4 级 AllGather，受运行期开关 `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR` 守卫，开关默认关闭）
- **Template 实现**：
  - `AllGatherMeshTemplate`（`template/aicpu/allgather_mesh_template.cc`）— Mesh AllGather，支持 DirectToOutput 模式
  - `AllGatherNhrTemplate`（`template/aicpu/allgather_nhr_template.cc`）— NHR AllGather，并行 PostCopy DMA 优化 + 末步直写输出优化
- **CommPlanner 实现**：
  - `RunMeshAllGather`（`template/comm_planners/mesh_comm_planner.cc`）
  - `RunNhrAllGather`（`template/comm_planners/nhr_comm_planner.cc`）— 递归减半算法
- **拓扑匹配**：`TopoMatchFourLevel`（`topo/topo_match_four_level.cc`）— 四级对称拓扑
- **执行器适配**：`AdaptorExecutorBase`（`executor/adaptor_executor.cc`）— 桥接 HCCL 框架，`REGISTER_ALG` 宏接入
- **OmniPipe 工具**：`OmniPipeXYdata` 数据结构 + 二维带宽建模 + 数据切片计算（`executor/omnipipe_utils.h` / `.cc`），已接入 `OrchestrateOmniPipeLoop` 主流程

### 未实现 / 规划中

- **多引擎**：仅 AICPU，AIV（AI Core Vector）/ CCU 未实现
- **多算子**：仅 AllGather，AllReduce / Broadcast / ReduceScatter / AlltoAll 等未实现
- **非对称拓扑**：仅支持对称四级拓扑
- **跨 super-pod CommPlanner**：layer3 的跨 super-pod 通信原语未实现

---

## 限制

1. **试验性，不编入商用版本**：本模块置于 `experimental/`，默认不参与生产构建，不保证兼容性，API 可能随时变更。
2. **单一引擎**：仅实现 AICPU 引擎的 Template，不支持 AIV / CCU 引擎。
3. **单一算子**：仅注册 AllGather 算法，不支持 AllReduce / Broadcast / ReduceScatter / AlltoAll / Send / Recv 等。
4. **执行策略覆盖有限**：SEQUENCE、PARALLEL、OMNIPIPE 三种策略均已实现，但 OMNIPIPE 要求子树全为 OMNIPIPE 策略且恰好 2 个子节点，当前已注册算法（AllGather）使用 SEQUENCE 策略，OMNIPIPE 尚未在已注册算子中实际使用。
5. **对称拓扑要求**：`TopoMatchFourLevel` 要求四级拓扑对称（各 rank 视角一致），不支持非对称拓扑。
6. **测试覆盖有限**：UT 已覆盖 5 组核心模块（omnipipe_utils/data_ops/comm_planner/algo_desc/data_transfer），ST 尚无覆盖。

---

## 参考

- experimental/ 约定：[experimental/README.md](../../../README.md)
- HCCL架构简介（中文）：[docs/zh/architecture/architecture-brief.md](../../../../docs/zh/architecture/architecture-brief.md)
