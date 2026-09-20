# Recursive Executor

English | [简体中文](./README.md)

> An experimental recursive executor that replaces hard-coded dispatch chains with an algorithm description tree + a generic interpreter, providing a unified, composable, and recursive execution framework for HCCL collective communication operators.

This module is the reference implementation of the HCCL architecture refactoring RFC ([issue #2](https://gitcode.com/luyang20/hccl/issues/2)). It resides under `experimental/` and is not included in production builds. Compatibility is not guaranteed.

---

## Table of Contents

- [Motivation](#motivation)
- [Design](#design)
- [Usage](#usage)
- [Status](#status)
- [Limitations](#limitations)

---

## Motivation

### Problems with the Current Architecture

The existing HCCL operator implementation has the following structural issues:

1. **Hard-coded dispatch logic**: The multi-level communication flow for each operator (e.g., AllGather's server→super-pod→cross-super-pod) is hard-coded as sequential code inside the operator. It cannot be reused, and adding a new operator requires rewriting the entire chain.
2. **Algorithm-execution coupling**: Algorithm selection, data partitioning, and transfer execution are mixed within the same function, making them difficult to test or replace independently.
3. **High cost of new operators**: Adding a new operator requires understanding the full-chain code and copying large amounts of boilerplate logic, creating a heavy maintenance burden.
4. **Fragmented multi-level topology support**: Communication planners for different levels are scattered across multiple places, lacking unified orchestration.
5. **Difficult pipeline overlap**: Overlap between data transfer and computation requires hand-written complex state machines at the operator level, which is hard to generalize.
6. **Poor testability**: Dispatch logic is embedded inside operators, making it impossible to unit-test individual communication levels.

### Driving Factor: Four-Level Topology

Ascend large-scale training clusters have a naturally four-level network topology:

| Level  | Topology Type      | Description                          |
| ------ | ------------------ | ------------------------------------ |
| layer0 | intra-server interconnect | Intra-server node communication        |
| layer1 | cross-server interconnect | Inter-server node communication        |
| layer2 | cross-super-pod interconnect | Inter-super-pod node communication |
| layer3 | cross-super-pod interconnect | Cross-super-pod node communication  |

The current architecture writes independent dispatch code for each topology level, preventing composition. The recursive executor uses a **unified algorithm tree** to describe multi-level topology flows, encapsulating each level's communication as a Template leaf node and automatically combining them through recursive orchestration.

### Goals

- **Unified dispatch**: Express arbitrary multi-level communication flows with a single algorithm description tree
- **Composable**: Algorithm nodes can nest sub-algorithms, enabling hierarchical recursion
- **Extensible**: New operators only need to assemble an algorithm tree + implement Templates, without modifying the executor
- **Testable**: Each Template/CommPlanner can be tested independently
- **Pipeline-friendly**: Execution strategies (SEQUENCE/PARALLEL/OMNIPIPE) are part of the algorithm description, handled uniformly by the executor

---

## Design

### Overall Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  AlgSelector (Registry)              │
│  Operator → HcclAlgorithm (static algorithm description:  │
│             algorithm tree + parameter mapping)      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│                  OpsExecutor (Generic Interpreter)   │
│  Recursively traverse algorithm tree → orchestrate  │
│  by strategy → invoke Template execution             │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Template (Single-Level Execution Unit)  │
│  PreCopy → RunAlgorithm(CommPlanner) → SendAll → PostCopy │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              CommPlanner (Communication Plan Generator)│
│  Compute peers, data slices, transfer direction     │
│  → invoke HCOMM transfer                             │
└─────────────────────────────────────────────────────┘
```

Four-layer separation of concerns:

| Layer | Component               | Responsibility                                          |
| ----- | ----------------------- | ------------------------------------------------------- |
| L1    | AlgSelector / HcclAlgorithm  | Static algorithm description: execution strategy tree + data parameter mapping |
| L2    | OpsExecutor              | Generic recursive interpreter: traverses algorithm tree, orchestrates child nodes by strategy |
| L3    | Template                | Single-level execution framework: PreCopy → CommPlanner → SendAll → PostCopy |
| L4    | CommPlanner               | Communication plan: compute peers, data slices, transfer direction |

### Core Data Structures

#### AlgoExecDesc — Recursive Algorithm Tree Node

```cpp
// inc/algo_desc.h

enum class HcclAlgExecPolicy {
    SEQUENCE,   // Children execute sequentially, prior output = next input
    PARALLEL,   // Children execute in parallel, data partitioned among them
    OMNIPIPE,   // Pipeline overlap: receive, compute, and send concurrently
};

// Leaf node: points to a Template execution description
struct TemplateExecDesc {
    TemplateDesc templateDesc;  // Template identifier + parameters
    int subCommIndex = 0;       // Sub-communicator index
    int netLayer = -1;          // Network layer index, -1 = iterate all layers, take first match
};

// Recursive node: itself is an algorithm tree
struct AlgoExecDesc;
using VariantType = std::variant<TemplateExecDesc, std::shared_ptr<AlgoExecDesc>>;

struct AlgoExecDesc {
    HcclAlgExecPolicy execPolicy = HcclAlgExecPolicy::SEQUENCE;
    std::vector<VariantType> children;
    std::vector<u32> dataSplitRatio;   // Parallel data split ratio
    u32 subCommMask{0};               // Sub-communicator bitmask for synchronization
    OmniPipeXYdata omniPipeXYdata;    // OmniPipe pipeline parameters
};
```

`VariantType` is a variant (`std::variant`) of `TemplateExecDesc` (leaf) or `std::shared_ptr<AlgoExecDesc>` (subtree), supporting arbitrary-depth recursive nesting.

#### DataParams — Unified Data Parameters

```cpp
// inc/data_types.h

struct DataParams {
    void* inputBufferPtr = nullptr;   // Input data pointer
    void* outputBufferPtr = nullptr;  // Output data pointer
    void* cclBufferPtr = nullptr;     // CCL intermediate buffer
    BufferType inputBufferType = BufferType::INPUT;
    BufferType outputBufferType = BufferType::OUTPUT;
    BufferType cclBufferType = BufferType::HCCL_BUFFER;
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    u64 dataOffset{0};       // Start offset in user memory for looping
    u64 sliceCount{0};       // Element count of current slice
    u64 sliceOffset{0};      // Parallel sub-slice offset
    u64 tailCount{0};        // Remainder for non-divisible tail
    u32 globalTailRankId{INVALID_VALUE_RANKID};  // Tail rank ID
    u64 dataStride{0};       // Spacing between adjacent slots in user memory
    u64 scratchStride{0};    // Spacing between adjacent slots in CCL buffer
    HcclReduceOp reduceOp{HCCL_REDUCE_RESERVED};
    u32 root{INVALID_VALUE_RANKID};
    bool enableRemoteMemAccess{false};  // OFFLOAD mode flag
    u32 userRankSize{0};               // Global rank count for bounds checking
    std::vector<u32> ranksForInputData;  // Owning rank of each slot in current buffer
    // ...
};
```

Unified memory model: `Input → CCL Buffer → ... → CCL Buffer → Output`

Each Template reads input from `DataParams`, writes output to `CCL Buffer`, and the next Template reads input from the same `CCL Buffer`, forming a pipeline.

### Execution Strategies

| Strategy       | Semantics               | Data Flow                                                          |
| -------------- | ----------------------- | ------------------------------------------------------------------ |
| **SEQUENCE**  | Children execute sequentially | Prior child's `ranksForOutputData` = next child's `ranksForInputData` |
| **PARALLEL**  | Children execute in parallel | Data is partitioned to each child for independent processing       |
| **OMNIPIPE**  | Pipeline overlap         | Receive, compute, and send concurrently; subtree must all be OMNIPIPE, exactly 2 children, supports recursive nesting |

`ranksForInputData` / `ranksForOutputData` are data ownership contracts connecting stages, ensuring data consistency across SEQUENCE stages.

### Algorithm Assembly Example

Using AllGather with four-level topology as an example, assembled as a flat SEQUENCE tree (see `experimental/ops/all_gather/all_gather.cc`):

```text
AlgoExecDesc(execPolicy=SEQUENCE)
├── TemplateExecDesc(Mesh, subCommIndex=3)  // layer3: cross-super-pod
├── TemplateExecDesc(NHR,  subCommIndex=2)  // layer2: cross-super-pod NHR
├── TemplateExecDesc(NHR,  subCommIndex=1)  // layer1: cross-server NHR
└── TemplateExecDesc(Mesh, subCommIndex=0)  // layer0: intra-server Mesh
```

### OpsExecutor Recursive Orchestration

```cpp
// executor/ops_executor.cc
// Note: The following is pseudocode, illustrating recursive orchestration logic only, not the real signature.

void OpsExecutor::OrchestrateLoop(const AlgoExecDesc& hcclAlgorithm, const DataParams& params) {
    switch (hcclAlgorithm.policy) {
        case SEQUENCE:
            // Execute children sequentially, passing ranksForOutputData → ranksForInputData
            for (auto& child : hcclAlgorithm.children) {
                if (child.isTemplate()) {
                    ExecuteTemplate(child.asTemplate(), params);
                } else {
                    OrchestrateLoop(child.asHcclAlgorithm(), params);  // Recurse
                }
            }
            break;
        case PARALLEL:
            // Partition data and execute in parallel
            break;
        case OMNIPIPE:
            // Pipeline overlap: subtree must all be OMNIPIPE, exactly 2 children,
            // computes step count and data slices via OmniPipeXYdata, recursively orchestrates subtree
            break;
    }
}
```

### Template / CommPlanner Separation

**Template** is responsible for the execution framework, with a fixed flow:

```text
PreCopy → RunAlgorithm(CommPlanner) → SendAll → PostCopy
```

- **PreCopy**: Move input data to CCL Buffer
- **RunAlgorithm**: Invoke CommPlanner to execute actual communication
- **SendAll**: Distribute CCL Buffer data to each rank
- **PostCopy**: Move CCL Buffer data to output

**CommPlanner** is responsible for communication plan generation: computing communication peers, data slices, and transfer direction, then invoking HCOMM transfer interfaces.

This separation ensures that Template only concerns itself with the execution framework, while CommPlanner only concerns communication details. Both can be developed and tested independently.

### Four-Level Topology Matching

`TopoMatchFourLevel` (`topo/topo_match_four_level.h`) implements four-level symmetric topology matching:

- Detects the current rank's position in the four-level topology
- Selects the corresponding Template for each level (the specific communication primitive depends on the actual topology)
- The current implementation requires symmetric topology (consistent view across all ranks)

---

## Usage

### Build

This module is compiled as a `RecursiveExecutor` OBJECT library into the main HCCL build, controlled by the top-level CMakeLists:

```bash
# Building the main repo compiles this module
bash build.sh --pkg
```

CMake configuration (`CMakeLists.txt`) key points:

- Source file list `RE_CORE_SRC` contains 14 `.cc` files
- Linked as an OBJECT library into `libhccl.so`
- Defines compilation macro `ENABLE_EXPERIMENTAL`
- Include paths: `src/`, `inc/`, CANN installation directory
- Reads CANN version from `cann_version.h`

### Runtime Switch

This feature is gated by the runtime switch `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR=true` (see [experimental/README_en.md](../../../README_en.md#5-runtime-switch)).

The switch function `IsRecursiveExecutorEnabled()` is defined in `executor/adaptor_executor.cc`. The switch is applied at registration time rather than call time: the `REGISTER_ALG` macro checks this switch during static initialization. When the switch is off, neither the algorithm nor the executor is registered, so the selector cannot select it and the execution path is never entered.

- The compile-time constant defaults to `false` (disabled). To enable at runtime, change `constexpr bool recursiveExecutorEnabled` in `IsRecursiveExecutorEnabled()` to `true` and rebuild.
- Runtime activation: set the environment variable `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR=true`.

### Registering a New Algorithm

Register algorithms to the `AlgSelector` singleton via the `REGISTER_ALG` macro:

```cpp
// experimental/ops/all_gather/all_gather.cc — existing registration example

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

`MakeAllGather4LevelAlgoExecDesc()` assembles the algorithm tree (`AlgoExecDesc`). `REGISTER_ALG` registers the algorithm into `AlgSelector` and the executor into `CollAlgExecRegistryV2` in one step, with both tables linked by the same algorithm name.

### Adding a New Template

1. Create a new Template class under `template/aicpu/`, inheriting from `AicpuBaseTemplate`
2. Implement `RunAlgorithm` (pure virtual); override `PreCopy`, `PostCopy` as needed, plus `SendAll` customization hooks (`BuildTransferContext`/`CanParallelPostCopy`/`LaunchPostCopy`)
3. Register the Template class to the factory table via the `REGISTER_RE_TEMPLATE` macro

```cpp
// template/aicpu/allgather_mesh_template.cc

REGISTER_RE_TEMPLATE(
    HcclCMDType::HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, AllGatherMeshTemplate)
```

`REGISTER_RE_TEMPLATE` registers the Template class into the `TemplateRegistry` singleton. The framework looks up and creates instances via `GetTemplate()` based on `TemplateDesc`. To add a new Template, simply invoke this macro—no need to modify `GetTemplate` itself.

### Adding a New CommPlanner

1. Create a new CommPlanner implementation file under `template/comm_planners/`
2. Implement `RunMeshXxx()` / `RunNhrXxx()` etc. functions to compute peers + data slices + transfer direction
3. Call from the corresponding Template's `RunAlgorithm()`

Existing CommPlanners:

| CommPlanner          | File                              | Description                                              |
| ------------------ | --------------------------------- | -------------------------------------------------------- |
| `RunMeshAllGather` | `comm_planners/mesh_comm_planner.cc`  | Mesh topology AllGather                                  |
| `RunNhrAllGather`  | `comm_planners/nhr_comm_planner.cc`   | NHR topology AllGather (halving algorithm) |

### Adding a New Operator

1. Create a new `<op_name>.cc` under `experimental/ops/<op>/` (e.g., `all_gather.cc`)
2. Assemble the algorithm tree `AlgoExecDesc`, setting execution strategy and data parameter mapping
3. Register with `REGISTER_ALG`
4. (Optional) Add new Template / CommPlanner

---

## Status

### Implementation Progress: Phase 1 / 3

The RFC is planned in three phases. Currently in Phase 1:

| Phase   | Goal                                                | Status     |
| -------- | --------------------------------------------------- | ---------- |
| Phase 1  | Core framework + AllGather + AICPU engine + four-level symmetric topology + SEQUENCE/PARALLEL/OMNIPIPE strategies | ✅ Implemented |
| Phase 2  | Multi-operator coverage + multi-engine (AIV/CCU) | ⏳ Planned  |
| Phase 3  | Asymmetric topology + production integration | ⏳ Planned  |

### Implemented

- **Core framework**: `AlgoExecDesc` recursive algorithm tree, `OpsExecutor` generic recursive interpreter, `DataParams` unified data model
- **Execution strategies**:
  - **SEQUENCE**: Children execute sequentially, prior output = next input
  - **PARALLEL**: Children execute in parallel, data partitioned by `dataSplitRatio`, with pre/post sub-communicator synchronization
  - **OMNIPIPE**: Pipeline overlap orchestration, computes step count and data slices based on `OmniPipeXYdata`, supports 2D bandwidth modeling
- **Algorithm registration**: `AlgSelector` singleton + `REGISTER_ALG` macro (`experimental/ops/all_gather/all_gather.cc` has instantiated the 4-level AllGather registration, guarded by the runtime switch `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR`, which is off by default)
- **Template implementations**:
  - `AllGatherMeshTemplate` (`template/aicpu/allgather_mesh_template.cc`) — Mesh AllGather, supports DirectToOutput mode
  - `AllGatherNhrTemplate` (`template/aicpu/allgather_nhr_template.cc`) — NHR AllGather, parallel PostCopy DMA optimization + last-step direct-write optimization
- **CommPlanner implementations**:
  - `RunMeshAllGather` (`template/comm_planners/mesh_comm_planner.cc`)
  - `RunNhrAllGather` (`template/comm_planners/nhr_comm_planner.cc`) — Recursive halving algorithm
- **Topology matching**: `TopoMatchFourLevel` (`topo/topo_match_four_level.cc`) — four-level symmetric topology
- **Executor adapter**: `AdaptorExecutorBase` (`executor/adaptor_executor.cc`) — bridges HCCL framework, `REGISTER_ALG` macro integration
- **OmniPipe utilities**: `OmniPipeXYdata` data structure + 2D bandwidth modeling + data slice computation (`executor/omnipipe_utils.h` / `.cc`), integrated into `OrchestrateOmniPipeLoop` main flow

### Not Implemented / Planned

- **Multi-engine**: Only AICPU; AIV (AI Core Vector) / CCU not implemented
- **Multi-operator**: Only AllGather; AllReduce / Broadcast / ReduceScatter / AlltoAll etc. not implemented
- **Asymmetric topology**: Only symmetric four-level topology supported
- **Cross-super-pod CommPlanner**: layer3 cross-super-pod communication primitive not implemented

---

## Limitations

1. **Experimental, not in production builds**: This module resides under `experimental/`, does not participate in production builds by default, and does not guarantee compatibility. APIs may change at any time.
2. **Single engine**: Only AICPU engine Templates are implemented; AIV / CCU engines are not supported.
3. **Single operator**: Only AllGather is registered; AllReduce / Broadcast / ReduceScatter / AlltoAll / Send / Recv etc. are not supported.
4. **Limited execution strategy coverage**: SEQUENCE, PARALLEL, and OMNIPIPE strategies are all implemented, but OMNIPIPE requires all subtree nodes to be OMNIPIPE with exactly 2 children; the currently registered algorithm (AllGather) uses SEQUENCE, and OMNIPIPE has not yet been used in any registered operator.
5. **Symmetric topology requirement**: `TopoMatchFourLevel` requires symmetric four-level topology (consistent view across all ranks); asymmetric topology is not supported.
6. **Limited test coverage**: UT covers 5 core modules (omnipipe_utils/data_ops/comm_planner/algo_desc/data_transfer); ST has no coverage yet.

---

## References

- experimental/ conventions: [experimental/README_en.md](../../../README_en.md)
- HCCL architecture brief (Chinese): [docs/zh/architecture/architecture-brief.md](../../../../docs/zh/architecture/architecture-brief.md)
