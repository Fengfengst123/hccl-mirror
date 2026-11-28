# Performing AllReduce with MindSpore

## Sample Description

This sample shows how to perform AllReduce using MindSpore APIs, covering:

- Specifying the device and running mode with `mindspore.set_context()`.
- Initializing the communication group with `mindspore.communication.init()` (collective communication on Ascend hardware is powered by HCCL).
- Performing AllReduce with the `ops.AllReduce()` operator.

## Environment Preparation

### Environment Requirements

This sample supports the following products in a single-server N-card configuration (N >= 2):

- Atlas A3 series products

The sample depends on the following Python package:

- `mindspore`: Huawei's open-source all-scenario AI framework. Its distributed communication capability on Ascend hardware is powered by HCCL.

### Setting Environment Variables

```bash
# Set CANN environment variables (root user default installation path as an example)
source /usr/local/Ascend/cann/set_env.sh
```

## Running the Sample

Each process corresponds to one card. Use the MindSpore launcher `msrun`, taking 8 cards as an example:

```bash
msrun --worker_num=8 hccl_mindspore_allreduce_test.py
```

The device ID is injected by the launcher. For single-process debugging, also launch the script via the launcher (running the script directly fails in `init()` because mandatory environment variables such as `RANK_ID` are not set):

```bash
msrun --worker_num=1 --local_worker_num=1 hccl_mindspore_allreduce_test.py
```

The script performs built-in expected-value verification; the sample passes when all processes print `AllReduce test passed`.

> Note: The `HCCL_OP_EXPANSION_MODE` environment variable configures the expansion mode of communication operators. For the supported ranges on different product models, see the usage of this variable in the [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Set the expansion mode of communication operators to the AI CPU communication engine
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Sample Output

Each rank initializes its data to 0~7. After AllReduce, each rank holds the element-wise sum across all 8 ranks.

```
[Rank 0] Input: [0. 1. 2. 3. 4. 5. 6. 7.]
[Rank 1] Input: [0. 1. 2. 3. 4. 5. 6. 7.]
...
[Rank 0] Output: [ 0.  8. 16. 24. 32. 40. 48. 56.]
[Rank 1] Output: [ 0.  8. 16. 24. 32. 40. 48. 56.]
...
```
