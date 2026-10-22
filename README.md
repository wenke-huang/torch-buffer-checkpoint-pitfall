# The non-persistent buffer pitfall: `meta` + `to_empty` + streaming checkpoint load

A minimal, deterministic reproduction of a real bug found while deploying a π0.5
vision-language-action policy to a dual-arm humanoid robot.

Companion write-up: [kuavo-pi05-vla](https://github.com/wenke-huang/kuavo-pi05-vla) — the
fine-tuning and deployment notes this bug came out of.

The model loaded without an error, produced finite action chunks, and moved the
robot. The actions were incoherent. The cause was not the data, the controller,
or the network: two position-encoding buffers had never been assigned a value,
because they are registered with `persistent=False` and the loader only copies
keys that exist in the checkpoint.

This repository contains a small from-scratch policy module, the buggy loading
path, the fix, and regression tests that make the failure reproducible in CI
instead of depending on what the memory allocator happens to return.

## Contents

| Path | What it is |
|---|---|
| [`src/mini_policy.py`](src/mini_policy.py) | Tiny attention policy with parameters, persistent buffers and two non-persistent buffers: a Gemma-style RoPE `inv_freq` and a SigLIP-style vision `position_ids` |
| [`src/loader.py`](src/loader.py) | The buggy path: build on `torch.device("meta")`, materialize with `to_empty`, stream only checkpoint keys |
| [`src/restore.py`](src/restore.py) | `restore_nonpersistent_buffers(module)`: rebuild from config, fail closed on unknown buffers |
| [`tests/test_pitfall.py`](tests/test_pitfall.py) | 7 deterministic regression tests, no GPU, no skips |
| [`run_demo.py`](run_demo.py) | Produces the before/after evidence |
| [`results/before_after.txt`](results/before_after.txt) | Real output of `python run_demo.py` on the machine described below |
| [`requirements.txt`](requirements.txt) | `torch`, `pytest` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python run_demo.py > results/before_after.txt
```

Tested with Python 3.13.5, torch 2.14.1 and pytest 9.1.1 on CPU. No GPU and no
model download are required.

## Symptom

On the robot, the policy produced action chunks that were plausible in shape and
finite in value but not coherent over time: the arm moved, but the trajectory
looked jittery and repeated itself instead of tracking the commanded task. No
exception was raised, no NaN appeared, and the load log looked normal. A
`state_dict`-level check can only confirm that all keys the loader knows about
were copied; it cannot see tensors that are not in `state_dict` at all.

The failure is silent by construction. That is what made it expensive to find.

## What `results/before_after.txt` shows

The buggy path leaves `rotary.inv_freq` and `vision.position_ids` at the
deterministic sentinel written after `to_empty`:

```text
reference inv_freq[:6] = [1.0000, 0.9306, 0.8660, 0.8058, 0.7499, 0.6978]
loaded    inv_freq[:6] = [1.4013e-45, 1.4013e-45, 1.4013e-45, 1.4013e-45, 1.4013e-45, 1.4013e-45]
torch.equal(inv_freq, reference) = False

reference position_ids = [0, 1, 2, 3, 4, 5, 6, 7]
loaded    position_ids = [2314861380256428066, 2314861380256428066, ...]
torch.equal(position_ids, reference) = False

forward output: shape=(8, 256), all finite=True, exception=none
max |buggy - reference| = 0.519259
torch.allclose(buggy, reference) = False
```

After the fix, both buffers and the forward output match the reference exactly:

```text
restored non-persistent buffers (2): ['rotary.inv_freq', 'vision.position_ids']
torch.equal(inv_freq, reference) = True
torch.equal(position_ids, reference) = True
max |restored - reference| = 0.000000
torch.equal(restored, reference) = True
```

The sentinel magnitudes are not arbitrary. On the real deployment, the float
buffer contained the smallest positive float32 subnormal, `1.4013e-45`, and the
integer index buffer contained values around `1e18`. The tests reuse those
magnitudes so the reproduction looks like the field failure while remaining
deterministic.

## The mechanism, step by step

### 1. A non-persistent buffer is not part of `state_dict`

```python
self.register_buffer("inv_freq", inv_freq, persistent=False)
```

`persistent=False` means: this tensor is part of the module, moves with it
through `.to()`, `.cuda()`, `.float()`, but is not serialized in `state_dict`.
Model code does this deliberately for tensors that are pure functions of
configuration: rotary inverse frequencies, patch position indices, causal
masks, fixed sinusoidal tables. There is no reason to store them in a
checkpoint because they can be recomputed.

### 2. The model is built on the `meta` device to save memory

```python
with torch.device("meta"):
    model = MiniPolicy(config)
```

Meta tensors carry shape, dtype and device but no storage. Building a large
policy on `meta` costs almost nothing; this is a common deployment trick for
memory-constrained or streaming loaders.

### 3. `to_empty` allocates storage without initializing it

```python
model = model.to_empty(device="cpu")  # or "cuda"
```

`torch.nn.Module.to()` refuses to move a meta module because there is no data
to copy. The error message points at `to_empty`, which is the only way out:
it allocates real storage and leaves it uninitialized. Every parameter, every
persistent buffer and every non-persistent buffer is now backed by whatever
memory the allocator returned.

On a fresh allocation that is often all zeros. Zeros are a particularly nasty
case: they do not crash, they do not produce NaN, and they look like a
deliberate initialization while being semantically wrong. The `INFORMATIONAL`
section of the demo prints this case.

### 4. The streaming loader copies checkpoint keys only

```python
for key, value in checkpoint.items():
    model.get_tensor(key).copy_(value)
```

The checkpoint is a `state_dict`, and a `state_dict` does not contain
non-persistent buffers. So this loop, no matter how many shards it streams
through or how carefully it validates shapes, never writes the two buffers. They
keep the uninitialized bytes from step 3. The loader may even print a
`missing keys` warning listing exactly those buffers, which is easy to read as
benign bookkeeping.

`load_state_dict(..., assign=True)` does not help. It also iterates over
`state_dict` keys; a key that is not in `state_dict` is not assigned.

## The real-deployment evidence

Recorded while debugging the humanoid deployment; environment details are
intentionally omitted here. The values are the ones this repository reproduces
(the reference `inv_freq` is reproduced bit-for-bit by `head_dim=256`,
`base=10000`):

```text
reference inv_freq = [1.0000, 0.9306, 0.8660, 0.8058, 0.7499, 0.6978]
after meta -> to_empty = [1.4013e-45, 1.4013e-45, 0, 1.3873e-43, 0, 0]
equal? False

reference position_ids = [0, 1, 2, 3, 4, 5, 6, 7]
after meta -> to_empty = [2314861380256428066, 4485057601257611296, ...]
equal? False
```

`inv_freq` values around `1.4e-45` are numerical zero for practical purposes:
the rotary embedding becomes the identity rotation, so the model can no longer
tell token positions apart. Position indices around `1e18` are meaningless as
patch positions. The forward pass still runs and returns finite numbers.

## Fix option A: rebuild from the configuration (used here)

After materialization, recompute every non-persistent buffer from the same
configuration that the constructor used:

```python
model = model.to_empty(device=device)
restore_nonpersistent_buffers(model)   # returns ['rotary.inv_freq', 'vision.position_ids']
report = stream_load_state_dict(model, checkpoint)
```

Why this is safe: those buffers are pure functions of config. `inv_freq` depends
only on `head_dim` and `rope_base`; `position_ids` depends only on the patch grid.
The rebuild is deterministic, cheap and idempotent.

Why it must fail closed: `restore_nonpersistent_buffers` raises
`NonPersistentBufferError` when it meets a non-persistent buffer it has no rule
for. A library upgrade can add a new non-persistent buffer at any time; skipping
unknown buffers would reintroduce exactly the silent failure the fix is meant to
remove. The error names the module and the buffer and points at the function to
extend:

```text
NonPersistentBufferError: no rebuild rule for non-persistent buffer 'mystery_cache'
of module 'ModuleWithUnknownBuffer'; its storage is uninitialized and the
checkpoint cannot restore it. Add a rule to
src.restore.rebuild_nonpersistent_buffer().
```

## Fix option B: make the values checkpointed

Register the buffers as persistent, or save and restore them explicitly in the
checkpoint metadata:

```python
self.register_buffer("inv_freq", inv_freq, persistent=True)
```

Trade-offs:

| | A. Rebuild from config | B. Persist in checkpoint |
|---|---|---|
| Old checkpoints | Work as-is | Do not contain the new keys; the loader must fall back to rebuilding anyway |
| `state_dict` compatibility | Unchanged | Changes the serialized format and the expected keys |
| New config values | Picked up automatically | Stale checkpoint value can override a config change, silently |
| Risk of forgetting | One explicit call after `to_empty` | Must remember to save and load the extra keys |
| Disk / bandwidth | None | A few kilobytes to megabytes per checkpoint |

For both options, the minimum viable production guard is the same: after
loading, assert that position buffers equal their configuration-derived
reference values, so a future regression fails loudly instead of driving the
robot. A rebuild followed by a value check costs microseconds.

Not constructing on `meta` at all also avoids the bug, at the cost of the memory
the meta trick was designed to save.

## Regression tests

```bash
python -m pytest tests/ -q
# 7 passed
```

| Test | What it pins down |
|---|---|
| `test_nonpersistent_buffers_are_absent_from_checkpoint_and_state_dict` | The checkpoint covers the entire `state_dict`; the two buffers are absent from both |
| `test_buggy_path_leaves_nonpersistent_buffers_uninitialized` | The streaming load reports the two missing buffers and leaves the sentinel in place |
| `test_buggy_path_restores_parameters_and_persistent_buffers_only` | Parameters and persistent buffers are correct; only non-persistent buffers are wrong |
| `test_restore_reproduces_reference_buffers_exactly` | The rebuild is bit-for-bit and idempotent |
| `test_forward_matches_reference_only_after_restore` | Buggy output is finite but wrong; restored output is bit-identical to a normally constructed model |
| `test_restore_fails_closed_on_unknown_nonpersistent_buffer` | An unknown non-persistent buffer raises instead of being skipped |
| `test_stream_loader_rejects_unknown_checkpoint_keys` | The streaming loader used by the other tests behaves as expected |

Why the tests do not depend on uninitialized memory: after `to_empty` the demo
calls `poison_uninitialized`, which fills every parameter and buffer with a
fixed sentinel (`2**-149` for floats, `2314861380256428066` for integers). The
assertions are then exact equality against the sentinel and against the
reference model, on every machine and every run. The genuinely uninitialized
allocation is shown only in the informational section of `run_demo.py` and is
never asserted on.

## When you can be bitten

This pitfall needs four conditions; the first three are common and the fourth is
the default mistake.

1. The module is constructed on `meta` and materialized with `to_empty`. This
   includes low-memory loaders, sharded/streaming loaders, and
   `torch.load(..., map_location="meta")` followed by materialization.
2. At least one buffer is registered with `persistent=False`. Common in
   transformer position encodings, vision position indices, causal masks and
   fixed sinusoidal tables.
3. The loader copies only checkpoint keys. Streaming loaders always do this;
   loaders that were ported to a new checkpoint format often do it accidentally.
4. Nothing rebuilds or verifies the non-persistent buffers after
   materialization.

Extra risk when the checkpoint format changed: a loader ported from a framework
or a serialization format where those buffers *were* saved may assume the same
`state_dict` guarantees still hold, which they do not.

How to audit a model in one minute:

```python
persistent_keys = set(model.state_dict())
for name, _ in model.named_buffers():
    if name not in persistent_keys:
        print(name)  # non-persistent: absent from every checkpoint
```

Every line printed is state that must be rebuilt after `to_empty` and that no
checkpoint can restore. Cross-check that list against
`restore_nonpersistent_buffers` and add a value-level test comparing the loaded
buffers with a normally constructed model.

## Fidelity notes

The mini policy is written from scratch for this repository; no production
loader or model source was copied. It reproduces the *buffer lifecycle* of a
real policy, not the full model. In the real system the two buffers belonged to
a Gemma-style rotary embedding and a SigLIP-style vision tower. Here the
attention has a single head, and the vision positional signal is computed
arithmetically from the indices rather than looked up in a learned table, so the
corruption stays finite and silent instead of depending on out-of-range index
behavior. The load path, the serialization semantics and the failure mode are
the ones that mattered in production.

No hostnames, addresses, credentials or internal paths from the deployment are
included in this repository.

## References

- [`torch.nn.Module.register_buffer`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer), including the `persistent` argument
- [`torch.nn.Module.to_empty`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to_empty)
- [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict) and the serialization notes
- Hugging Face Transformers model code for examples of position buffers registered with `persistent=False` in rotary embeddings and vision towers

## 中文要点

- **现象**：具身智能 VLA 模型在真机部署时动作不连贯；加载不报错、推理不抛异常、输出有限，纯静默失效。
- **根因**：模型先在 `torch.device("meta")` 上构造省内存，再 `to_empty()` 实体化；`to_empty` 只分配、不初始化。`register_buffer(..., persistent=False)` 的缓冲区不在 `state_dict` 里，checkpoint 流式加载只按 checkpoint 的 key 拷贝，所以这类缓冲区**永远不会**被赋正确值，残留的是未初始化内存。
- **受影响的量**：Gemma RoPE 的 `inv_freq`、SigLIP 的 `position_ids`。位置编码损坏后模型仍能前向、输出有限值，但位置信息已经失效，因此不抛异常。
- **修法 A（本仓库采用）**：`to_empty` 之后按 config 重建非持久缓冲区，即 `restore_nonpersistent_buffers(model)`；对未知缓冲区类型 fail-closed 直接报错，避免库升级后再次静默残留。重建是确定性的、幂等的。
- **修法 B**：把这些缓冲区注册为 persistent，或显式写进 checkpoint。代价是会改变 `state_dict` 格式、旧 checkpoint 仍然缺这些 key，而且可能用旧值覆盖新配置。
- **`load_state_dict(assign=True)` 不能修**：它同样只遍历 `state_dict` 的 key。
- **测试为什么稳定**：真实未初始化内存不可复现，所以 `to_empty` 之后用固定哨兵值显式填充，断言的是「等于哨兵」与「修复后与参考模型逐位相等」，不依赖随机垃圾。
- **什么时候会踩到**：meta 构造 + `to_empty`、存在非持久缓冲区、加载器只按 checkpoint key 拷贝、加载后不重建也不校验，四者同时成立。这正是流式/低内存加载器的常见组合。
- **最小防护**：加载后把位置编码缓冲区与配置推导出的参考值做一次断言；成本微秒级，能把静默失效变成显式失败。
