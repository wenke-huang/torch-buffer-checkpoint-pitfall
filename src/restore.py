"""Rebuild non-persistent buffers after ``meta`` construction + ``to_empty``.

Why this is needed
------------------
``register_buffer(name, tensor, persistent=False)`` excludes the tensor from
``state_dict``. A checkpoint loader that iterates over the checkpoint's keys
(a streaming loader, for instance) therefore has no idea the buffer exists.

That is harmless as long as the module is constructed normally: the buffer gets
a correct value from ``__init__``. It becomes a bug when the module is first
built on ``torch.device("meta")`` to save memory and later materialized with
``to_empty(device)``. ``to_empty`` allocates the storage but does not initialize
it, so the non-persistent buffers hold whatever bytes the allocator returned -
and nothing in the loading path will ever overwrite them.

The fix
-------
Recompute every non-persistent buffer from the module's configuration after
materialization. This is deterministic, cheap, and idempotent.

Fail-closed
-----------
If a module has a non-persistent buffer for which no rebuild rule exists, raise
``NonPersistentBufferError`` instead of continuing. A silently skipped buffer is
exactly the failure we are trying to eliminate; when a library upgrade adds a new
non-persistent buffer, the loader must stop and ask for a rule, not run with
uninitialized memory.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from .mini_policy import GemmaStyleRotaryEmbedding, VisionPositionEmbedding


class NonPersistentBufferError(RuntimeError):
    """A non-persistent buffer has no known rebuild rule; refuse to continue."""


def _iter_nonpersistent_buffers(module: nn.Module) -> Iterator[tuple[str, nn.Module, str]]:
    """Yield ``(qualified_name, owning_module, local_name)`` for non-persistent buffers.

    Detection uses only public APIs: a buffer is non-persistent exactly when it
    appears in ``named_buffers()`` but not in ``state_dict()``. This stays valid
    across PyTorch versions and needs no private attribute.
    """

    persistent_keys = set(module.state_dict().keys())
    for name, tensor in module.named_buffers():
        if tensor is None or name in persistent_keys:
            continue
        prefix, _, local_name = name.rpartition(".")
        owner = module.get_submodule(prefix) if prefix else module
        yield name, owner, local_name


def nonpersistent_buffer_names(module: nn.Module) -> list[str]:
    """Qualified names of all non-persistent buffers in ``module`` (sorted)."""

    return sorted(name for name, _, _ in _iter_nonpersistent_buffers(module))


def rebuild_nonpersistent_buffer(submodule: nn.Module, local_name: str, like: Tensor) -> Tensor:
    """Return a correct value for one non-persistent buffer.

    ``like`` is the current (possibly uninitialized) tensor; it provides the
    target device and dtype. Add new rules here when a policy gains a new kind of
    non-persistent buffer.
    """

    if isinstance(submodule, GemmaStyleRotaryEmbedding) and local_name == "inv_freq":
        return submodule.build_inv_freq(device=like.device, dtype=like.dtype)

    if isinstance(submodule, VisionPositionEmbedding) and local_name == "position_ids":
        return submodule.build_position_ids(device=like.device, dtype=like.dtype)

    raise NonPersistentBufferError(
        f"no rebuild rule for non-persistent buffer {local_name!r} "
        f"of module {type(submodule).__name__!r}; its storage is uninitialized and "
        f"the checkpoint cannot restore it. Add a rule to "
        f"{rebuild_nonpersistent_buffer.__module__}.rebuild_nonpersistent_buffer()."
    )


def restore_nonpersistent_buffers(module: nn.Module) -> list[str]:
    """Rebuild every non-persistent buffer of ``module`` in place.

    Returns the qualified names of the buffers that were restored (sorted).
    Raises ``NonPersistentBufferError`` if any non-persistent buffer has no
    rebuild rule.
    """

    restored: list[str] = []
    for name, submodule, local_name in _iter_nonpersistent_buffers(module):
        current = submodule._buffers[local_name]
        rebuilt = rebuild_nonpersistent_buffer(submodule, local_name, current)
        if rebuilt.shape != current.shape:
            raise NonPersistentBufferError(
                f"rebuild rule for {name!r} produced shape {tuple(rebuilt.shape)}, "
                f"expected {tuple(current.shape)}"
            )
        if rebuilt.device != current.device:
            rebuilt = rebuilt.to(device=current.device)
        with torch.no_grad():
            current.copy_(rebuilt)
        restored.append(name)
    return sorted(restored)
