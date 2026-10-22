"""The loading path that loses non-persistent buffers (the bug in this repo).

It mirrors what a memory-conscious deployment loader does for a large policy:

1. Build the module on ``torch.device("meta")``. Meta tensors have shape, dtype
   and device (``meta``) but no storage, so construction costs almost nothing.
2. Materialize the module on the target device with ``to_empty(device=...)``.
   This allocates storage for every parameter and buffer but does *not*
   initialize it. ``model.to(device)`` would raise here instead; ``to_empty`` is
   the API that lets a loader get past the meta device.
3. Stream the checkpoint: walk the checkpoint's keys and copy each tensor into
   the matching parameter/buffer. This is where the bug is. Non-persistent
   buffers are absent from ``state_dict``, so they are absent from the
   checkpoint, so this loop never touches them. They keep the uninitialized
   bytes from step 2.

Nothing in this path raises. The model runs, produces finite outputs, and is
numerically wrong. That is why the bug survived into a real robot deployment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .mini_policy import MiniPolicy, MiniPolicyConfig
from .restore import nonpersistent_buffer_names, restore_nonpersistent_buffers

# Deterministic stand-ins for uninitialized memory. The tests must not depend on
# what the allocator happened to leave in a fresh page, so after ``to_empty`` we
# explicitly fill every tensor with these values. They are chosen to match the
# magnitudes observed on the real deployment:
#   * float32 buffers held the smallest positive subnormal, 2**-149 = 1.4013e-45;
#   * int64 index buffers held values on the order of 1e18.
# The bug does not depend on these exact numbers; they only make it reproducible.
FLOAT_SENTINEL = 2.0**-149  # 1.4013e-45
INT_SENTINEL = 2_314_861_380_256_428_066


@dataclass(frozen=True)
class LoadReport:
    """What a streaming load actually did."""

    copied: int
    missing: list[str] = field(default_factory=list)
    missing_nonpersistent: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)


def build_checkpoint(config: MiniPolicyConfig, *, seed: int = 0) -> dict[str, Tensor]:
    """Create an in-memory checkpoint (state_dict) from a normal construction.

    ``state_dict`` contains parameters and persistent buffers only. The
    non-persistent buffers are deliberately absent, exactly as in a real
    checkpoint.
    """

    torch.manual_seed(seed)
    model = MiniPolicy(config)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def construct_on_meta(config: MiniPolicyConfig) -> MiniPolicy:
    """Build the policy on the meta device, as a low-memory loader would."""

    with torch.device("meta"):
        model = MiniPolicy(config)
    return model


def materialize_empty(model: nn.Module, device: torch.device | str = "cpu") -> nn.Module:
    """Allocate real storage for a meta module without initializing it."""

    return model.to_empty(device=torch.device(device))


def poison_uninitialized(model: nn.Module, *, float_value: float = FLOAT_SENTINEL, int_value: int = INT_SENTINEL) -> None:
    """Write deterministic garbage into every parameter and buffer.

    Stands in for uninitialized allocation. Parameters are poisoned as well, so
    that a checkpoint key that is actually missing shows up as a deterministic
    wrong value rather than a random one.
    """

    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            tensor.fill_(float_value if tensor.is_floating_point() else int_value)


def stream_load_state_dict(model: nn.Module, checkpoint: Mapping[str, Tensor]) -> LoadReport:
    """Copy checkpoint tensors into the module key by key.

    Only keys present in the checkpoint are copied; there is no second pass that
    recomputes anything. Missing keys are reported but do not raise, mirroring
    real loaders whose "missing keys" warning is easy to ignore.
    """

    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    targets: dict[str, Tensor] = {**named_parameters, **named_buffers}

    unexpected = sorted(key for key in checkpoint if key not in targets)
    if unexpected:
        raise ValueError(f"checkpoint contains keys unknown to the module: {unexpected}")

    missing = sorted(key for key in targets if key not in checkpoint)
    missing_nonpersistent = sorted(set(missing) & set(nonpersistent_buffer_names(model)))

    with torch.no_grad():
        for key, value in checkpoint.items():
            target = targets[key]
            if target.shape != value.shape:
                raise ValueError(f"shape mismatch for {key!r}: checkpoint {tuple(value.shape)} vs module {tuple(target.shape)}")
            target.copy_(value)

    return LoadReport(
        copied=len(checkpoint),
        missing=missing,
        missing_nonpersistent=missing_nonpersistent,
        unexpected=unexpected,
    )


def load_policy_buggy(
    config: MiniPolicyConfig,
    checkpoint: Mapping[str, Tensor],
    *,
    device: torch.device | str = "cpu",
    poison: bool = True,
) -> tuple[MiniPolicy, LoadReport]:
    """The full buggy path: meta -> to_empty -> (garbage) -> stream checkpoint.

    Returns the loaded policy and a report. The report's ``missing_nonpersistent``
    field names the buffers that were never written: that list is the bug.
    """

    model = construct_on_meta(config)
    model = materialize_empty(model, device)  # type: ignore[assignment]
    if poison:
        poison_uninitialized(model)
    report = stream_load_state_dict(model, checkpoint)
    return model, report  # type: ignore[return-value]


def load_policy_fixed(
    config: MiniPolicyConfig,
    checkpoint: Mapping[str, Tensor],
    *,
    device: torch.device | str = "cpu",
) -> tuple[MiniPolicy, LoadReport, list[str]]:
    """Same path as ``load_policy_buggy`` plus the actual fix."""

    model, report = load_policy_buggy(config, checkpoint, device=device)
    restored = restore_nonpersistent_buffers(model)
    return model, report, restored
