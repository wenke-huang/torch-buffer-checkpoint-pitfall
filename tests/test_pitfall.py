"""Regression tests for the non-persistent buffer / meta + to_empty pitfall.

All tests are deterministic. The buggy path does not assert "the allocator
happened to leave garbage"; ``poison_uninitialized`` writes a fixed sentinel
after ``to_empty``, so the assertions hold on every machine and every run.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.loader import (
    FLOAT_SENTINEL,
    INT_SENTINEL,
    build_checkpoint,
    load_policy_buggy,
    stream_load_state_dict,
)
from src.mini_policy import MiniPolicy, MiniPolicyConfig
from src.restore import (
    NonPersistentBufferError,
    nonpersistent_buffer_names,
    restore_nonpersistent_buffers,
)

EXPECTED_NONPERSISTENT = ["rotary.inv_freq", "vision.position_ids"]


@pytest.fixture(scope="module")
def config() -> MiniPolicyConfig:
    return MiniPolicyConfig()


@pytest.fixture(scope="module")
def checkpoint(config: MiniPolicyConfig) -> dict[str, torch.Tensor]:
    return build_checkpoint(config, seed=0)


@pytest.fixture(scope="module")
def reference(config: MiniPolicyConfig, checkpoint: dict[str, torch.Tensor]) -> MiniPolicy:
    """A normally constructed model with the same checkpoint loaded."""

    model = MiniPolicy(config)
    model.load_state_dict(checkpoint)  # state_dict has no non-persistent buffers, so strict=True is fine
    model.eval()
    return model


def test_nonpersistent_buffers_are_absent_from_checkpoint_and_state_dict(reference: MiniPolicy, checkpoint: dict[str, torch.Tensor]) -> None:
    """The root cause: the checkpoint cannot restore what it never contains."""

    assert nonpersistent_buffer_names(reference) == EXPECTED_NONPERSISTENT
    assert set(reference.state_dict()) == set(checkpoint)  # checkpoint covers the entire state_dict ...
    for name in EXPECTED_NONPERSISTENT:
        assert name not in reference.state_dict()  # ... and the non-persistent buffers are simply not in it
        assert name not in checkpoint


def test_buggy_path_leaves_nonpersistent_buffers_uninitialized(config: MiniPolicyConfig, checkpoint: dict[str, torch.Tensor]) -> None:
    """After meta -> to_empty -> streaming load, the non-persistent buffers are still garbage."""

    model, report = load_policy_buggy(config, checkpoint)

    # The loader itself reports the keys it could not fill.
    assert report.missing_nonpersistent == EXPECTED_NONPERSISTENT
    assert report.copied == len(checkpoint)

    # Deterministic stand-in for uninitialized memory.
    assert torch.equal(model.rotary.inv_freq, torch.full_like(model.rotary.inv_freq, FLOAT_SENTINEL))
    assert torch.equal(model.vision.position_ids, torch.full_like(model.vision.position_ids, INT_SENTINEL))


def test_buggy_path_restores_parameters_and_persistent_buffers_only(config: MiniPolicyConfig, checkpoint: dict[str, torch.Tensor], reference: MiniPolicy) -> None:
    """Parameters and persistent buffers are fine; only the non-persistent buffers are wrong."""

    model, _ = load_policy_buggy(config, checkpoint)

    assert torch.equal(model.qkv.weight, reference.qkv.weight)
    assert torch.equal(model.qkv.bias, reference.qkv.bias)
    assert torch.equal(model.o_proj.weight, reference.o_proj.weight)
    assert torch.equal(model.obs_mean, reference.obs_mean)
    assert torch.equal(model.obs_std, reference.obs_std)
    assert torch.equal(model.vision.position_freqs, reference.vision.position_freqs)

    assert not torch.equal(model.rotary.inv_freq, reference.rotary.inv_freq)
    assert not torch.equal(model.vision.position_ids, reference.vision.position_ids)


def test_restore_reproduces_reference_buffers_exactly(config: MiniPolicyConfig, checkpoint: dict[str, torch.Tensor], reference: MiniPolicy) -> None:
    """The fix rebuilds the buffers bit-for-bit, and is idempotent."""

    model, _ = load_policy_buggy(config, checkpoint)

    restored = restore_nonpersistent_buffers(model)
    assert restored == EXPECTED_NONPERSISTENT
    assert torch.equal(model.rotary.inv_freq, reference.rotary.inv_freq)
    assert torch.equal(model.vision.position_ids, reference.vision.position_ids)

    # Calling it twice must not change anything.
    restore_nonpersistent_buffers(model)
    assert torch.equal(model.rotary.inv_freq, reference.rotary.inv_freq)
    assert torch.equal(model.vision.position_ids, reference.vision.position_ids)


def test_forward_matches_reference_only_after_restore(config: MiniPolicyConfig, checkpoint: dict[str, torch.Tensor], reference: MiniPolicy) -> None:
    """The corruption is silent: finite output, wrong values, no exception."""

    model, _ = load_policy_buggy(config, checkpoint)
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(config.num_positions, config.hidden_size, generator=generator)

    with torch.no_grad():
        buggy_out = model(x)
        reference_out = reference(x)

    assert torch.isfinite(buggy_out).all()  # no NaN, no inf, no IndexError, nothing to notice
    assert not torch.allclose(buggy_out, reference_out)  # but numerically wrong

    restore_nonpersistent_buffers(model)
    with torch.no_grad():
        fixed_out = model(x)
    assert torch.equal(fixed_out, reference_out)


def test_restore_fails_closed_on_unknown_nonpersistent_buffer() -> None:
    """An unknown non-persistent buffer must stop the load, not be skipped."""

    class ModuleWithUnknownBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mystery_cache", torch.zeros(4), persistent=False)

    model = ModuleWithUnknownBuffer()
    with pytest.raises(NonPersistentBufferError) as excinfo:
        restore_nonpersistent_buffers(model)

    message = str(excinfo.value)
    assert "mystery_cache" in message
    assert "ModuleWithUnknownBuffer" in message


def test_stream_loader_rejects_unknown_checkpoint_keys(config: MiniPolicyConfig, checkpoint: dict[str, torch.Tensor]) -> None:
    """Sanity check on the loader used by every test above."""

    model = MiniPolicy(config)
    with pytest.raises(ValueError, match="unknown to the module"):
        stream_load_state_dict(model, {**checkpoint, "not.a.real.key": torch.zeros(1)})
