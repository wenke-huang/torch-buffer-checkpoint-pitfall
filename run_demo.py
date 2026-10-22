"""Reproduce the before/after evidence for the non-persistent buffer pitfall.

Usage
-----
    python run_demo.py                 # prints to stdout
    python run_demo.py > results/before_after.txt

Output is deterministic: the buggy path is poisoned with a fixed sentinel after
``to_empty`` instead of relying on whatever the allocator left behind. The one
informational section that reads genuinely uninitialized memory is labelled as
such and is not asserted on.
"""

from __future__ import annotations

import platform

import torch
from torch import Tensor

from src.loader import (
    FLOAT_SENTINEL,
    INT_SENTINEL,
    build_checkpoint,
    construct_on_meta,
    load_policy_buggy,
    load_policy_fixed,
    materialize_empty,
    stream_load_state_dict,
)
from src.mini_policy import MiniPolicy, MiniPolicyConfig
from src.restore import NonPersistentBufferError, nonpersistent_buffer_names, restore_nonpersistent_buffers


def fmt_floats(tensor: Tensor, count: int) -> str:
    values = tensor.flatten()[:count].tolist()
    rendered = [f"{v:.4e}" if v != 0.0 and abs(v) < 1e-3 else f"{v:.4f}" for v in values]
    return "[" + ", ".join(rendered) + "]"


def fmt_ints(tensor: Tensor, count: int) -> str:
    return "[" + ", ".join(str(int(v)) for v in tensor.flatten()[:count].tolist()) + "]"


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> None:
    config = MiniPolicyConfig()
    checkpoint = build_checkpoint(config, seed=0)
    reference = MiniPolicy(config)
    reference.load_state_dict(checkpoint)
    reference.eval()

    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(config.num_positions, config.hidden_size, generator=generator)

    section("environment")
    print(f"python   {platform.python_version()}")
    print(f"torch    {torch.__version__}")
    print(f"device   cpu")
    print(f"policy   MiniPolicy(hidden={config.hidden_size}, head_dim={config.head_dim}, "
          f"num_positions={config.num_positions})")

    section("what the checkpoint contains")
    print(f"checkpoint keys ({len(checkpoint)}): {sorted(checkpoint)}")
    print(f"non-persistent buffers ({len(nonpersistent_buffer_names(reference))}, absent from state_dict): "
          f"{nonpersistent_buffer_names(reference)}")

    section("why to_empty is used: moving a meta module with .to() fails")
    meta_model = construct_on_meta(config)
    try:
        meta_model.to("cpu")
    except NotImplementedError as exc:
        print(f"model.to(cpu) raised NotImplementedError: {exc}")

    section("BEFORE: meta -> to_empty -> deterministic sentinel -> streaming load")
    buggy_model, report = load_policy_buggy(config, checkpoint)
    print(f"streaming loader copied {report.copied} keys")
    print(f"missing keys reported by the loader: {report.missing}")
    print(f"of which non-persistent buffers:     {report.missing_nonpersistent}")
    print(f"sentinel written after to_empty: float={FLOAT_SENTINEL!r}, int={INT_SENTINEL}")
    print()
    print(f"reference inv_freq[:6] = {fmt_floats(reference.rotary.inv_freq, 6)}")
    print(f"loaded    inv_freq[:6] = {fmt_floats(buggy_model.rotary.inv_freq, 6)}")
    print(f"torch.equal(inv_freq, reference) = {torch.equal(buggy_model.rotary.inv_freq, reference.rotary.inv_freq)}")
    print()
    print(f"reference position_ids = {fmt_ints(reference.vision.position_ids, 8)}")
    print(f"loaded    position_ids = {fmt_ints(buggy_model.vision.position_ids, 8)}")
    print(f"torch.equal(position_ids, reference) = "
          f"{torch.equal(buggy_model.vision.position_ids, reference.vision.position_ids)}")
    print()
    with torch.no_grad():
        buggy_out = buggy_model(x)
        reference_out = reference(x)
    print(f"forward output: shape={tuple(buggy_out.shape)}, all finite={bool(torch.isfinite(buggy_out).all())}, "
          f"exception=none")
    print(f"max |buggy - reference| = {(buggy_out - reference_out).abs().max().item():.6f}")
    print(f"torch.allclose(buggy, reference) = {torch.allclose(buggy_out, reference_out)}")

    section("AFTER: same path + restore_nonpersistent_buffers(model)")
    fixed_model, fixed_report, restored = load_policy_fixed(config, checkpoint)
    print(f"streaming loader copied {fixed_report.copied} keys")
    print(f"restored non-persistent buffers ({len(restored)}): {restored}")
    print(f"reference inv_freq[:6] = {fmt_floats(reference.rotary.inv_freq, 6)}")
    print(f"restored  inv_freq[:6] = {fmt_floats(fixed_model.rotary.inv_freq, 6)}")
    print(f"torch.equal(inv_freq, reference) = {torch.equal(fixed_model.rotary.inv_freq, reference.rotary.inv_freq)}")
    print()
    print(f"reference position_ids = {fmt_ints(reference.vision.position_ids, 8)}")
    print(f"restored  position_ids = {fmt_ints(fixed_model.vision.position_ids, 8)}")
    print(f"torch.equal(position_ids, reference) = "
          f"{torch.equal(fixed_model.vision.position_ids, reference.vision.position_ids)}")
    print()
    with torch.no_grad():
        fixed_out = fixed_model(x)
    print(f"max |restored - reference| = {(fixed_out - reference_out).abs().max().item():.6f}")
    print(f"torch.equal(restored, reference) = {torch.equal(fixed_out, reference_out)}")

    section("FAIL-CLOSED: unknown non-persistent buffer refuses to load")
    class ModuleWithUnknownBuffer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mystery_cache", torch.zeros(4), persistent=False)

    try:
        restore_nonpersistent_buffers(ModuleWithUnknownBuffer())
    except NonPersistentBufferError as exc:
        print(f"raised NonPersistentBufferError: {exc}")

    section("INFORMATIONAL: genuinely uninitialized memory (allocator-dependent, not asserted)")
    fresh = materialize_empty(construct_on_meta(config), "cpu")
    fresh_report = stream_load_state_dict(fresh, checkpoint)
    print(f"streaming loader copied {fresh_report.copied} keys, still missing: {fresh_report.missing}")
    print(f"inv_freq[:6] = {fmt_floats(fresh.rotary.inv_freq, 6)}")
    print(f"position_ids = {fmt_ints(fresh.vision.position_ids, 8)}")
    print(f"torch.equal(inv_freq, reference) = {torch.equal(fresh.rotary.inv_freq, reference.rotary.inv_freq)}")
    print(f"torch.equal(position_ids, reference) = {torch.equal(fresh.vision.position_ids, reference.vision.position_ids)}")
    print("note: these bytes are whatever the allocator returned; they are not reproducible,")
    print("      which is why the tests do not depend on them.")


if __name__ == "__main__":
    main()
