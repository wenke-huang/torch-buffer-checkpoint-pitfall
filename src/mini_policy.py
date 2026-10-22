"""A deliberately tiny policy-shaped module with a realistic buffer lifecycle.

The point of this module is not the attention math. It is the *buffer lifecycle*
that a real vision-language-action policy has, reduced to something small enough
to read in one sitting. The module contains three kinds of state:

1. Parameters
   ``qkv.weight/bias``, ``o_proj.weight/bias``, ``vision.proj.weight/bias``.
   Present in ``state_dict``, restored by the checkpoint loader.

2. Persistent buffers
   ``obs_mean``, ``obs_std``, ``vision.position_freqs``.
   Registered without ``persistent=False``, so they *are* in ``state_dict`` and
   the checkpoint loader restores them.

3. Non-persistent buffers
   ``rotary.inv_freq`` and ``vision.position_ids``, registered with
   ``register_buffer(..., persistent=False)``. They are *not* in ``state_dict``
   and therefore can never be restored by a loader that only walks checkpoint
   keys. This is the trap the repository demonstrates.

The two non-persistent buffers mirror the two real-world cases that matter:

* ``GemmaStyleRotaryEmbedding.inv_freq`` - inverse frequencies of a Gemma-style
  rotary position embedding (float buffer, derived from head_dim and base).
* ``VisionPositionEmbedding.position_ids`` - patch position indices of a
  SigLIP-style vision tower (integer index buffer, derived from the patch grid).

Everything here is written from scratch. No production loader or model code was
copied; only the failure mechanism was reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MiniPolicyConfig:
    """Configuration needed to rebuild the non-persistent buffers.

    In a real loader this comes from the checkpoint's ``config.json``. The two
    non-persistent buffers are pure functions of these fields, which is why they
    can be rebuilt after materialization.
    """

    hidden_size: int = 256
    head_dim: int = 256
    rope_base: float = 10_000.0
    num_positions: int = 8

    def __post_init__(self) -> None:
        if self.hidden_size != self.head_dim:
            raise ValueError("mini policy uses a single attention head; hidden_size must equal head_dim")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embedding to ``x`` of shape ``[T, D]``.

    ``cos`` / ``sin`` have shape ``[T, D]`` (frequencies repeated twice, as in
    the usual RoPE formulation).
    """

    def rotate_half(t: Tensor) -> Tensor:
        first, second = t.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    return x * cos + rotate_half(x) * sin


class GemmaStyleRotaryEmbedding(nn.Module):
    """Rotary embedding with a non-persistent ``inv_freq`` buffer.

    ``inv_freq`` is derived from ``head_dim`` and ``base``, so it never needs to
    be checkpointed. That is exactly why it is easy to forget it after a meta
    construction: nothing in the checkpoint will ever write it.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("inv_freq", self.build_inv_freq(), persistent=False)

    def build_inv_freq(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Rebuild ``inv_freq`` from ``head_dim`` and ``base`` only."""

        half = self.head_dim // 2
        exponent = torch.arange(half, dtype=torch.float32, device=device) / float(half)
        inv_freq = torch.pow(self.base, -exponent)
        if dtype is not None:
            inv_freq = inv_freq.to(dtype=dtype)
        return inv_freq

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        # positions: [T] float32
        angles = torch.outer(positions.to(torch.float32), self.inv_freq.to(torch.float32))
        emb = torch.cat((angles, angles), dim=-1)  # [T, head_dim]
        return emb.cos(), emb.sin()


class VisionPositionEmbedding(nn.Module):
    """SigLIP-style vision positional encoding with a non-persistent index buffer.

    ``position_ids`` holds the patch indices ``0 .. num_positions - 1``. It is
    not checkpointed (it is a pure function of the patch grid), so a meta /
    ``to_empty`` construction leaves it uninitialized unless it is rebuilt.

    The positional signal is computed arithmetically from the indices instead of
    looking up a learned table. That keeps this example faithful to the silent
    failure mode: garbage indices produce finite but meaningless encodings, with
    no exception raised.
    """

    def __init__(self, num_positions: int, hidden_size: int) -> None:
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even for sinusoidal vision positions")
        self.num_positions = num_positions
        self.hidden_size = hidden_size

        half = hidden_size // 2
        exponent = torch.arange(half, dtype=torch.float32) / float(half)
        self.register_buffer("position_freqs", torch.pow(10_000.0, -exponent), persistent=True)
        self.register_buffer(
            "position_ids",
            torch.arange(num_positions, dtype=torch.long).unsqueeze(0),  # [1, num_positions]
            persistent=False,
        )
        self.proj = nn.Linear(hidden_size, hidden_size)

    def build_position_ids(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Rebuild ``position_ids`` from ``num_positions`` only."""

        ids = torch.arange(self.num_positions, dtype=torch.long, device=device).unsqueeze(0)
        if dtype is not None:
            ids = ids.to(dtype=dtype)
        return ids

    def forward(self, x: Tensor) -> Tensor:
        # x: [T, hidden_size]
        if x.shape[0] > self.num_positions:
            raise ValueError(f"sequence length {x.shape[0]} exceeds num_positions {self.num_positions}")
        ids = self.position_ids[0, : x.shape[0]].to(torch.float32)  # [T]
        angles = ids.unsqueeze(-1) * self.position_freqs.unsqueeze(0)  # [T, hidden/2]
        encoding = torch.cat((angles.sin(), angles.cos()), dim=-1)  # [T, hidden]
        return x + self.proj(encoding)


class MiniPolicy(nn.Module):
    """Tiny stack: vision position encoding + rotary self-attention.

    Parameters and persistent buffers are restored by the checkpoint. The two
    non-persistent buffers are rebuilt either by the constructor or, after a
    meta / ``to_empty`` materialization, by ``restore_nonpersistent_buffers``.
    """

    def __init__(self, config: MiniPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniPolicyConfig()
        hidden = self.config.hidden_size

        # Persistent buffers: derived from config *and* stored in the checkpoint.
        self.register_buffer("obs_mean", torch.linspace(-0.4, 0.4, hidden), persistent=True)
        self.register_buffer("obs_std", torch.linspace(0.6, 1.4, hidden), persistent=True)

        self.vision = VisionPositionEmbedding(self.config.num_positions, hidden)
        self.rotary = GemmaStyleRotaryEmbedding(self.config.head_dim, self.config.rope_base)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.o_proj = nn.Linear(hidden, hidden)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``[T, hidden_size]`` observation features. Returns ``[T, hidden_size]``."""

        if x.dim() != 2 or x.shape[1] != self.config.hidden_size:
            raise ValueError(f"expected [T, {self.config.hidden_size}], got {tuple(x.shape)}")

        x = (x - self.obs_mean) / self.obs_std
        x = self.vision(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        positions = torch.arange(x.shape[0], dtype=torch.float32, device=x.device)
        cos, sin = self.rotary(positions)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        scores = (q @ k.transpose(-1, -2)) / (self.config.head_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        return self.o_proj(weights @ v)
