"""Minimal reproduction: non-persistent buffers after meta + to_empty.

The public API is re-exported here so that notebooks and scripts can simply do
``import src`` (or ``from src import ...``) after adding the repository root to
``sys.path``.
"""

from .loader import (
    FLOAT_SENTINEL,
    INT_SENTINEL,
    LoadReport,
    build_checkpoint,
    construct_on_meta,
    load_policy_buggy,
    load_policy_fixed,
    materialize_empty,
    poison_uninitialized,
    stream_load_state_dict,
)
from .mini_policy import (
    GemmaStyleRotaryEmbedding,
    MiniPolicy,
    MiniPolicyConfig,
    VisionPositionEmbedding,
    apply_rotary,
)
from .restore import (
    NonPersistentBufferError,
    nonpersistent_buffer_names,
    rebuild_nonpersistent_buffer,
    restore_nonpersistent_buffers,
)

__all__ = [
    "FLOAT_SENTINEL",
    "INT_SENTINEL",
    "GemmaStyleRotaryEmbedding",
    "LoadReport",
    "MiniPolicy",
    "MiniPolicyConfig",
    "NonPersistentBufferError",
    "VisionPositionEmbedding",
    "apply_rotary",
    "build_checkpoint",
    "construct_on_meta",
    "load_policy_buggy",
    "load_policy_fixed",
    "materialize_empty",
    "nonpersistent_buffer_names",
    "poison_uninitialized",
    "rebuild_nonpersistent_buffer",
    "restore_nonpersistent_buffers",
    "stream_load_state_dict",
]
