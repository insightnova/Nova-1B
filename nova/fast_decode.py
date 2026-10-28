"""Inference lifecycle for GDN2 state + convolution state + attention KV cache.

Use eval() and no_grad()/inference_mode(), enable once, then reset for each
new batch of equal-length prompts. No global OLMo or FLA monkey patches.
"""

from __future__ import annotations

import torch

from olmo_core.nn.attention import Attention
from olmo_core.nn.attention.kv_cache import KVCacheManager

from .gdn2 import GatedDeltaNet2

__all__ = ["enable", "reset", "disable", "is_enabled"]


class _HostPositionKVCache(KVCacheManager):
    """Track a common prompt position without synchronizing a GPU scalar.

    This subclass is attached only to the model passed to enable(). The base
    KVCacheManager used by gdnh1 and other models is unchanged.
    """

    def __init__(self, **kwargs):
        # Keep mutable KV buffers usable when reset() is called outside the
        # inference_mode context in which enable() originally ran.
        with torch.inference_mode(False):
            super().__init__(**kwargs)
        self._host_pos = 0

    def update_seqlen(self, seqlen: int):
        super().update_seqlen(seqlen)
        self._host_pos += int(seqlen)

    def current_position(self):
        return self._host_pos

    def zero_cache(self):
        super().zero_cache()
        self._host_pos = 0

    def reallocate(self, *args, **kwargs):
        with torch.inference_mode(False):
            super().reallocate(*args, **kwargs)
        self._host_pos = 0


def _modules(model):
    attn, gdn = [], []
    for module in model.modules():
        if isinstance(module, GatedDeltaNet2):
            gdn.append(module)
        elif isinstance(module, Attention):
            attn.append(module)
    return attn, gdn


def enable(model, max_seq_len: int, batch_size: int = 1) -> None:
    """Enable empty caches. Existing prompts are discarded, including on reuse."""
    if model.training:
        raise ValueError("Call model.eval() before fast_decode.enable()")
    if batch_size <= 0 or max_seq_len <= 0:
        raise ValueError("batch_size and max_seq_len must be positive")
    attn, gdn = _modules(model)
    if not gdn:
        raise ValueError("No gdnh2 GatedDeltaNet2 layers found")
    # Validate all backends before allocating or mutating any caches.
    for module in attn:
        module.backend.assert_supports_kv_cache()
    try:
        for module in attn:
            cache = module.kv_cache_manager
            if isinstance(cache, _HostPositionKVCache):
                cache.reset(batch_size, max_seq_len)
            else:
                dtype = module.w_k.weight.dtype
                if dtype not in (torch.float16, torch.bfloat16):
                    dtype = torch.bfloat16
                module.kv_cache_manager = _HostPositionKVCache(
                    batch_size=batch_size, max_seq_len=max_seq_len,
                    num_kv_heads=module.n_kv_heads, head_dim=module.head_dim,
                    device=module.w_k.weight.device, dtype=dtype,
                )
        for module in gdn:
            module.enable_decode(batch_size, max_seq_len)
    except Exception:
        disable(model)
        raise


def reset(model, batch_size: int = 1, max_seq_len: int | None = None) -> None:
    """Start an independent prompt batch, resetting every type of state."""
    _, gdn = _modules(model)
    if not gdn or not all(module.decode_enabled for module in gdn):
        raise ValueError("Call fast_decode.enable() before reset()")
    length = max_seq_len if max_seq_len is not None else gdn[0]._decode_max_seq_len
    enable(model, max_seq_len=length, batch_size=batch_size)


def disable(model) -> None:
    """Release inference caches before training or uncached evaluation."""
    attn, gdn = _modules(model)
    for module in attn:
        module.kv_cache_manager = None
    for module in gdn:
        module.disable_decode()


def is_enabled(model) -> bool:
    _, gdn = _modules(model)
    return bool(gdn) and all(module.decode_enabled for module in gdn)
