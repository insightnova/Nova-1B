"""OLMo SequenceMixer adapter for the upstream FLA GatedDeltaNet2 layer.

All recurrence and convolution kernels come from flash-linear-attention 0.5.2.
This module owns OLMo initialization, document packing and inference lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig

from .sequence_utils import flatten_documents
from .gdn2_spec import GDN2Spec

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class GatedDeltaNet2(SequenceMixer):
    """Tensor-in/tensor-out wrapper around ``fla.layers.GatedDeltaNet2``.

    Each mixer owns one FLA Cache with a local layer index of zero. The cache
    holds both recurrent and short-convolution states and is never serialized.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        head_dim: int | None = None,
        n_v_heads: int | None = None,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ) -> None:
        super().__init__()
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        self.spec = GDN2Spec(
            d_model=d_model, n_heads=n_heads,
            head_dim=head_dim if head_dim is not None else d_model // n_heads,
            n_v_heads=n_v_heads, expand_v=expand_v,
            conv_size=conv_size, conv_bias=conv_bias,
        )
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        from fla.layers.gdn2 import GatedDeltaNet2 as FLAGatedDeltaNet2

        # FLA accepts **kwargs but does not implement OLMo's init_device/dtype
        # arguments. Use PyTorch's device context, including for meta tensors.
        with torch.device(init_device):
            self.layer = FLAGatedDeltaNet2(
                hidden_size=d_model, num_heads=n_heads,
                num_v_heads=self.spec.value_heads, head_dim=self.spec.head_dim,
                expand_v=expand_v, allow_neg_eigval=allow_neg_eigval,
                use_short_conv=True, conv_size=conv_size, conv_bias=conv_bias,
                norm_eps=norm_eps, mode="chunk", layer_idx=0,
            )
        self.layer.to(dtype=dtype)
        # These two parameters must retain their initialization precision.
        for name in ("A_log", "dt_bias"):
            param = nn.Parameter(getattr(self.layer, name).float())
            # Do not attach optimizer-only attributes before registration.
            # Transformers' init_empty_weights reconstructs Parameters from
            # their __dict__; torch.nn.Parameter rejects _no_weight_decay.
            setattr(self.layer, name, param)
        self.d_model = d_model
        self._decode_cache = None
        self._decode_batch_size: int | None = None
        self._decode_max_seq_len: int | None = None
        self._decode_tokens = 0

    @property
    def decode_enabled(self) -> bool:
        return self._decode_cache is not None

    def enable_decode(self, batch_size: int, max_seq_len: int) -> None:
        if self.training:
            raise ValueError("Call model.eval() before enabling incremental decoding")
        if batch_size <= 0 or max_seq_len <= 0:
            raise ValueError("batch_size and max_seq_len must be positive")
        from fla.models.utils import Cache

        self._decode_cache = Cache()
        self._decode_batch_size = batch_size
        self._decode_max_seq_len = max_seq_len
        self._decode_tokens = 0

    def disable_decode(self) -> None:
        self._decode_cache = None
        self._decode_batch_size = None
        self._decode_max_seq_len = None
        self._decode_tokens = 0

    def train(self, mode: bool = True):
        if mode and self.decode_enabled:
            raise ValueError("Call fast_decode.disable(model) before returning to training")
        return super().train(mode)

    def forward(self, x: torch.Tensor, cu_doc_lens=None, **kwargs) -> torch.Tensor:
        # OLMo forwards attention-specific kwargs (positions, max_doc_len, ...).
        # Passing them to FLA as a padding attention_mask would lose doc resets.
        del kwargs
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Expected nonempty x with shape [batch, sequence, d_model]")
        original_shape = x.shape
        if self.decode_enabled:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Cached decoding requires eval() and torch.no_grad()/inference_mode()")
            if cu_doc_lens is not None:
                raise ValueError("Cached decoding does not accept packed documents; reset per prompt batch")
            if x.shape[0] != self._decode_batch_size:
                raise ValueError("Decode batch size changed; call fast_decode.reset() first")
            if self._decode_tokens + x.shape[1] > self._decode_max_seq_len:
                raise ValueError("Decode context exceeds max_seq_len; allocate a larger cache")
        else:
            x = flatten_documents(x, cu_doc_lens)

        # FLA 0.5.2 switches short eval sequences to a forward-only recurrent
        # kernel even when gradients are enabled. Explicitly use its training
        # branch for eval-with-grad, without changing the public wrapper mode.
        original_training = self.layer.training
        if torch.is_grad_enabled() and not original_training:
            self.layer.training = True
        try:
            out, _, _ = self.layer(
                hidden_states=x,
                attention_mask=None,
                cu_seqlens=cu_doc_lens,
                past_key_values=self._decode_cache,
                use_cache=self.decode_enabled,
            )
        finally:
            self.layer.training = original_training
        if self.decode_enabled:
            self._decode_tokens += original_shape[1]
        return out.reshape(original_shape)

    @torch.no_grad()
    def init_weights(
        self, *, init_method: "InitMethod", d_model: int, block_idx: int,
        num_blocks: int, std: float = 0.02, generator: torch.Generator | None = None,
    ) -> None:
        """Reinitialize every parameter after OLMo's meta -> to_empty step."""
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError("fan_in initialization is not supported for GDN2")
        if init_method == InitMethod.normalized:
            std = d_model ** -0.5
        layer = self.layer
        for module in layer.modules():
            if isinstance(module, nn.Linear) and module is not layer.o_proj:
                init_linear(module, std=std, generator=generator)
        for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
            init_linear(conv, std=std, generator=generator)
        layer.o_norm.reset_parameters()
        layer.A_log.uniform_(1, 16, generator=generator).log_()
        # FLA's per-key-channel time steps, initialized through inverse softplus.
        dt = torch.empty_like(layer.dt_bias, dtype=torch.float32).uniform_(
            math.log(0.001), math.log(0.1), generator=generator,
        ).exp_().clamp_min_(1e-4)
        layer.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        if init_method in (InitMethod.llama, InitMethod.normalized):
            std /= math.sqrt(2 * num_blocks)
        elif init_method == InitMethod.llama_depth:
            std /= math.sqrt(2 * (block_idx + 1))
        init_linear(layer.o_proj, std=std, generator=generator)
        self.disable_decode()

    def apply_tp(self, tp_mesh, **kwargs):
        raise NotImplementedError("GDN2 adapter does not support tensor parallelism; use FSDP")

    def apply_cp(self, cp_mesh, **kwargs):
        raise NotImplementedError("GDN2 adapter does not support context parallelism; use FSDP")

    def num_flops_per_token(self, seq_len: int) -> int:
        """Analytical forward+backward estimate; excludes chunk-solver overhead."""
        del seq_len
        linear = sum(m.weight.numel() for m in self.layer.modules() if isinstance(m, nn.Linear))
        conv = self.spec.conv_size * (2 * self.spec.key_dim + self.spec.value_dim)
        # State decay, erase read, outer product, and query read.
        return 3 * (2 * (linear + conv) + 7 * self.spec.state_elements)


@SequenceMixerConfig.register("gated_delta_net2")
@dataclass
class GatedDeltaNet2Config(SequenceMixerConfig[GatedDeltaNet2]):
    n_heads: int = 16
    n_v_heads: int | None = None
    head_dim: int | None = None
    expand_v: float = 2.0
    allow_neg_eigval: bool = True
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-5
    dtype: DType = DType.float32

    def spec(self, d_model: int) -> GDN2Spec:
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        return GDN2Spec(
            d_model=d_model, n_heads=self.n_heads, n_v_heads=self.n_v_heads,
            head_dim=self.head_dim if self.head_dim is not None else d_model // self.n_heads,
            expand_v=self.expand_v, conv_size=self.conv_size, conv_bias=self.conv_bias,
        )

    def num_params(self, d_model: int) -> int:
        return self.spec(d_model).num_params

    def build(self, d_model: int, *, layer_idx: int, n_layers: int,
              init_device: str = "cpu", cache=None) -> GatedDeltaNet2:
        del layer_idx, n_layers, cache  # FLA keeps a private one-layer cache per mixer.
        self.spec(d_model)
        return GatedDeltaNet2(
            d_model=d_model, n_heads=self.n_heads, n_v_heads=self.n_v_heads,
            head_dim=self.head_dim, expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval, conv_size=self.conv_size,
            conv_bias=self.conv_bias, norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(), init_device=init_device,
        )
