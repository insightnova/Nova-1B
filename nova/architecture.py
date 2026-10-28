"""GDN2 + attention hybrid model construction."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import torch

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
    flash_attn_api,
)
from .gdn2 import GatedDeltaNet2Config
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)

from .architecture_config import MODEL_SHAPES, ModelOptions, ModelShape, ModelSize, SEED

log = logging.getLogger(__name__)

def default_swa_window(sequence_length: int, context_ratio: int) -> int:
    if context_ratio <= 0:
        raise ValueError("swa_context_ratio must be positive")
    return sequence_length // context_ratio


def get_shape(
    size: ModelSize,
    *,
    options: ModelOptions,
) -> ModelShape:
    shape = MODEL_SHAPES[size]
    updates = {
        name: value
        for name, value in {
            "d_model": options.d_model,
            "n_layers": options.n_layers,
            "n_heads": options.n_heads,
            "head_dim": options.head_dim,
            "ffn_hidden": options.ffn_hidden,
        }.items()
        if value is not None
    }
    if options.n_kv_heads is not None:
        updates["n_kv_heads"] = options.n_kv_heads if options.n_kv_heads > 0 else None
    return replace(shape, **updates) if updates else shape


def resolve_swa_window(sequence_length: Optional[int], options: ModelOptions) -> Optional[int]:
    """Resolve an explicit SWA window or the opt-in ratio-based window."""

    if options.swa_window is not None:
        return options.swa_window if options.swa_window > 0 else None
    if not options.enable_swa:
        return None
    if sequence_length is None:
        raise ValueError("sequence_length is required when --swa is enabled")
    window = default_swa_window(sequence_length, options.swa_context_ratio)
    if window <= 0:
        raise ValueError("resolved SWA window must be positive")
    return window


def select_attention_backend(override: Optional[str] = None) -> AttentionBackendName:
    """Select an installed backend; fall back to PyTorch SDPA safely."""

    if override is not None:
        return AttentionBackendName(override)

    if not torch.cuda.is_available():
        return AttentionBackendName.torch

    name = torch.cuda.get_device_name(0)
    if ("B200" in name or "GB200" in name) and flash_attn_api.has_flash_attn_4():
        return AttentionBackendName.flash_4
    if ("H100" in name or "H200" in name) and flash_attn_api.has_flash_attn_3():
        return AttentionBackendName.flash_3
    if flash_attn_api.has_flash_attn_2():
        return AttentionBackendName.flash_2

    log.warning("flash-attn backend is unavailable; falling back to PyTorch SDPA")
    return AttentionBackendName.torch


def build_model(
    size: ModelSize,
    vocab_size: int,
    *,
    model_options: Optional[ModelOptions] = None,
    sequence_length: Optional[int] = None,
    init_seed: int = SEED,
    dtype: DType = DType.float32,
) -> TransformerConfig:
    options = model_options or ModelOptions()
    shape = get_shape(size, options=options)

    if shape.d_model <= 0 or shape.n_layers <= 0 or shape.n_heads <= 0 or shape.head_dim <= 0:
        raise ValueError("d_model, n_layers, n_heads, and head_dim must be positive")
    if shape.ffn_hidden <= 0:
        raise ValueError("ffn_hidden must be positive")
    if shape.n_heads * shape.head_dim != shape.d_model:
        raise ValueError("n_heads * head_dim must equal d_model")
    if shape.n_layers % 2:
        raise ValueError("GDN2:ATTN 1:1 pattern requires an even layer count")
    if shape.n_kv_heads is not None:
        if shape.n_kv_heads <= 0 or shape.n_heads % shape.n_kv_heads:
            raise ValueError("n_kv_heads must divide n_heads")
    if options.gdn_expand_v <= 0:
        raise ValueError("gdn_expand_v must be positive")
    if options.gdn_conv_size <= 0:
        raise ValueError("gdn_conv_size must be positive")
    if options.layer_norm_eps <= 0:
        raise ValueError("layer_norm_eps must be positive")

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=options.layer_norm_eps,
        bias=options.layer_norm_bias,
        dtype=dtype,
    )
    feed_forward = FeedForwardConfig(
        hidden_size=shape.ffn_hidden,
        bias=options.ffn_bias,
        dtype=dtype,
    )

    gdn_block = TransformerBlockConfig(
        name=TransformerBlockType.reordered_norm,
        sequence_mixer=GatedDeltaNet2Config(
            n_heads=shape.n_heads,
            head_dim=shape.head_dim,
            expand_v=options.gdn_expand_v,
            allow_neg_eigval=options.gdn_allow_neg_eigval,
            conv_size=options.gdn_conv_size,
            conv_bias=options.gdn_conv_bias,
            norm_eps=options.layer_norm_eps,
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )

    sliding_window = None
    if swa_window := resolve_swa_window(sequence_length, options):
        sliding_window = SlidingWindowAttentionConfig(
            pattern=[swa_window],
            force_full_attention_on_first_layer=options.swa_full_attention_first_layer,
            force_full_attention_on_last_layer=options.swa_full_attention_last_layer,
        )

    attn_block = TransformerBlockConfig(
        name=TransformerBlockType.reordered_norm,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=shape.n_heads,
            n_kv_heads=shape.n_kv_heads,
            head_dim=shape.head_dim,
            bias=options.attention_bias,
            rope=RoPEConfig(
                name=RoPEType.default,
                theta=options.rope_theta,
                full_precision=options.rope_full_precision,
            ),
            qk_norm=layer_norm if options.attn_qk_norm else None,
            backend=select_attention_backend(options.attention_backend),
            sliding_window=sliding_window,
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )

    return TransformerConfig(
        d_model=shape.d_model,
        vocab_size=vocab_size,
        n_layers=shape.n_layers,
        block={"gdn2": gdn_block, "attn": attn_block},
        block_pattern=["gdn2", "attn"],
        lm_head=LMHeadConfig(
            layer_norm=layer_norm,
            bias=options.lm_head_bias,
            dtype=dtype,
        ),
        dtype=dtype,
        tie_word_embeddings=options.tie_word_embeddings,
        init_seed=init_seed,
    )
