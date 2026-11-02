"""Architecture presets and options used to rebuild the accelerated inference model."""

from dataclasses import dataclass

from typing import Optional, Literal, Dict

ModelSize = Literal["main"]

SEED = 42


@dataclass(frozen=True)
class ModelShape:
    """A named model-size preset; individual fields can be overridden per checkpoint."""

    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    ffn_hidden: int
    n_kv_heads: Optional[int] = None


# Shape of the published Nova-1B checkpoint. NovaConfig.architecture_options may
# override any field when rebuilding a checkpoint.
MODEL_SHAPES: Dict[ModelSize, ModelShape] = {
    "main": ModelShape(
        d_model=1536,
        n_layers=26,
        n_heads=12,
        head_dim=128,
        ffn_hidden=6144,
        n_kv_heads=4,
    ),
}


@dataclass(frozen=True)
class ModelOptions:
    """Architecture options used to reconstruct an inference checkpoint."""

    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    head_dim: Optional[int] = None
    ffn_hidden: Optional[int] = None
    n_kv_heads: Optional[int] = None

    rope_theta: float = 500_000
    rope_full_precision: bool = True
    layer_norm_eps: float = 1e-5
    layer_norm_bias: bool = False
    attn_qk_norm: bool = True
    attention_bias: bool = False
    ffn_bias: bool = False
    lm_head_bias: bool = False
    tie_word_embeddings: bool = True
    attention_backend: Optional[str] = None

    enable_swa: bool = False
    swa_window: Optional[int] = None
    swa_context_ratio: int = 2
    swa_full_attention_first_layer: bool = False
    swa_full_attention_last_layer: bool = False

    # GDN2 layer dimensions of the published checkpoint. These intentionally
    # differ from the FLA GatedDeltaNet2 defaults (expand_v 1.0, allow_neg_eigval False).
    gdn_expand_v: float = 2.0
    gdn_allow_neg_eigval: bool = True
    gdn_conv_size: int = 4
    gdn_conv_bias: bool = False
