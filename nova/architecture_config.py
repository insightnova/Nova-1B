"""Inference architecture options; no training or curriculum configuration."""

from dataclasses import dataclass

from typing import Optional, Literal, Dict

ModelSize = Literal["pilot", "main", "pilot_para", "pilot_para3"]

SEED = 42

@dataclass(frozen=True)
class ModelShape:
    """A named model-size preset; individual fields can be overridden per run."""

    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    ffn_hidden: int
    n_kv_heads: Optional[int] = None

MODEL_SHAPES: Dict[ModelSize, ModelShape] = {
    # v2(2026-09-10): 18 -> 24. 비임베딩 340,092,304 -> 453,456,064 (x1.333).
    # 종횡비 d/L 이 56.9 -> 42.7 로 위 주석의 "통상 범위 50~130" 아래로 나간다.
    # GDN 은 recurrent scan 이라 실행 시간이 층 수에 직접 비례하므로 학습 시간도
    # 1.33배 이상 는다. 대신 GPU 2대를 써서 벽시계는 v1(1대)보다 짧아진다.
    "pilot": ModelShape(
        d_model=1024,
        n_layers=24,
        n_heads=8,
        head_dim=128,
        ffn_hidden=4096,
    ),
    "main": ModelShape(
        d_model=2048,
        n_layers=22,
        n_heads=16,
        head_dim=128,
        ffn_hidden=4096,
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

    # Keep the H1 comparison dimensions. FLA GDN2 defaults (1.0 / False)
    # are intentionally overridden; these fields now configure the GDN2 layer.
    gdn_expand_v: float = 2.0
    gdn_allow_neg_eigval: bool = True
    gdn_conv_size: int = 4
    gdn_conv_bias: bool = False

MODEL_SHAPES["pilot_para"] = MODEL_SHAPES["pilot"]

MODEL_SHAPES["pilot_para3"] = MODEL_SHAPES["pilot"]
