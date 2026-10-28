"""Portable configuration for the trained Nova GDN2/GQA architecture."""
from transformers import PretrainedConfig

DEFAULT_OPTIONS = dict(d_model=1536, n_layers=26, n_heads=12, n_kv_heads=4,
    head_dim=128, ffn_hidden=6144, rope_theta=500000, rope_full_precision=True,
    layer_norm_eps=1e-5, layer_norm_bias=False, attn_qk_norm=True,
    attention_bias=False, ffn_bias=False, lm_head_bias=False, tie_word_embeddings=True,
    attention_backend="flash_2", enable_swa=True, swa_window=4096,
    swa_full_attention_first_layer=False, swa_full_attention_last_layer=False,
    gdn_expand_v=2.0, gdn_allow_neg_eigval=True, gdn_conv_size=4, gdn_conv_bias=False)


class NovaConfig(PretrainedConfig):
    model_type = "nova_gdn2"

    def __init__(self, native_size="main", native_options=None, vocab_size=65536,
                 hidden_size=1536, num_hidden_layers=26, max_position_embeddings=8192,
                 fast_decode=True, compute_dtype="bfloat16", tokenizer_vocab_size=None,
                 generation_context_policy="remaining", inference_backend="torch", **kwargs):
        # Architecture code is bundled, never imported from a training-machine path.
        kwargs.pop("native_repo", None)
        kwargs.pop("use_cache", None)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("pad_token_id", 0)
        super().__init__(use_cache=False, **kwargs)
        self.native_size = native_size
        self.native_options = dict(DEFAULT_OPTIONS if native_options is None else native_options)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.fast_decode = fast_decode
        self.compute_dtype = compute_dtype
        # Keep old exported configurations readable; new callers use torch/flash.
        self.inference_backend = "flash" if inference_backend == "cuda" else inference_backend
        self.tokenizer_vocab_size = tokenizer_vocab_size or vocab_size
        self.generation_context_policy = generation_context_policy
        if not 0 < self.tokenizer_vocab_size <= vocab_size:
            raise ValueError("Tokenizer vocabulary must fit within the model vocabulary")
        if compute_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported compute_dtype")
        if self.inference_backend not in ("torch", "flash", "cpu"):
            raise ValueError("inference_backend must be torch, flash or cpu")
        if generation_context_policy not in ("remaining", "error"):
            raise ValueError("generation_context_policy must be remaining or error")


NovaConfig.register_for_auto_class("AutoConfig")
