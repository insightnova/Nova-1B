"""Inference adapter with standard Transformers checkpoint/Hub loading."""
from contextlib import nullcontext
import importlib.util
import sys
from typing import TYPE_CHECKING

import torch
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_nova import NovaConfig
from .generation_nova import generate_batched, validate_inputs

# Transformers 4.57 copies only direct relative imports for LOCAL remote-code
# bundles. Declare the complete native dependency closure without importing GPU
# libraries at module import time. Hub downloads follow the same file manifest.
if TYPE_CHECKING:
    from .fast_decode import enable
    from .gdn2 import GatedDeltaNet2
    from .gdn2_spec import GDN2Spec
    from .sequence_utils import flatten_documents


def build_native(config):
    if config.inference_backend in ("torch", "cpu"):
        from .cpu_backend import TorchBackbone
        return TorchBackbone(config)
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Flash inference requires Linux + NVIDIA CUDA. Use inference_backend='torch' without FlashAttention, or example_cpu.py for CPU.")
    for package in ("olmo_core", "fla", "flash_attn"):
        if importlib.util.find_spec(package) is None:
            raise ImportError("Flash backend dependencies are missing. Install .[flash] and flash-attn as described in README, or select inference_backend='torch'.")
    from .architecture import build_model
    from .architecture_config import ModelOptions
    architecture_config = build_model(config.native_size, vocab_size=config.vocab_size,
        model_options=ModelOptions(**dict(config.native_options, attention_backend="flash_2")),
        sequence_length=config.max_position_embeddings)
    return architecture_config.build(init_device="cpu")


class NovaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NovaConfig
    base_model_prefix = "native"
    _supports_cache_class = False
    _tied_weights_keys = ["native.lm_head.w_out.weight"]
    # Transformers 4.57.6 applies this also for BF16 from_pretrained loading.
    _keep_in_fp32_modules_strict = ["A_log", "dt_bias"]

    def __init__(self, config):
        super().__init__(config)
        self.native = build_native(config)
        self.post_init()

    def _init_weights(self, module):
        # Native architecture owns initialization; published inference requires weights.
        pass

    def get_input_embeddings(self):
        return self.native.embeddings

    def get_output_embeddings(self):
        return self.native.lm_head.w_out

    def set_input_embeddings(self, value):
        self.native.embeddings = value

    def set_output_embeddings(self, value):
        self.native.lm_head.w_out = value

    def _apply(self, function, recurse=True):
        masters = [(n, p.detach()) for n, p in self.native.named_parameters()
                   if n.endswith((".A_log", ".dt_bias"))]
        result = super()._apply(function, recurse=recurse)
        for name, original in masters:
            parameter = self.native.get_parameter(name)
            parameter.data = original.to(device=parameter.device, dtype=torch.float32)
        return result

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Native cached decoding calls modules directly; arbitrary Accelerate dispatch
        # and offload hooks are not a supported execution path.
        device_map = kwargs.get("device_map")
        if device_map is not None:
            raise ValueError("device_map/offloading is not supported. Load then call model.to('cuda').")
        if kwargs.get("quantization_config") is not None or kwargs.get("load_in_4bit") or kwargs.get("load_in_8bit"):
            raise ValueError("Quantized Nova inference is not validated in this release")
        return_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(pretrained_model_name_or_path, *args,
                                             output_loading_info=True, **kwargs)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete or incompatible Nova checkpoint: {info}")
        return (model, info) if return_info else model

    def _native_logits(self, ids):
        context = (torch.autocast("cuda", dtype=getattr(torch, self.config.compute_dtype))
                   if ids.is_cuda and self.config.compute_dtype != "float32"
                   and not torch.is_autocast_enabled("cuda") else nullcontext())
        with context:
            result = self.native(ids, return_logits=True)
        logits = result if isinstance(result, torch.Tensor) else result.logits
        return logits[..., :self.config.tokenizer_vocab_size]

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, return_dict=True,
                logits_to_keep=0, labels=None, token_type_ids=None, **kwargs):
        if labels is not None:
            raise ValueError("This distribution provides inference only")
        if past_key_values is not None or use_cache:
            raise ValueError("Native caches are managed by generate(); use_cache must be False")
        if kwargs:
            raise TypeError(f"Unsupported forward arguments: {sorted(kwargs)}")
        if token_type_ids is not None and (token_type_ids.shape != input_ids.shape or torch.any(token_type_ids != 0)):
            raise ValueError("Nova does not use nonzero token_type_ids")
        mask = validate_inputs(input_ids, attention_mask, self.config.max_position_embeddings)
        aligned = []
        for i in range(input_ids.shape[0]):
            positions = mask[i].nonzero(as_tuple=True)[0]
            if position_ids is not None and not torch.equal(position_ids[i, positions],
                    torch.arange(len(positions), device=input_ids.device)):
                raise ValueError("Positions must start at zero for each unpadded prompt")
            logits = self._native_logits(input_ids[i, positions].unsqueeze(0))[0]
            aligned.append(logits.new_zeros(input_ids.shape[1], logits.shape[-1]).index_copy(0, positions, logits))
        logits = torch.stack(aligned)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return CausalLMOutput(logits=logits) if return_dict else (logits,)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, generation_config=None, **kwargs):
        settings = generation_config or self.generation_config
        token_types = kwargs.pop("token_type_ids", None)
        if token_types is not None and (input_ids is None or token_types.shape != input_ids.shape or torch.any(token_types != 0)):
            raise ValueError("Nova does not use nonzero token_type_ids")
        supported = {"max_new_tokens", "max_length", "do_sample", "temperature", "top_p", "top_k",
                     "eos_token_id", "pad_token_id", "num_return_sequences", "generation_batch_size",
                     "num_beams", "num_beam_groups", "return_dict_in_generate", "output_scores",
                     "output_logits", "output_attentions", "output_hidden_states",
                     "repetition_penalty", "no_repeat_ngram_size", "min_new_tokens", "min_length"}
        unknown = set(kwargs) - supported
        if unknown:
            raise TypeError(f"Unsupported generation arguments: {sorted(unknown)}")
        def option(key, default=None):
            value = kwargs.get(key, getattr(settings, key, default))
            return default if value is None else value
        for key, default in {"num_beams": 1, "num_beam_groups": 1, "return_dict_in_generate": False,
                             "output_scores": False, "output_logits": False, "output_attentions": False,
                             "output_hidden_states": False, "repetition_penalty": 1.0,
                             "no_repeat_ngram_size": 0, "min_new_tokens": 0, "min_length": 0}.items():
            if option(key, default) != default:
                raise ValueError(f"{key} is not supported; expected {default!r}")
        # A supplied GenerationConfig must not silently enable an unsupported
        # processor. Native cache ownership is independent of HF's use_cache flag.
        defaults = GenerationConfig().to_dict()
        handled = supported | {"use_cache", "bos_token_id", "transformers_version"}
        for key, value in settings.to_dict().items():
            if key not in handled and not key.startswith("_") and value != defaults.get(key):
                raise ValueError(f"generation_config.{key} is not supported")
        mask = validate_inputs(input_ids, attention_mask, self.config.max_position_embeddings)
        budget = option("max_new_tokens")
        if budget is None:
            budget = option("max_length", self.config.max_position_embeddings) - input_ids.shape[1]
        if type(budget) is not int or budget <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        returns = option("num_return_sequences", 1)
        limit = option("generation_batch_size", 1)
        if type(returns) is not int or returns < 1 or type(limit) is not int or limit < 1:
            raise ValueError("Return count and generation batch size must be positive integers")
        if returns > 1 and not option("do_sample", False):
            raise ValueError("Multiple return sequences require do_sample=True")
        ids = input_ids.repeat_interleave(returns, dim=0)
        mask = mask.repeat_interleave(returns, dim=0)
        eos = option("eos_token_id", self.config.eos_token_id)
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])
        pad = option("pad_token_id", self.config.pad_token_id)
        return generate_batched(self, ids, mask, budget, eos_ids, pad, option, limit)


NovaForCausalLM.register_for_auto_class("AutoModelForCausalLM")
