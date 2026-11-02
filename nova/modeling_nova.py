"""Inference adapter with standard Transformers checkpoint/Hub loading.

Two AutoClass entry points share one backbone module and one checkpoint layout:

* ``NovaModel`` (``AutoModel``): final-normalized hidden states, optional
  per-layer hidden states, no vocabulary projection.
* ``NovaForCausalLM`` (``AutoModelForCausalLM``): logits and ``generate()``.
"""
from contextlib import nullcontext
import importlib.util
import sys
from typing import TYPE_CHECKING
import warnings

import torch
from torch import nn
import transformers
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from .configuration_nova import AUTO_MAP, NovaConfig
from .generation_nova import generate_batched, validate_inputs

# Transformers resolves remote-code dependencies by scanning relative imports
# (also inside this TYPE_CHECKING block) and downloads the whole closure from the
# Hub. Declare it here without importing GPU libraries at module import time; the
# flash-only modules guard their own optional imports.
if TYPE_CHECKING:
    from .architecture import build_model
    from .architecture_config import ModelOptions
    from .cpu_backend import TorchBackbone
    from .fast_decode import enable
    from .gdn2 import GatedDeltaNet2
    from .gdn2_spec import GDN2Spec
    from .sequence_utils import flatten_documents

VALIDATED_TRANSFORMERS = "4.57"
if not transformers.__version__.startswith(VALIDATED_TRANSFORMERS + "."):
    warnings.warn(
        f"Nova-1B inference code was validated with transformers {VALIDATED_TRANSFORMERS}.x; "
        f"found {transformers.__version__}. Pin transformers=={VALIDATED_TRANSFORMERS}.6 if "
        "loading or generation misbehaves.", stacklevel=2)


def build_backbone(config):
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
    architecture_config = build_model(config.architecture_preset, vocab_size=config.vocab_size,
        model_options=ModelOptions(**dict(config.architecture_options, attention_backend="flash_2")),
        sequence_length=config.max_position_embeddings)
    return architecture_config.build(init_device="cpu")


def _reject_token_type_ids(token_type_ids, input_ids):
    if token_type_ids is not None and (input_ids is None or token_type_ids.shape != input_ids.shape
                                       or torch.any(token_type_ids != 0)):
        raise ValueError("Nova does not use nonzero token_type_ids")


def _prompt_rows(input_ids, attention_mask, position_ids, context):
    """Validate a left/right padded batch and yield (row, unpadded positions)."""
    mask = validate_inputs(input_ids, attention_mask, context)
    for i in range(input_ids.shape[0]):
        positions = mask[i].nonzero(as_tuple=True)[0]
        if position_ids is not None and not torch.equal(position_ids[i, positions],
                torch.arange(len(positions), device=input_ids.device)):
            raise ValueError("Positions must start at zero for each unpadded prompt")
        yield i, positions


def _scatter_rows(rows, length):
    """Place per-prompt [tokens, features] results back into a zero-padded batch."""
    return torch.stack([values.new_zeros(length, values.shape[-1]).index_copy(0, positions, values)
                        for positions, values in rows])


class NovaPreTrainedModel(PreTrainedModel):
    """Shared checkpoint layout, precision rules and loading checks."""

    config_class = NovaConfig
    base_model_prefix = "backbone"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _supports_cache_class = False
    _tied_weights_keys = ["backbone.lm_head.w_out.weight"]
    # Transformers 4.57.6 applies this also for BF16 from_pretrained loading.
    _keep_in_fp32_modules_strict = ["A_log", "dt_bias"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = build_backbone(config)
        self.post_init()

    def _init_weights(self, module):
        # Published inference requires trained weights; nothing is randomly initialized.
        pass

    def get_input_embeddings(self):
        return self.backbone.embeddings

    def get_output_embeddings(self):
        return self.backbone.lm_head.w_out

    def set_input_embeddings(self, value):
        self.backbone.embeddings = value

    def set_output_embeddings(self, value):
        self.backbone.lm_head.w_out = value

    def _apply(self, function, recurse=True):
        masters = [(n, p.detach()) for n, p in self.backbone.named_parameters()
                   if n.endswith((".A_log", ".dt_bias"))]
        result = super()._apply(function, recurse=recurse)
        for name, original in masters:
            parameter = self.backbone.get_parameter(name)
            parameter.data = original.to(device=parameter.device, dtype=torch.float32)
        return result

    def save_pretrained(self, *args, **kwargs):
        # Every saved bundle advertises all remote-code entry points, not only the
        # class that happened to be saved.
        self.config.auto_map = dict(getattr(self.config, "auto_map", None) or {}, **AUTO_MAP)
        return super().save_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Cached decoding calls backbone modules directly; arbitrary Accelerate dispatch
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

    def _compute_context(self, ids):
        if (ids.is_cuda and self.config.compute_dtype != "float32"
                and not torch.is_autocast_enabled("cuda")):
            return torch.autocast("cuda", dtype=getattr(torch, self.config.compute_dtype))
        return nullcontext()

    def _backbone_logits(self, ids):
        with self._compute_context(ids):
            result = self.backbone(ids, return_logits=True)
        logits = result if isinstance(result, torch.Tensor) else result.logits
        return logits[..., :self.config.tokenizer_vocab_size]

    def _backbone_hidden(self, ids, collect_layers):
        """Final-normalized hidden states for one unpadded prompt batch.

        Per-layer states (embedding output followed by each block output) are
        captured with forward hooks so both the torch and the flash backbone,
        which share the ``embeddings`` / ``blocks`` / ``lm_head.norm`` layout,
        expose the same AutoModel outputs.
        """
        backbone = self.backbone
        captured = {"final": None, "layers": []}
        handles = []

        def record(_module, _inputs, output):
            captured["layers"].append(output if isinstance(output, torch.Tensor) else output[0])

        def record_final(_module, _inputs, output):
            captured["final"] = output

        torch_backend = self.config.inference_backend in ("torch", "cpu")
        try:
            if collect_layers:
                handles.append(backbone.embeddings.register_forward_hook(record))
                blocks = backbone.blocks.values() if isinstance(backbone.blocks, nn.ModuleDict) else backbone.blocks
                for block in blocks:
                    handles.append(block.register_forward_hook(record))
            if not torch_backend:
                handles.append(backbone.lm_head.norm.register_forward_hook(record_final))
            with self._compute_context(ids):
                if torch_backend:
                    captured["final"] = backbone(ids, return_logits=False)
                else:
                    backbone(ids, return_logits=True)
        finally:
            for handle in handles:
                handle.remove()
        if captured["final"] is None:
            raise RuntimeError("Backbone did not expose its final hidden states")
        if collect_layers and len(captured["layers"]) != self.config.num_hidden_layers + 1:
            raise RuntimeError(f"Expected {self.config.num_hidden_layers + 1} hidden states, "
                               f"captured {len(captured['layers'])}")
        return captured["final"], captured["layers"]


class NovaModel(NovaPreTrainedModel):
    """Backbone for ``AutoModel``: hidden states without the vocabulary projection.

    ``last_hidden_state`` is the output of the final RMSNorm that precedes the
    (tied) output projection, so ``model.get_output_embeddings()(last_hidden_state)``
    reproduces ``NovaForCausalLM`` logits. ``hidden_states`` follows the Transformers
    convention: embedding output first, then one entry per block, before the final norm.
    Padded positions are zero-filled.
    """

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=False,
                output_attentions=None, output_hidden_states=None, return_dict=True,
                token_type_ids=None, **kwargs):
        if inputs_embeds is not None:
            raise ValueError("Nova accepts token IDs only; inputs_embeds is not supported")
        if past_key_values is not None or use_cache:
            raise ValueError("Generation caches are managed by generate(); use_cache must be False")
        if output_attentions:
            raise ValueError("Nova does not expose attention weights (GDN2 layers have none)")
        if kwargs:
            raise TypeError(f"Unsupported forward arguments: {sorted(kwargs)}")
        _reject_token_type_ids(token_type_ids, input_ids)
        collect = bool(self.config.output_hidden_states if output_hidden_states is None
                       else output_hidden_states)
        finals, layers = [], []
        for i, positions in _prompt_rows(input_ids, attention_mask, position_ids,
                                         self.config.max_position_embeddings):
            final, states = self._backbone_hidden(input_ids[i, positions].unsqueeze(0), collect)
            finals.append((positions, final[0]))
            layers.append([(positions, state[0]) for state in states])
        length = input_ids.shape[1]
        last_hidden_state = _scatter_rows(finals, length)
        hidden_states = None
        if collect:
            hidden_states = tuple(_scatter_rows([row[k] for row in layers], length)
                                  for k in range(len(layers[0])))
        if not return_dict:
            return tuple(value for value in (last_hidden_state, hidden_states) if value is not None)
        return BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


class NovaForCausalLM(NovaPreTrainedModel, GenerationMixin):
    """``AutoModelForCausalLM`` entry point: logits and cached generation."""

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, return_dict=True,
                logits_to_keep=0, labels=None, token_type_ids=None, **kwargs):
        if labels is not None:
            raise ValueError("This distribution provides inference only")
        if past_key_values is not None or use_cache:
            raise ValueError("Generation caches are managed by generate(); use_cache must be False")
        if kwargs:
            raise TypeError(f"Unsupported forward arguments: {sorted(kwargs)}")
        _reject_token_type_ids(token_type_ids, input_ids)
        rows = [(positions, self._backbone_logits(input_ids[i, positions].unsqueeze(0))[0])
                for i, positions in _prompt_rows(input_ids, attention_mask, position_ids,
                                                 self.config.max_position_embeddings)]
        logits = _scatter_rows(rows, input_ids.shape[1])
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return CausalLMOutput(logits=logits) if return_dict else (logits,)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, generation_config=None, **kwargs):
        settings = generation_config or self.generation_config
        _reject_token_type_ids(kwargs.pop("token_type_ids", None), input_ids)
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
        # processor. Backbone cache ownership is independent of HF's use_cache flag.
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


NovaModel.register_for_auto_class("AutoModel")
NovaForCausalLM.register_for_auto_class("AutoModelForCausalLM")
