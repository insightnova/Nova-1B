"""Nova-1B inference. Importing this package does not load CUDA kernels."""
from .configuration_nova import NovaConfig


def register():
    """Enable local AutoClass loading after installing nova-1b-inference."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from .modeling_nova import NovaForCausalLM
    AutoConfig.register(NovaConfig.model_type, NovaConfig, exist_ok=True)
    AutoModelForCausalLM.register(NovaConfig, NovaForCausalLM, exist_ok=True)


__all__ = ["NovaConfig", "register"]
