"""Nova-1B inference. Importing this package does not load CUDA kernels."""
from .configuration_nova import AUTO_MAP, NovaConfig


def register():
    """Enable local AutoClass loading after installing nova-1b-inference.

    Maps ``AutoConfig``, ``AutoModel`` and ``AutoModelForCausalLM`` to the bundled
    implementation so exported folders and Hub repositories load with
    ``trust_remote_code=False``.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from .modeling_nova import NovaForCausalLM, NovaModel
    AutoConfig.register(NovaConfig.model_type, NovaConfig, exist_ok=True)
    AutoModel.register(NovaConfig, NovaModel, exist_ok=True)
    AutoModelForCausalLM.register(NovaConfig, NovaForCausalLM, exist_ok=True)


__all__ = ["AUTO_MAP", "NovaConfig", "register"]
