"""Backend selection and cached CUDA parity without external FlashAttention."""
import builtins

import pytest
import torch

from nova.configuration_nova import NovaConfig
from nova.modeling_nova import NovaForCausalLM
from test_cpu_backend import config


def test_default_backend_never_imports_cuda_extensions(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'flash_attn', 'fla', 'olmo_core', 'triton'}:
            raise AssertionError(f'Unexpected optional dependency: {name}')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    assert NovaConfig().inference_backend == 'torch'
    cfg = config()
    cfg.inference_backend = 'torch'
    model = NovaForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 4, 5, 6, 7]])
    cached = model.generate(ids, max_new_tokens=3, eos_token_id=[])
    model.config.fast_decode = False
    assert torch.equal(cached, model.generate(ids, max_new_tokens=3, eos_token_id=[]))


def test_legacy_cuda_configuration_selects_flash():
    assert NovaConfig(inference_backend='cuda').inference_backend == 'flash'
    with pytest.raises(ValueError, match='inference_backend'):
        NovaConfig(inference_backend='unknown')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_torch_cuda_cache_matches_full_forward(dtype):
    torch.manual_seed(47)
    cfg = config()
    cfg.inference_backend = 'torch'
    model = NovaForCausalLM(cfg).to(device='cuda', dtype=dtype).eval()
    # More than two windows: exercise cache eviction and absolute RoPE positions.
    ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10, 11, 12]], device='cuda')
    with torch.inference_mode():
        full = model.native(ids)
        model.native.enable_decode(max_seq_len=16)
        pieces = []
        for start, end in [(0, 5), (5, 8), (8, 9), (9, 10)]:
            pieces.append(model.native(ids[:, start:end])[:, -1])
        model.native.disable_decode()
    tolerance = 0.04 if dtype == torch.bfloat16 else 3e-5
    for result, end in zip(pieces, [5, 8, 9, 10]):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, full[:, end-1], atol=tolerance, rtol=tolerance)
