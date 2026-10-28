"""Numerical and cache tests for the real CPU architecture at reduced size."""
import pytest
import torch
from transformers import AutoModelForCausalLM

import nova
from nova.configuration_nova import NovaConfig, DEFAULT_OPTIONS
from nova.modeling_nova import NovaForCausalLM
from nova.cpu_backend import Attention, GDN2Layer, recurrent_gdn2


def config():
    options = dict(DEFAULT_OPTIONS, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1,
                   head_dim=8, ffn_hidden=32, swa_window=4)
    return NovaConfig(native_options=options, vocab_size=32, hidden_size=16,
                      num_hidden_layers=2, max_position_embeddings=32,
                      inference_backend='cpu', compute_dtype='float32')


def test_recurrence_matches_matrix_equation():
    torch.manual_seed(11)
    q, k, decay, erase = [torch.randn(2, 5, 2, 3) for _ in range(4)]
    v, write = [torch.randn(2, 5, 2, 4) for _ in range(2)]
    decay, erase, write = -decay.abs(), 2*erase.sigmoid(), write.sigmoid()
    state = torch.randn(2, 2, 3, 4)
    result, final = recurrent_gdn2(q, k, v, decay, erase, write, state)
    expected, h = [], state.clone()
    for t in range(5):
        key = k[:, t] / (k[:, t].square().sum(-1, keepdim=True)+1e-6).sqrt()
        query = q[:, t] / (q[:, t].square().sum(-1, keepdim=True)+1e-6).sqrt() / 3**0.5
        transform = torch.eye(3)-key.unsqueeze(-1) @ (erase[:, t]*key).unsqueeze(-2)
        h = transform @ torch.diag_embed(decay[:, t].exp()) @ h
        h = h + key.unsqueeze(-1) @ (write[:, t]*v[:, t]).unsqueeze(-2)
        expected.append((query.unsqueeze(-2) @ h).squeeze(-2))
    torch.testing.assert_close(result, torch.stack(expected, 1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(final, h, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('kernel', [1, 4])
def test_gdn_cache(kernel):
    torch.manual_seed(13)
    options = config().native_options
    options['gdn_conv_size'] = kernel
    layer = GDN2Layer(options).eval()
    x = torch.randn(2, 11, 16)
    with torch.no_grad():
        whole = layer(x)
        layer.cached = True
        chunks = torch.cat([layer(x[:, :5]), layer(x[:, 5:8]), layer(x[:, 8:])], 1)
    torch.testing.assert_close(chunks, whole, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('window', [1, 4, 0])
def test_attention_cache_and_window(window):
    torch.manual_seed(17)
    options = config().native_options
    options['swa_window'] = window
    layer = Attention(options).eval()
    x = torch.randn(2, 11, 16)
    with torch.no_grad():
        whole = layer(x)
        layer.cached = True
        chunks = torch.cat([layer(x[:, :5]), layer(x[:, 5:8]), layer(x[:, 8:])], 1)
        layer.cached = False
        perturbed = x.clone()
        perturbed[:, 0] += 20
        changed = layer(perturbed)
    torch.testing.assert_close(chunks, whole, atol=2e-6, rtol=2e-5)
    if window:
        torch.testing.assert_close(whole[:, window:], changed[:, window:], atol=0, rtol=0)


def test_real_cpu_architecture_save_load_and_generate(tmp_path):
    torch.manual_seed(19)
    nova.register()
    model = NovaForCausalLM(config()).eval()
    ids = torch.tensor([[1, 4, 5, 6, 7], [1, 8, 9, 10, 11]])
    cached = model.generate(ids, max_new_tokens=5, eos_token_id=[], generation_batch_size=2)
    repeated = model.generate(ids, max_new_tokens=5, eos_token_id=[], generation_batch_size=2)
    assert torch.equal(cached, repeated)
    assert model.native._cache_limit is None
    model.config.fast_decode = False
    assert torch.equal(cached, model.generate(ids, max_new_tokens=5, eos_token_id=[]))
    model.config.fast_decode = True
    model.save_pretrained(tmp_path)
    loaded, info = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True, output_loading_info=True)
    assert not any(info.values())
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    assert torch.equal(cached, loaded.generate(ids, max_new_tokens=5, eos_token_id=[], generation_batch_size=2))


def test_cpu_cache_cleanup_on_failure(monkeypatch):
    model = NovaForCausalLM(config()).eval()
    def fail(*args, **kwargs):
        raise RuntimeError('test failure')
    monkeypatch.setattr(model.native.blocks['1'], 'forward', fail)
    with pytest.raises(RuntimeError, match='test failure'):
        model.generate(torch.tensor([[1, 3]]), max_new_tokens=1)
    assert model.native._cache_limit is None
    assert model.native.blocks['0'].attention.layer.state is None
