"""Inference-only PyTorch implementation for CPU and CUDA without flash-attn.

Preserves the published OLMo/FLA parameter names. GDN2 uses its matrix-state
recurrence (independent erase/write gates), not an approximation or GDN1.
See THIRD_PARTY_NOTICES.txt for the upstream reference and license.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .configuration_nova import DEFAULT_OPTIONS


class RMSNorm(nn.Module):
    def __init__(self, size, eps, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None
        self.eps = eps

    def forward(self, x, gate=None):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.float()
        if self.bias is not None:
            y = y + self.bias.float()
        if gate is not None:
            y = y * gate.float().sigmoid()
        return y.to(x.dtype)


def recurrent_gdn2(q, k, v, decay, erase, write, state=None):
    """FP32 state, q/k L2 normalization and the FLA default attention scale."""
    dtype = v.dtype
    q, k, v, decay, erase, write = (a.float() for a in (q, k, v, decay, erase, write))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    if state is None:
        state = v.new_zeros(v.shape[0], v.shape[2], k.shape[-1], v.shape[-1])
    outputs = []
    for t in range(q.shape[1]):
        key = k[:, t]
        state = state * decay[:, t].exp().unsqueeze(-1)
        removed = ((erase[:, t] * key).unsqueeze(-1) * state).sum(-2)
        value = write[:, t] * v[:, t] - removed
        state = state + key.unsqueeze(-1) * value.unsqueeze(-2)
        outputs.append((q[:, t].unsqueeze(-1) * state).sum(-2))
    return torch.stack(outputs, dim=1).to(dtype), state


class GDN2Layer(nn.Module):
    def __init__(self, o):
        super().__init__()
        d, h, k = o['d_model'], o['n_heads'], o['head_dim']
        v = int(k * o['gdn_expand_v'])
        self.heads, self.key_size, self.value_size = h, k, v
        self.kernel = o['gdn_conv_size']
        self.negative = o['gdn_allow_neg_eigval']
        for name, size in [('q', h*k), ('k', h*k), ('v', h*v)]:
            setattr(self, name+'_proj', nn.Linear(d, size, bias=False))
            setattr(self, name+'_conv1d', nn.Conv1d(size, size, self.kernel,
                    groups=size, bias=o['gdn_conv_bias']))
        self.f_proj = nn.Sequential(nn.Linear(d, v, bias=False), nn.Linear(v, h*k, bias=False))
        self.b_proj = nn.Linear(d, h*k, bias=False)
        self.w_proj = nn.Linear(d, h*v, bias=False)
        self.A_log = nn.Parameter(torch.zeros(h))
        self.dt_bias = nn.Parameter(torch.zeros(h*k))
        self.g_proj = nn.Sequential(nn.Linear(d, v, bias=False), nn.Linear(v, h*v, bias=True))
        self.o_norm = RMSNorm(v, o['layer_norm_eps'])
        self.o_proj = nn.Linear(h*v, d, bias=False)
        self.cached = False
        self.clear()

    def clear(self):
        self.state = None
        self.conv_states = {}

    def _conv(self, name, x):
        projected = getattr(self, name+'_proj')(x).transpose(1, 2)
        prior = self.conv_states.get(name) if self.cached else None
        if prior is None:
            prior = projected.new_zeros(*projected.shape[:2], self.kernel-1)
        joined = torch.cat((prior, projected), dim=-1)
        if self.cached:
            self.conv_states[name] = joined[..., -(self.kernel-1):].clone() if self.kernel > 1 else joined[..., :0]
        return F.silu(getattr(self, name+'_conv1d')(joined)).transpose(1, 2)

    def forward(self, x):
        batch, length, _ = x.shape
        shape = (batch, length, self.heads, self.key_size)
        q, k = (self._conv(name, x).reshape(shape) for name in ('q', 'k'))
        v = self._conv('v', x).reshape(batch, length, self.heads, self.value_size)
        decay = F.softplus(self.f_proj(x).float() + self.dt_bias.float()).reshape(shape)
        decay = -self.A_log.float().exp()[None, None, :, None] * decay
        erase = self.b_proj(x).sigmoid().reshape(shape) * (2.0 if self.negative else 1.0)
        write = self.w_proj(x).sigmoid().reshape_as(v)
        out, state = recurrent_gdn2(q, k, v, decay, erase, write, self.state if self.cached else None)
        if self.cached:
            self.state = state
        out = self.o_norm(out, self.g_proj(x).reshape_as(v)).flatten(-2)
        return self.o_proj(out)


class GDN2Mixer(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.layer = GDN2Layer(o)

    def forward(self, x):
        return self.layer(x)


class Attention(nn.Module):
    def __init__(self, o):
        super().__init__()
        d = o['d_model']
        self.heads, self.kv_heads, self.head_dim = o['n_heads'], o['n_kv_heads'] or o['n_heads'], o['head_dim']
        self.theta = o['rope_theta']
        self.rope_full_precision = o['rope_full_precision']
        self.window = o['swa_window'] if o['swa_window'] and o['swa_window'] > 0 else None
        for name, width in [('q', self.heads*self.head_dim), ('k', self.kv_heads*self.head_dim),
                            ('v', self.kv_heads*self.head_dim)]:
            setattr(self, 'w_'+name, nn.Linear(d, width, bias=o['attention_bias']))
        self.w_out = nn.Linear(d, d, bias=o['attention_bias'])
        self.q_norm = RMSNorm(d, o['layer_norm_eps'], o['layer_norm_bias']) if o['attn_qk_norm'] else None
        self.k_norm = RMSNorm(self.kv_heads*self.head_dim, o['layer_norm_eps'], o['layer_norm_bias']) if o['attn_qk_norm'] else None
        self.cached = False
        self.clear()

    def clear(self):
        self.keys = self.values = None
        self.position = 0

    def _rope(self, x, start):
        dtype = x.dtype
        y = x.float() if self.rope_full_precision else x
        inv = 1.0 / self.theta ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim)
        positions = torch.arange(start, start+x.shape[1], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), dim=-1)[None, :, None, :]
        first, second = y.chunk(2, dim=-1)
        return (y * angles.cos().to(y.dtype) + torch.cat((-second, first), dim=-1) * angles.sin().to(y.dtype)).to(dtype)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        q = self.q_norm(q) if self.q_norm is not None else q
        k = self.k_norm(k) if self.k_norm is not None else k
        start = self.position if self.cached else 0
        q = self._rope(q.reshape(batch, length, self.heads, self.head_dim), start).transpose(1, 2)
        k = self._rope(k.reshape(batch, length, self.kv_heads, self.head_dim), start).transpose(1, 2)
        v = v.reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.cached and self.keys is not None:
            k, v = torch.cat((self.keys, k), dim=2), torch.cat((self.values, v), dim=2)
        key_start = start+length-k.shape[2]
        if self.cached:
            # OLMo passes (W-1, 0) to FlashAttention: W keys including self.
            keep = self.window-1 if self.window is not None else k.shape[2]
            self.keys = k[:, :, -keep:].clone() if keep else k[:, :, :0]
            self.values = v[:, :, -keep:].clone() if keep else v[:, :, :0]
            self.position += length
        groups = self.heads // self.kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        # Bound the temporary attention scores for long prompts on either device.
        pieces = []
        key_pos = torch.arange(key_start, start+length, device=x.device)
        for offset in range(0, length, 128):
            query = q[:, :, offset:offset+128]
            query_pos = torch.arange(start+offset, start+offset+query.shape[2], device=x.device)
            distance = query_pos[:, None]-key_pos[None, :]
            allowed = distance >= 0
            if self.window is not None:
                allowed &= distance < self.window
            pieces.append(F.scaled_dot_product_attention(query, k, v, attn_mask=allowed, dropout_p=0.0))
        return self.w_out(torch.cat(pieces, dim=2).transpose(1, 2).reshape(batch, length, -1))


class FeedForward(nn.Module):
    def __init__(self, o):
        super().__init__()
        d, f = o['d_model'], o['ffn_hidden']
        self.w1 = nn.Linear(d, f, bias=o['ffn_bias'])
        self.w2 = nn.Linear(f, d, bias=o['ffn_bias'])
        self.w3 = nn.Linear(d, f, bias=o['ffn_bias'])

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, o, index):
        super().__init__()
        self.attention = Attention(o) if index % 2 else GDN2Mixer(o)
        self.attention_norm = RMSNorm(o['d_model'], o['layer_norm_eps'], o['layer_norm_bias'])
        self.feed_forward = FeedForward(o)
        self.feed_forward_norm = RMSNorm(o['d_model'], o['layer_norm_eps'], o['layer_norm_bias'])

    def forward(self, x):
        # OLMo reordered_norm: normalize branch OUTPUT, not the branch input.
        x = x + self.attention_norm(self.attention(x))
        return x + self.feed_forward_norm(self.feed_forward(x))


class TorchBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        unknown = set(config.native_options)-set(DEFAULT_OPTIONS)
        if unknown:
            raise ValueError(f'Unsupported portable architecture options: {sorted(unknown)}')
        o = dict(DEFAULT_OPTIONS, **config.native_options)
        if (o['d_model'] != o['n_heads']*o['head_dim'] or o['n_layers'] % 2
                or o['head_dim'] % 2 or o['gdn_conv_size'] < 1
                or o['n_heads'] % (o['n_kv_heads'] or o['n_heads'])
                or int(o['head_dim']*o['gdn_expand_v']) != o['head_dim']*o['gdn_expand_v']):
            raise ValueError('Invalid portable GDN2/GQA dimensions')
        if o['swa_full_attention_first_layer'] or o['swa_full_attention_last_layer']:
            raise ValueError('Portable backend supports the published uniform sliding-window configuration')
        if o['enable_swa'] and o['swa_window'] is None:
            raise ValueError('Portable backend requires an explicit swa_window')
        self.embeddings = nn.Embedding(config.vocab_size, o['d_model'])
        self.blocks = nn.ModuleDict({str(i): Block(o, i) for i in range(o['n_layers'])})
        self.lm_head = nn.Module()
        self.lm_head.norm = RMSNorm(o['d_model'], o['layer_norm_eps'], o['layer_norm_bias'])
        self.lm_head.w_out = nn.Linear(o['d_model'], config.vocab_size, bias=o['lm_head_bias'])
        if o['tie_word_embeddings']:
            self.lm_head.w_out.weight = self.embeddings.weight
        self._cache_limit = self._cache_batch = None
        self._position = 0

    def enable_decode(self, max_seq_len, batch_size=1):
        if self.training or max_seq_len <= 0 or batch_size <= 0:
            raise ValueError('Cached inference requires eval() and positive cache dimensions')
        self.disable_decode()
        self._cache_limit, self._cache_batch = max_seq_len, batch_size
        for module in self.modules():
            if isinstance(module, (GDN2Layer, Attention)):
                module.cached = True

    def disable_decode(self):
        self._cache_limit = self._cache_batch = None
        self._position = 0
        for module in self.modules():
            if isinstance(module, (GDN2Layer, Attention)):
                module.cached = False
                module.clear()

    def forward(self, ids, return_logits=True):
        if not return_logits:
            raise ValueError('Portable backend provides inference logits only')
        if self._cache_limit is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError('Cached decoding requires eval() and no_grad()/inference_mode()')
            if ids.shape[0] != self._cache_batch or self._position+ids.shape[1] > self._cache_limit:
                raise ValueError('Decode batch or context exceeds the allocated cache')
            self._position += ids.shape[1]
        x = self.embeddings(ids)
        for block in self.blocks.values():
            x = block(x)
        # generate only consumes the last logit; avoid a prompt-by-vocab allocation.
        if self._cache_limit is not None:
            x = x[:, -1:]
        return self.lm_head.w_out(self.lm_head.norm(x))
