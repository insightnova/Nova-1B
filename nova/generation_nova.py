"""Inference-only generation, independent per-request caches and EOS handling."""
import torch
import torch.nn.functional as F


def validate_inputs(ids, attention_mask, context):
    if ids is None or ids.ndim != 2 or not ids.shape[0] or not ids.shape[1]:
        raise ValueError("Expected a nonempty [batch, sequence] input_ids tensor")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must contain integer token IDs")
    if attention_mask is not None and (attention_mask.shape != ids.shape or attention_mask.device != ids.device):
        raise ValueError("attention_mask shape and device must match input_ids")
    if attention_mask is not None and not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must contain only zero and one")
    mask = torch.ones_like(ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    for row in mask:
        pos = row.nonzero(as_tuple=True)[0]
        if not len(pos) or int(pos[-1]-pos[0]+1) != len(pos):
            raise ValueError("Only contiguous nonempty prompts with left/right padding are supported")
        if len(pos) > context:
            raise ValueError("Prompt exceeds the model context")
    return mask


def sample_token(logits, do_sample, temperature=1.0, top_p=1.0, top_k=0):
    if not do_sample:
        return logits.argmax(dim=-1)
    if temperature <= 0 or not 0 < top_p <= 1 or type(top_k) is not int or top_k < 0:
        raise ValueError("Invalid sampling parameters")
    scores = logits.float() / temperature
    if top_k:
        cutoff = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if top_p < 1:
        ordered, indices = scores.sort(descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scores = scores.masked_fill(torch.zeros_like(remove).scatter(1, indices, remove), -torch.inf)
    return torch.multinomial(scores.softmax(-1), 1).squeeze(-1)


def prompt_batches(ids, mask, limit):
    lengths = mask.sum(-1).tolist()
    start = 0
    while start < len(lengths):
        end = start + 1
        while end < len(lengths) and end-start < limit and lengths[end] == lengths[start]:
            end += 1
        yield torch.stack([ids[i, mask[i]] for i in range(start, end)])
        start = end


@torch.no_grad()
def generate_batched(model, ids, mask, requested, eos_ids, pad, option, limit):
    decoder = None
    if model.config.fast_decode:
        if model.config.inference_backend in ("torch", "cpu"):
            decoder = (lambda native, **kw: native.enable_decode(**kw),
                       lambda native: native.disable_decode())
        else:
            from .fast_decode import enable, disable
            decoder = (enable, disable)
    pad = pad if pad is not None else min(eos_ids, default=0)
    eos = torch.tensor(sorted(eos_ids), device=ids.device, dtype=ids.dtype)
    was_training = model.training
    completions = []
    model.eval()
    try:
        for prompt in prompt_batches(ids, mask, limit):
            remaining = model.config.max_position_embeddings-prompt.shape[1]
            if remaining < 1 or (requested > remaining and model.config.generation_context_policy == "error"):
                raise ValueError("Requested generation exceeds the model context")
            budget = min(requested, remaining)
            count = prompt.shape[0]
            tail = ids.new_full((count, budget), pad)
            alive = torch.ones(count, device=ids.device, dtype=torch.bool)
            lengths = torch.zeros(count, device=ids.device, dtype=torch.long)
            try:
                if decoder:
                    decoder[0](model.native, max_seq_len=prompt.shape[1]+budget, batch_size=count)
                current = prompt
                for step in range(budget):
                    token = sample_token(model._native_logits(current)[:, -1], option("do_sample", False),
                        option("temperature", 1.0), option("top_p", 1.0), option("top_k", 0))
                    token = torch.where(alive, token, pad)
                    tail[:, step] = token
                    lengths += alive
                    if eos.numel():
                        alive &= ~(token[:, None] == eos[None, :]).any(-1)
                    if not alive.any():
                        break
                    current = token[:, None] if decoder else torch.cat([current, token[:, None]], dim=1)
            finally:
                if decoder:
                    decoder[1](model.native)
            completions.extend(tail[row, :n] for row, n in enumerate(lengths.tolist()))
    finally:
        model.train(was_training)
    width = max(map(len, completions))
    tails = torch.stack([F.pad(x, (0, width-len(x)), value=pad) for x in completions])
    return torch.cat([ids, tails], dim=1)
