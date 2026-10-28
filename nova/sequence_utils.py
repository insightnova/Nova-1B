"""Pack OLMo's document offsets for FLA; no global forward patch."""

import torch


def flatten_documents(x: torch.Tensor, cu_doc_lens: torch.Tensor | None) -> torch.Tensor:
    """Flatten before BOTH short convolutions and the recurrent kernel.

    OLMo's offsets already refer to the row-major flattened batch. Returning a
    view preserves autograd and lets the caller restore the original shape.
    """
    if x.ndim != 3:
        raise ValueError("Expected x with shape [batch, sequence, hidden]")
    if cu_doc_lens is None:
        return x
    if cu_doc_lens.ndim != 1 or cu_doc_lens.numel() < 2:
        raise ValueError("cu_doc_lens must be a one-dimensional offsets tensor")
    if cu_doc_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_doc_lens must have integer dtype int32 or int64")
    if cu_doc_lens.device != x.device:
        raise ValueError("cu_doc_lens and x must be on the same device")
    batch, length, hidden = x.shape
    # Avoid per-layer host synchronization and graph breaks under torch.compile.
    if not torch.compiler.is_compiling():
        if int(cu_doc_lens[0]) != 0 or int(cu_doc_lens[-1]) != batch * length:
            raise ValueError("cu_doc_lens must start at 0 and end at batch * sequence")
        if not bool((cu_doc_lens[1:] > cu_doc_lens[:-1]).all()):
            raise ValueError("cu_doc_lens must increase strictly (no empty documents)")
    return x.reshape(1, batch * length, hidden)
