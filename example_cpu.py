"""Load Nova-1B on CPU and translate an English sentence into Korean."""
import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from example import translate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='exports/Nova-1B')
    parser.add_argument('--text', default='My happy life')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1),
                        help='PyTorch CPU thread count (default: up to 4)')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--revision', help='Reviewed Hugging Face commit')
    args = parser.parse_args()
    if not args.text.strip() or args.max_new_tokens <= 0 or args.threads <= 0:
        parser.error('Text must be nonempty; token budget and thread count must be positive')
    torch.set_num_threads(args.threads)
    if not args.trust_remote_code:
        import nova
        nova.register()
    print(f'Loading Nova-1B on CPU (float32, {args.threads} threads)...', flush=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=args.trust_remote_code,
        inference_backend='cpu', compute_dtype='float32', dtype=torch.float32,
    ).to('cpu').eval()
    print(f'Model loaded in {time.perf_counter()-started:.2f}s; generating...', flush=True)
    started = time.perf_counter()
    answer = translate(model, tokenizer, args.text, device='cpu', max_new_tokens=args.max_new_tokens)
    print(f'English: {args.text}')
    print(f'Korean: {answer}')
    print(f'Generation time: {time.perf_counter()-started:.2f}s')


if __name__ == '__main__':
    main()
