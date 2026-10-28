"""Translate a short English sentence with Nova-1B using a plain-text prompt."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def translation_prompt(text):
    return (
        "Translate the following English sentence into Korean. "
        "Output only the translation.\n"
        f"English: {text}\nKorean:"
    )


def translate(model, tokenizer, text, device="cuda:0", max_new_tokens=64):
    # Explicit BOS, without applying a chat template or duplicating special tokens.
    ids = [tokenizer.bos_token_id] + tokenizer.encode(
        translation_prompt(text), add_special_tokens=False
    )
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(
        output[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="exports/Nova-1B",
                        help="Complete exported model directory or Hugging Face model ID")
    parser.add_argument("--text", default="My happy life")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("torch", "flash"), default="torch",
                        help="torch: no flash-attn installation; flash: accelerated Linux CUDA backend")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", help="Reviewed Hugging Face commit")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu or cuda:N")
    if args.backend == "flash" and device.type != "cuda":
        parser.error("--backend flash requires --device cuda:N")
    if not args.trust_remote_code:
        import nova
        nova.register()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        inference_backend=args.backend,
        compute_dtype="float32" if device.type == "cpu" else "bfloat16",
    ).to(args.device).eval()
    answer = translate(model, tokenizer, args.text, args.device, args.max_new_tokens)
    print(f"English: {args.text}")
    print(f"Korean: {answer}")


if __name__ == "__main__":
    main()
