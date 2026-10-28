"""Generate a reply from a plain-text question with Nova-1B."""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(question, instruction=None):
    question_text = f"질문: {question}\n답변:"
    return f"{instruction}\n\n{question_text}" if instruction else question_text


def encode_prompt(tokenizer, question, instruction=None, device='cpu'):
    text = build_prompt(question, instruction)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        tokens = [tokenizer.bos_token_id] + tokens
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--system',help='Optional plain-text instruction prepended to the question')
    p.add_argument('--revision',help='Pin a reviewed Hub commit')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--backend',choices=('torch','flash'),default='torch')
    p.add_argument('--threads',type=int,default=min(4,os.cpu_count() or 1),help='CPU thread count')
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--sample',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trust-remote-code',action='store_true')
    args=p.parse_args()
    if not args.prompt.strip() or args.max_new_tokens <= 0 or args.threads <= 0:
        p.error('Prompt must be nonempty; token budget and thread count must be positive')
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda'):
        p.error('--device must be cpu or cuda:N')
    if args.backend=='flash' and device.type!='cuda':
        p.error('--backend flash requires --device cuda:N')
    if device.type=='cpu':
        torch.set_num_threads(args.threads)
    if not args.trust_remote_code:
        import nova
        nova.register()
    torch.manual_seed(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    options={'inference_backend':'cpu','compute_dtype':'float32','dtype':torch.float32} if device.type=='cpu' else {'inference_backend':args.backend}
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,
        trust_remote_code=args.trust_remote_code,**options).to(device).eval()
    inputs=encode_prompt(tokenizer,args.prompt,args.system,device)
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=args.sample,
            temperature=0.8,top_p=0.95,top_k=0)
    print(tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip())


if __name__=='__main__':main()
