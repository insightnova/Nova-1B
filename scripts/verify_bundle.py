"""Check exported files on CPU; optionally exercise real CUDA inference."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def verify(path, cuda=False, backend='torch'):
    path=Path(path)
    manifest=json.loads((path/'export_manifest.json').read_text(encoding='utf-8'))
    for name,entry in manifest['files'].items():
        f=(path/name).resolve()
        if not f.is_relative_to(path.resolve()):raise ValueError('Manifest path escapes bundle')
        if f.stat().st_size!=entry['bytes']:raise ValueError(f'Size mismatch: {name}')
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=entry['sha256']:raise ValueError(f'Hash mismatch: {name}')
    config=AutoConfig.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    assert config.model_type=='nova_gdn2'
    assert len(tokenizer)==config.tokenizer_vocab_size
    text='안녕하세요. Hello, Nova! 123'
    assert tokenizer.decode(tokenizer.encode(text,add_special_tokens=False))==text
    prompts=[f'질문: {x}\n답변:' for x in ['당신은 누구인가요?','What is 2 + 2?']]
    result={'files_verified':len(manifest['files']),'tokenizer_roundtrip':True,
            'metadata_only':manifest['metadata_only'],'cuda_inference':'not_run'}
    if cuda:
        if manifest['metadata_only']:raise ValueError('This is a metadata-only preview, not an inference bundle')
        import torch
        from transformers import AutoModelForCausalLM
        model,loading=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,
            local_files_only=True,output_loading_info=True,inference_backend=backend)
        if loading['missing_keys'] or loading['unexpected_keys'] or loading['mismatched_keys']:
            raise ValueError(f'Weight loading mismatch: {loading}')
        model=model.to('cuda').eval()
        tokens=[([tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else [])
                + tokenizer.encode(text,add_special_tokens=False) for text in prompts]
        inputs=tokenizer.pad({'input_ids':tokens},padding=True,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            cached=model.generate(**inputs,max_new_tokens=8,do_sample=False,generation_batch_size=2)
            repeated=model.generate(**inputs,max_new_tokens=8,do_sample=False,generation_batch_size=2)
            assert torch.equal(cached,repeated),'Cache state leaked between requests'
            model.config.fast_decode=False
            uncached=model.generate(**inputs,max_new_tokens=8,do_sample=False)
        result.update(cuda_inference='passed',backend=backend,repeat_request_equal=True,
            cached_vs_uncached_tokens_equal=torch.equal(cached,uncached))
        if not result['cached_vs_uncached_tokens_equal']:
            raise ValueError('Cached/uncached greedy tokens differ; investigate before publication')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--cuda',action='store_true')
    p.add_argument('--backend',choices=('torch','flash'),default='torch')
    args=p.parse_args()
    print(json.dumps(verify(args.model,args.cuda,args.backend),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
