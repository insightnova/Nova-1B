"""Export an existing inference checkpoint; never import training packages or CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nova.configuration_nova import NovaConfig

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def checkpoint_state(source):
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (source/index_name).exists():
            index = read_json(source/index_name)
            names = sorted(set(index["weight_map"].values()))
            break
    else:
        names = [name for name in ("model.safetensors", "pytorch_model.bin") if (source/name).exists()]
        if len(names) != 1:
            raise ValueError("Expected exactly one safetensors/bin checkpoint, or a shard index")
    state = {}
    for name in names:
        path = (source/name).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Weight index escapes the checkpoint directory")
        part = (load_file(str(path), device="cpu") if path.suffix == ".safetensors"
                else torch.load(path, map_location="cpu", weights_only=True, mmap=True))
        if not isinstance(part, dict) or any(not isinstance(v, torch.Tensor) for v in part.values()):
            raise ValueError("Expected an inference state_dict, not an optimizer/training checkpoint")
        if state.keys() & part.keys():
            raise ValueError("Repeated tensor keys across weight shards")
        state.update(part)
    if not state or any(not k.startswith("native.") for k in state):
        raise ValueError("Use a Nova inference checkpoint with native.* weights; raw pretraining DCP is not supported")
    if "native.embeddings.weight" not in state:
        raise ValueError("Missing token embeddings")
    return state


def write_weights(state, output, max_bytes):
    groups, group, size = [], {}, 0
    for name, value in sorted(state.items()):
        nbytes = value.numel()*value.element_size()
        if group and size+nbytes > max_bytes:
            groups.append(group)
            group, size = {}, 0
        group[name] = value
        size += nbytes
    if group:
        groups.append(group)
    weight_map = {}
    total = 0
    for i, group in enumerate(groups, 1):
        name = "model.safetensors" if len(groups)==1 else f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        # Clone shared tensors only within this shard. No precision conversion.
        tensors = {k:v.detach().cpu().contiguous().clone() for k,v in group.items()}
        save_file(tensors, str(output/name), metadata={"format":"pt"})
        total += sum(v.numel()*v.element_size() for v in tensors.values())
        weight_map.update({k:name for k in tensors})
        del tensors
    if len(groups)>1:
        write_json(output/"model.safetensors.index.json", {"metadata":{"total_size":total},"weight_map":weight_map})
    return weight_map


def export(source, output, max_shard_bytes=1_000_000_000, metadata_only=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be empty; existing exports are never overwritten")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    raw = read_json(source/"config.json")
    if raw.get("model_type") not in ("gdn2_hybrid_posttraining", "nova_gdn2"):
        raise ValueError("Unsupported model architecture")
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if len(tok) != raw.get("tokenizer_vocab_size", raw["vocab_size"]):
        raise ValueError("Tokenizer and model vocabulary sizes differ")
    if not tok.chat_template:
        raise ValueError("Missing chat template")
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if raw.get(k) != getattr(tok, k):
            raise ValueError(f"Tokenizer/model mismatch: {k}")
    raw = {k:v for k,v in raw.items() if k not in
           ("model_type", "architectures", "auto_map", "native_repo", "_name_or_path", "_commit_hash")}
    config = NovaConfig(**raw)
    config.architectures = ["NovaForCausalLM"]
    config.auto_map = {"AutoConfig":"configuration_nova.NovaConfig",
                       "AutoModelForCausalLM":"modeling_nova.NovaForCausalLM"}
    state = None if metadata_only else checkpoint_state(source)
    if state is not None:
        shape = tuple(state["native.embeddings.weight"].shape)
        if shape != (config.vocab_size, config.hidden_size):
            raise ValueError("Embedding tensor does not match config")
    output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output)
    for f in (ROOT/"nova").glob("*.py"):
        if f.name != "__init__.py":
            shutil.copy2(f, output/f.name)
    for name in TOKENIZER_FILES:
        if (source/name).is_file():
            shutil.copy2(source/name, output/name)
    tok_config = read_json(output/"tokenizer_config.json")
    for key in ("name_or_path", "_name_or_path", "tokenizer_file", "special_tokens_map_file"):
        tok_config.pop(key, None)
    tok_config["padding_side"] = "left"
    tok_config["model_input_names"] = ["input_ids", "attention_mask"]
    write_json(output/"tokenizer_config.json", tok_config)
    if (source/"generation_config.json").exists():
        gen = read_json(source/"generation_config.json")
        gen.pop("_from_model_config", None)
        gen.pop("_name_or_path", None)
        write_json(output/"generation_config.json", gen)
    shutil.copy2(ROOT/"THIRD_PARTY_NOTICES.txt", output/"THIRD_PARTY_NOTICES.txt")
    shutil.copy2(ROOT/"LICENSE", output/"LICENSE")
    shutil.copy2(ROOT/"SOURCE_PROVENANCE.json", output/"SOURCE_PROVENANCE.json")
    shutil.copy2(ROOT/"MODEL_CARD.md", output/"README.md")
    weight_map = {} if metadata_only else write_weights(state, output, max_shard_bytes)
    if metadata_only:
        (output/"METADATA_ONLY.txt").write_text("No weights. This directory cannot perform model inference.\n",encoding="utf-8")
    manifest = {"metadata_only":metadata_only,"tensor_count":len(weight_map),
                "weights_precision":"preserved without casting","files":{}}
    for p in sorted(output.iterdir()):
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            manifest["files"][p.name]={"bytes":p.stat().st_size,"sha256":h.hexdigest()}
    write_json(output/"export_manifest.json",manifest)
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-shard-gb',type=float,default=1.0)
    ap.add_argument('--metadata-only',action='store_true')
    args=ap.parse_args()
    m=export(args.checkpoint,args.output,int(args.max_shard_gb*1e9),args.metadata_only)
    print(json.dumps({'output':str(args.output),'metadata_only':m['metadata_only'],'tensors':m['tensor_count']},indent=2))


if __name__=='__main__':main()
