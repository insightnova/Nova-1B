"""CPU contract tests use a tiny backend; they do not validate GDN2 CUDA kernels."""
import json
from pathlib import Path
import shutil
import sys
import types

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, GenerationConfig
from transformers.dynamic_module_utils import get_relative_import_files, get_class_from_dynamic_module
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import nova
from nova.configuration_nova import NovaConfig
from nova import modeling_nova
from nova.modeling_nova import NovaForCausalLM
from export_hf import export


class TinyNative(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings=nn.Embedding(config.vocab_size,config.hidden_size)
        self.lm_head=nn.Module()
        self.lm_head.w_out=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.lm_head.w_out.weight=self.embeddings.weight
        self.mixer=nn.Module()
        self.mixer.A_log=nn.Parameter(torch.tensor([0.123456789]))
        self.mixer.dt_bias=nn.Parameter(torch.tensor([-0.987654321]))

    def forward(self, ids, return_logits=True):
        return self.lm_head.w_out(self.embeddings(ids))


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(modeling_nova,'build_native',TinyNative)
    nova.register()
    config=NovaConfig(vocab_size=8,hidden_size=4,num_hidden_layers=2,
        max_position_embeddings=8,fast_decode=False,compute_dtype='float32',native_options={})
    return NovaForCausalLM(config).eval()


def test_standard_save_load_preserves_logits_and_tied_weights(model,tmp_path):
    x=torch.tensor([[1,4,5]])
    original=model(x).logits
    model.save_pretrained(tmp_path)
    loaded=AutoModelForCausalLM.from_pretrained(tmp_path,trust_remote_code=False).eval()
    torch.testing.assert_close(loaded(x).logits,original,rtol=0,atol=0)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight


def test_fp32_masters_survive_bfloat16_loading_and_cast(model,tmp_path):
    original=model.native.mixer.A_log.detach().clone()
    model.save_pretrained(tmp_path)
    loaded=AutoModelForCausalLM.from_pretrained(tmp_path,dtype=torch.bfloat16,trust_remote_code=False)
    assert loaded.native.embeddings.weight.dtype==torch.bfloat16
    assert loaded.native.mixer.A_log.dtype==torch.float32
    assert torch.equal(loaded.native.mixer.A_log,original)
    loaded.to(torch.bfloat16)
    assert torch.equal(loaded.native.mixer.A_log,original)


def deterministic(model):
    def logits(ids):
        result=torch.full((*ids.shape,8),-100.0)
        target=(ids+1)%8
        return result.scatter(-1,target[...,None],100.0)
    model._native_logits=logits
    return model


def test_batch_matches_serial_and_stops_at_eos(model):
    model=deterministic(model)
    ids=torch.tensor([[0,1],[4,5]])
    mask=torch.tensor([[0,1],[1,1]])
    batch=model.generate(ids,attention_mask=mask,max_new_tokens=4,eos_token_id=3)
    for i in range(2):
        one=model.generate(ids[i:i+1],attention_mask=mask[i:i+1],max_new_tokens=4,eos_token_id=3)
        assert torch.equal(batch[i,:one.shape[1]],one[0])
    assert batch[0,2:].tolist()==[2,3,0,0]
    assert not model.training


def test_context_budget_and_repeat_calls(model):
    model=deterministic(model)
    ids=torch.tensor([[1,2,4,4,4,4,4]])
    a=model.generate(ids,max_new_tokens=10,eos_token_id=None)
    assert a.shape[1]==8
    assert torch.equal(a,model.generate(ids,max_new_tokens=10,eos_token_id=None))
    model.config.generation_context_policy='error'
    with pytest.raises(ValueError,match='context'):model.generate(ids,max_new_tokens=10)


@pytest.mark.parametrize('kwargs',[{'num_beams':2},{'repetition_penalty':1.2},{'streamer':object()},
                                  {'return_dict_in_generate':True},{'no_repeat_ngram_size':3}])
def test_unsupported_controls_fail_loudly(model,kwargs):
    with pytest.raises((ValueError,TypeError)):
        model.generate(torch.tensor([[1,4]]),max_new_tokens=2,**kwargs)


def test_invalid_padding_rejected(model):
    with pytest.raises(ValueError,match='contiguous'):
        model.generate(torch.tensor([[1,0,4]]),attention_mask=torch.tensor([[1,0,1]]),max_new_tokens=1)


def test_unsupported_model_modes(model,tmp_path):
    model.save_pretrained(tmp_path)
    with pytest.raises(ValueError,match='device_map'):
        NovaForCausalLM.from_pretrained(tmp_path,device_map='auto')
    with pytest.raises(ValueError,match='inference only'):
        model(torch.tensor([[1,4]]),labels=torch.tensor([[1,4]]))


def make_checkpoint(model,source):
    source.mkdir()
    config=model.config.to_dict()
    config.update(model_type='gdn2_hybrid_posttraining',native_repo='/old/private/server/path',auto_map={})
    (source/'config.json').write_text(json.dumps(config),encoding='utf-8')
    torch.save(model.state_dict(),source/'pytorch_model.bin')
    backend=Tokenizer(WordLevel({'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'hi':4,'nova':5,'yes':6,'no':7},unk_token='<unk>'))
    backend.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',bos_token='<bos>',eos_token='<eos>',unk_token='<unk>')
    tok.chat_template="{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}{{ eos_token }}"
    tok.save_pretrained(source)


def test_export_shards_is_lossless_and_relocatable(model,tmp_path):
    src,out=tmp_path/'source',tmp_path/'export'
    make_checkpoint(model,src)
    manifest=export(src,out,max_shard_bytes=64)
    assert not manifest['metadata_only']
    assert (out/'LICENSE').read_bytes()==(Path(__file__).parents[1]/'LICENSE').read_bytes()
    assert 'LICENSE' in manifest['files']
    assert (out/'model.safetensors.index.json').exists()
    assert '/old/private' not in (out/'config.json').read_text()
    assert 'native_repo' not in json.loads((out/'config.json').read_text())
    moved=tmp_path/'moved'
    shutil.move(str(out),str(moved))
    loaded=AutoModelForCausalLM.from_pretrained(moved,trust_remote_code=False).eval()
    x=torch.tensor([[1,4,5]])
    torch.testing.assert_close(model(x).logits,loaded(x).logits,rtol=0,atol=0)
    tok=AutoTokenizer.from_pretrained(moved,trust_remote_code=False)
    assert tok.encode('hi',add_special_tokens=False)==[4]
    assert 'token_type_ids' not in tok('hi')
    dynamic=AutoConfig.from_pretrained(moved,trust_remote_code=True,local_files_only=True)
    assert dynamic.model_type=='nova_gdn2'
    for f in get_relative_import_files(moved/'modeling_nova.py'):
        assert Path(f).is_file(),f


def test_export_does_not_overwrite_and_metadata_only_is_explicit(model,tmp_path):
    src,out=tmp_path/'source',tmp_path/'export'
    make_checkpoint(model,src)
    export(src,out,metadata_only=True)
    assert (out/'METADATA_ONLY.txt').exists()
    assert not list(out.glob('*.safetensors'))
    with pytest.raises(FileExistsError):export(src,out)
    with pytest.raises(ValueError):export(src,src/'nested')


def test_token_type_ids_from_generic_fast_tokenizer(model):
    ids=torch.tensor([[1,4]])
    result=model.generate(ids,max_new_tokens=1,token_type_ids=torch.zeros_like(ids))
    assert result.shape==(1,3)
    with pytest.raises(ValueError,match='token_type_ids'):
        model.generate(ids,max_new_tokens=1,token_type_ids=torch.ones_like(ids))


def test_remote_code_autoclass_with_tiny_backend(model,tmp_path,monkeypatch):
    src,out=tmp_path/'source',tmp_path/'remote_bundle'
    make_checkpoint(model,src)
    export(src,out,max_shard_bytes=64)
    cls=get_class_from_dynamic_module('modeling_nova.NovaForCausalLM',out,local_files_only=True)
    # Only substitute hardware-dependent construction. All AutoClass resolution,
    # exported code, standard weight loading and generation stay in the release.
    module=sys.modules[cls.__module__]
    monkeypatch.setattr(module,'build_native',TinyNative)
    loaded=AutoModelForCausalLM.from_pretrained(out,trust_remote_code=True,local_files_only=True).eval()
    assert type(loaded).__module__.startswith('transformers_modules.')
    x=torch.tensor([[1,4,5]])
    torch.testing.assert_close(model(x).logits,loaded(x).logits,rtol=0,atol=0)
    assert torch.equal(model.generate(x,max_new_tokens=2),loaded.generate(x,max_new_tokens=2))


def test_custom_generation_config_does_not_silently_ignore_processors(model):
    config=GenerationConfig(sequence_bias={(4,):1.0})
    with pytest.raises(ValueError,match='sequence_bias'):
        model.generate(torch.tensor([[1,4]]),generation_config=config,max_new_tokens=1)


def test_decoder_cleanup_on_exception(model,monkeypatch):
    model.config.inference_backend='flash'
    calls=[]
    decoder=types.ModuleType('nova.fast_decode')
    decoder.enable=lambda *a,**kw:calls.append('enable')
    decoder.disable=lambda *a,**kw:calls.append('disable')
    monkeypatch.setitem(sys.modules,'nova.fast_decode',decoder)
    model.config.fast_decode=True
    def fail(ids):raise RuntimeError('simulated decode failure')
    model._native_logits=fail
    with pytest.raises(RuntimeError,match='simulated'):
        model.generate(torch.tensor([[1,4]]),max_new_tokens=1)
    assert calls==['enable','disable']
    assert not model.training


def test_missing_model_weights_are_rejected(model,tmp_path):
    from safetensors.torch import load_file,save_file
    model.save_pretrained(tmp_path)
    path=tmp_path/'model.safetensors'
    state=load_file(str(path))
    del state['native.mixer.A_log']
    save_file(state,str(path),metadata={'format':'pt'})
    with pytest.raises(ValueError,match='Incomplete'):
        AutoModelForCausalLM.from_pretrained(tmp_path,trust_remote_code=False)


def test_generation_batch_size_controls_grouping(model,monkeypatch):
    calls=[]
    original=model._native_logits
    def record(ids):
        calls.append(ids.shape[0])
        return original(ids)
    monkeypatch.setattr(model,'_native_logits',record)
    ids=torch.tensor([[1,4],[1,5],[1,6]])
    grouped=model.generate(ids,max_new_tokens=1,generation_batch_size=2)
    assert calls==[2,1]
    calls.clear()
    single=model.generate(ids,max_new_tokens=1)
    assert calls==[1,1,1]
    assert torch.equal(grouped,single)
    with pytest.raises(ValueError,match='generation batch size'):
        model.generate(ids,max_new_tokens=1,generation_batch_size=0)
