import hashlib,json,struct
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
import range_io as io
ROOT=Path(__file__).resolve().parent
PARENT='5065b8a8-e87d-4f06-b52b-183fcb3bab8c'
TOKEN=248067
SCALE=0.95

def main():
    m=io.metadata(PARENT)
    def verified_asset(name):
        f=io.file_info(m,name);raw=io.fetch(f['url'],32*1024**2)
        assert hashlib.sha256(raw).hexdigest()==f['sha256']
        return raw,f['sha256']
    tok,th=verified_asset('tokenizer.json')
    template,ch=verified_asset('chat_template.jinja')
    assert {x['content']:x['id'] for x in json.loads(tok)['added_tokens']}['</tool_response>']==TOKEN
    assert '</tool_response>' in template.decode()
    f=io.file_info(m,'model.safetensors');spec,offset,hh=io.header(f)
    row=io.embedding_row(f,spec,offset,TOKEN)
    p=torch.tensor([struct.unpack('<f',struct.pack('<I',x<<16))[0] for x in struct.unpack('<5120H',row)],dtype=torch.float32)
    target=(p*SCALE).to(torch.bfloat16);mask=target!=p.to(torch.bfloat16)
    indices=(torch.arange(5120,dtype=torch.int64)+TOKEN*5120)[mask]
    out=ROOT/'submission';out.mkdir(exist_ok=True)
    save_file({'indices':indices,'values':target[mask]},str(out/'weights.safetensors'))
    raw=(out/'weights.safetensors').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    manifest={'schema_version':'delta.submission.v1','reconstruction_version':'delta-reconstruct-v1','model_config_id':io.MODEL,'parent_version_id':PARENT,'parent_content_hash':m['content_hash'],'format':'sparse-diff','operation':'replace','artifacts':[{'name':'weights.safetensors','sha256':digest,'size_bytes':len(raw)}],'tensors':{io.EMBED:{'indices':'indices','values':'values'}}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    saved=load_file(str(out/'weights.safetensors'));r=p.to(torch.bfloat16).clone()
    assert indices.numel()>0 and indices.unique().numel()==indices.numel()
    r[saved['indices']-TOKEN*5120]=saved['values']
    assert torch.equal(r,target) and torch.isfinite(r).all()
    report={'parent':PARENT,'parent_content_hash':m['content_hash'],'evaluation':'qwen38-27b-nvfp4-agentic300-8k-v1','tensor':io.EMBED,'token':TOKEN,'token_text':'</tool_response>','scale':SCALE,'seed':None,'changed_entries':int(mask.sum()),'artifact_sha256':digest,'artifact_bytes':len(raw),'parent_row_sha256':hashlib.sha256(row).hexdigest(),'parent_header_sha256':hh,'tokenizer_sha256':th,'chat_template_sha256':ch,'marker_present_in_template':True,'network_bytes':io.network_bytes,'reconstruction_verified':True,'full_parent_hash_verified_locally':False,'inference_run':False,'quality_gain':None}
    (ROOT/'build-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
