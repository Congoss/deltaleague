"""Exploratory norm-preserving input direction edit; no inference claims."""
import hashlib,json,math,struct
from pathlib import Path
import range_io as io
import torch
from safetensors.torch import save_file,load_file
ROOT=Path(__file__).resolve().parent
ALPHA=0.03
TARGET=248069
GUIDE=248058

def main():
    meta=io.metadata(io.PARENT)
    info=io.file_info(meta,'model.safetensors')
    specs,offset,header_hash=io.header(info)
    tokinfo=io.file_info(meta,'tokenizer.json')
    raw=io.fetch(tokinfo['url'],32*1024**2)
    assert hashlib.sha256(raw).hexdigest()==tokinfo['sha256']
    vocab={x['content']:x['id'] for x in json.loads(raw)['added_tokens']}
    assert vocab['</think>']==TARGET and vocab['<tool_call>']==GUIDE
    rows={i:io.embedding_row(info,specs,offset,i) for i in [TARGET,GUIDE]}
    def decode(raw):
        return torch.tensor([struct.unpack('<f',struct.pack('<I',x<<16))[0] for x in struct.unpack('<5120H',raw)],dtype=torch.float64)
    a,b=decode(rows[TARGET]),decode(rows[GUIDE])
    unit=a/a.norm()
    direction=b/b.norm()
    orth=direction-torch.dot(direction,unit)*unit
    assert orth.norm()>1e-8
    candidate=(unit+ALPHA*orth/orth.norm())/math.sqrt(1+ALPHA**2)*a.norm()
    values=candidate.to(torch.bfloat16)
    indices=torch.arange(5120,dtype=torch.int64)+TARGET*5120
    changed=values!=a.to(torch.bfloat16)
    out=ROOT/'submission';out.mkdir(exist_ok=True)
    save_file({'indices':indices[changed],'values':values[changed]},str(out/'weights.safetensors'))
    payload=(out/'weights.safetensors').read_bytes()
    digest=hashlib.sha256(payload).hexdigest()
    manifest={'schema_version':'delta.submission.v1','reconstruction_version':'delta-reconstruct-v1','model_config_id':io.MODEL,'parent_version_id':io.PARENT,'parent_content_hash':meta['content_hash'],'format':'sparse-diff','operation':'replace','artifacts':[{'name':'weights.safetensors','sha256':digest,'size_bytes':len(payload)}],'tensors':{io.EMBED:{'indices':'indices','values':'values'}}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    loaded=load_file(str(out/'weights.safetensors'))
    rebuilt=a.to(torch.bfloat16).clone()
    rebuilt[loaded['indices']-TARGET*5120]=loaded['values']
    assert torch.equal(rebuilt,values) and torch.isfinite(rebuilt).all()
    assert loaded['indices'].unique().numel()==int(changed.sum())
    norm_error=abs((rebuilt.double().norm()/a.norm()).item()-1)
    assert norm_error<0.001
    report={'parent':io.PARENT,'parent_content_hash':meta['content_hash'],'evaluation':'qwen38-27b-nvfp4-agentic300-8k-v1','target':TARGET,'guide':GUIDE,'alpha':ALPHA,'seed':None,'changed_entries':int(changed.sum()),'artifact_sha256':digest,'artifact_bytes':len(payload),'relative_l2_change':float((rebuilt.double()-a).norm()/a.norm()),'relative_norm_error':norm_error,'source_row_sha256':{str(k):hashlib.sha256(v).hexdigest() for k,v in rows.items()},'header_sha256':header_hash,'tokenizer_sha256':tokinfo['sha256'],'network_bytes':io.network_bytes,'full_parent_hash_verified':False,'sparse_reconstruction_verified':True,'inference_run':False,'quality_gain':None}
    (ROOT/'build-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
