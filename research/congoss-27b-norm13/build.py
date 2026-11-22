"""One deterministic final-RMSNorm ablation on the 30% leader."""
import hashlib,json,struct
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
import range_io as io
ROOT=Path(__file__).resolve().parent
PARENT='5065b8a8-e87d-4f06-b52b-183fcb3bab8c'
TENSOR='model.language_model.norm.weight'
GAIN=1.3

def read(meta):
    info=io.file_info(meta,'model.safetensors')
    header,offset,hh=io.header(info)
    spec=header[TENSOR]
    assert spec['dtype']=='BF16' and spec['shape']==[5120]
    a,b=spec['data_offsets'];assert b-a==10240
    raw=io.fetch(info['url'],b-a,offset+a,offset+b-1,info['size_bytes'])
    values=torch.tensor([struct.unpack('<f',struct.pack('<I',x<<16))[0] for x in struct.unpack('<5120H',raw)],dtype=torch.float32)
    return values,hashlib.sha256(raw).hexdigest(),hh

def main():
    parent=io.metadata(PARENT);original=io.metadata(io.ORIGINAL)
    p,ph,hh=read(parent);o,oh,_=read(original)
    expected=((o+1)*1.4-1).to(torch.bfloat16)
    assert torch.equal(expected,p.to(torch.bfloat16)), 'Parent is not the expected original-relative 1.4 norm'
    target=((o+1)*GAIN-1).to(torch.bfloat16)
    mask=target!=p.to(torch.bfloat16)
    indices=torch.arange(5120,dtype=torch.int64)[mask]
    out=ROOT/'submission';out.mkdir(exist_ok=True)
    save_file({'indices':indices,'values':target[mask]},str(out/'weights.safetensors'))
    raw=(out/'weights.safetensors').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    manifest={'schema_version':'delta.submission.v1','reconstruction_version':'delta-reconstruct-v1','model_config_id':io.MODEL,'parent_version_id':PARENT,'parent_content_hash':parent['content_hash'],'format':'sparse-diff','operation':'replace','artifacts':[{'name':'weights.safetensors','sha256':digest,'size_bytes':len(raw)}],'tensors':{TENSOR:{'indices':'indices','values':'values'}}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    saved=load_file(str(out/'weights.safetensors'))
    reconstructed=p.to(torch.bfloat16).clone();reconstructed[saved['indices']]=saved['values']
    assert torch.equal(reconstructed,target) and torch.isfinite(reconstructed).all()
    assert indices.numel()>0 and indices.unique().numel()==indices.numel()
    report={'model':io.MODEL,'parent':PARENT,'parent_content_hash':parent['content_hash'],'original':io.ORIGINAL,'evaluation':'qwen38-27b-nvfp4-agentic300-8k-v1','tensor':TENSOR,'original_relative_parent_gain':1.4,'original_relative_candidate_gain':GAIN,'seed':None,'parent_14_verified_exactly':True,'parent_row_sha256':ph,'original_row_sha256':oh,'parent_header_sha256':hh,'changed_entries':int(mask.sum()),'artifact_sha256':digest,'artifact_bytes':len(raw),'network_bytes':io.network_bytes,'reconstruction_verified':True,'full_parent_hash_verified_locally':False,'inference_run':False,'quality_gain':None}
    (ROOT/'build-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
