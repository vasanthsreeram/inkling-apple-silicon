"""Estimate logical weight bytes per generated token from a GGUF header."""
import json
import os
import re
import struct
import sys

def read_header(f, tensor_types=None, tensor_fields=None):
    def num(fmt):
        return struct.unpack("<"+fmt, f.read(struct.calcsize("<"+fmt)))[0]
    def string():
        return f.read(num("Q")).decode()
    fmt = dict(enumerate(["B","b","H","h","I","i","f","?"]))
    fmt.update({10:"Q",11:"q",12:"d"})
    def value(t):
        if t==8: return string()
        if t==9:
            ty,count=num("I"),num("Q")
            if ty in fmt: f.seek(count*struct.calcsize("<"+fmt[ty]),1)
            else:
                for _ in range(count): value(ty)
            return
        return num(fmt[t])
    assert f.read(4)==b"GGUF" and num("I")==3
    count,keys=num("Q"),num("Q")
    meta={}
    for _ in range(keys):
        k=string()
        meta[k]=value(num("I"))
    entries=[]
    for _ in range(count):
        name=string()
        shape=[num("Q") for _ in range(num("I"))]
        if tensor_fields is not None:
            tensor_fields[name] = f.tell()
        typ,offset=num("I"),num("Q")
        if tensor_types is not None:
            tensor_types[name] = typ
        entries.append((offset,name,shape))
    align=meta.get("general.alignment",32)
    base=(f.tell()+align-1)//align*align
    return sorted(entries),base,meta

def active_share(name, shape, meta):
    arch = meta["general.architecture"]
    layer = re.match(r"blk\.(\d+)\.", name)
    core = layer and meta[arch+".loop.start"] <= int(layer[1]) < meta[arch+".loop.end"]
    visits = meta[arch+".loop.count"] if core else 1
    if core and ".loop_" in name:
        visits -= 1  # The initial visit runs without cross-pass messages or controls.
    if name == "token_embd.weight":
        return visits / shape[-1]
    if "_exps." in name:
        return visits * meta[arch+".expert_used_count"] / shape[-1]
    return visits


def estimate(path):
    with open(path,"rb") as f:
        entries,base,meta=read_header(f)
    counts={}
    total=0
    for i,(off,name,shape) in enumerate(entries):
        end = entries[i+1][0] if i+1<len(entries) else os.stat(path).st_size-base
        weight=end-off
        share=active_share(name,shape,meta)
        total+=weight*share
        group = "experts" if "_exps." in name else "other"
        counts[group]=counts.get(group,0)+weight*share
    print(json.dumps(dict(file=os.path.basename(path),weight_bytes_per_token=total,breakdown=counts)))
if __name__=="__main__": estimate(sys.argv[1])
