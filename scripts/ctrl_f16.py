"""Copy a GGUF, storing the F32 loop-controller matrices and expert routers as F16 (everything else byte-identical)."""
import os, re, sys
import numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", "build/source/gguf-py"))
import gguf
src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = gguf.GGUFWriter(dst, arch)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
pat = re.compile(r"blk\.\d+\.(loop_\w+_ctrl_\w+|ffn_gate_inp)\.weight$")
conv = {}
for t in r.tensors:
    if t.tensor_type == gguf.GGMLQuantizationType.F32 and pat.match(t.name):
        conv[t.name] = np.asarray(t.data).astype(np.float16)
        w.add_tensor_info(t.name, conv[t.name].shape, np.float16, conv[t.name].nbytes, gguf.GGMLQuantizationType.F16)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(conv.get(t.name, t.data), tensor_endianess=r.endianess)
w.close()
print("converted", len(conv), "tensors")
