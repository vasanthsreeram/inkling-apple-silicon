# Source this to enable the optimised Inkling paths:  . ./env.sh
# Every change is opt-in; with none of these set the binary behaves like upstream.
export INKLING_SITU=1 INKLING_CLIP=1 INKLING_GATE_MERGE=1 INKLING_SHORTCONV=1   # fused activation, clip, gate merge, 4-tap conv
export INKLING_REL_MASK=1 INKLING_DEPTH_CACHE=1                                 # relative mask, cached depth norms
export INKLING_DEPTH_MIX=1 INKLING_REL_BIAS=1                                   # prefill glue: fused depth attention, relative bias in the mask
export INKLING_CTRL_FUSE=1 INKLING_CONV_STATE=1 INKLING_ORDER=1                 # decode: fused loop controllers, conv state in place, grouped independent kernels
if [ "$(sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -c -E 'M5')" = 1 ]; then
    export INKLING_MOE64=32 INKLING_MOE64_WS=1                                   # M5 only: 64-token expert tiles on the neural accelerators
fi
