# input length threshold for flash attn
export FLASH_ATTN_THRESHOLD=400

# config for first token
export ENABLE_CAT_MLP=1
#export ENABLE_MKL_GEMM=1


# config for next tokens
export ENABLE_CAT_NEXT_MLP=1
#export ENABLE_MKL_NEXT_GEMM=1


#export XFT_ONECCL=1
#export ENABLE_TUNED_COMM=0
#export ENABLE_KV_TRANS=1
#export ENABLE_SKIP_MASK=1

#export LD_PRELOAD=../3rdparty/mklml/lib/libiomp5.so:${LD_PRELOAD}

LOOP=2
WARMUP=1
DTYPE=bf16
SeqLen=512
OutLen=256
#PrefixLen=0
BSIZE=24

# env for mkl and shm
export ENV_PACK_M=`expr $SeqLen \* $BSIZE`
export ENV_PACK_NEXT_M=`expr $BSIZE`

bash run_benchmark.sh -m llama-2-13b -d $DTYPE -s $1 -bs $BSIZE -in $SeqLen -out $OutLen -i $LOOP -w $WARMUP
