# input length threshold for flash attn
#export FLASH_ATTN_THRESHOLD=400

# config for first token
export ENABLE_CAT_MLP=1
#export ENABLE_MKL_GEMM=1


# config for next tokens
export ENABLE_CAT_NEXT_MLP=1
#export ENABLE_MKL_NEXT_GEMM=1


#export XFT_ONECCL=1
#export ENABLE_TUNED_COMM=0
export ENABLE_KV_TRANS=1
#export ENABLE_SKIP_MASK=1

#export LD_PRELOAD=../3rdparty/mklml/lib/libiomp5.so:${LD_PRELOAD}

LOOP=2
WARMUP=1
DTYPE=bf16
SeqLen=512
OutLen=512
#PrefixLen=0
BSIZE=12

# env for mkl and shm
export ENV_PACK_M=`expr $SeqLen \* $BSIZE`
export ENV_PACK_NEXT_M=`expr $BSIZE`

ncores=96
stde=48

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_name=llama-2-13b
model_path="${SCRIPT_DIR}"/../examples/model_config/${model_name}/

benchmark_cmd="python "${SCRIPT_DIR}"/benchmark.py \
    --token_path "${model_path}" \
    --model_path "${model_path}" \
    --prompt_path "${SCRIPT_DIR}"/prompt.json \
    --model_name "${model_name}" \
    --dtype "${DTYPE}" \
    --batch_size "${BSIZE}" \
    --token_in ${SeqLen}        \
    --token_out ${OutLen} \
    --beam_width 1 \
    --iteration $LOOP \
    --warmup $WARMUP \
    --padding=False"

for ((i=0;i<$ncores;i+=$stde));
do
        ie=`expr $i + $stde - 1`
        id=`expr $i / $stde`
        ih=`expr $id + 2`
        FIRST_TOKEN_WEIGHT_LOCATION=$id NEXT_TOKEN_WEIGHT_LOCATION=$ih OMP_NUM_THREADS=$stde \
        numactl -C $i-${ie} -p $ih $benchmark_cmd &
done
wait
