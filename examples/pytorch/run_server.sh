#export $(python3 -c 'import xfastertransformer as xft; print(xft.get_env())')
TOKEN_PATH=/data02/Qwen2.5-14B-Instruct/
MODEL_PATH=/data02/Qwen2.5-14B-Instruct-xft/
#OMP_NUM_THREADS=16 numactl -C 0-31 -l \
#OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') \
#OMP_NUM_THREADS=16 numactl -C $(seq -s, 0 2 $(($(lscpu | grep "^CPU(s):" | awk '{print $NF}') - 2))) \

# just for cloud
OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') \
  python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_PATH} \
  --tokenizer ${TOKEN_PATH} \
  --dtype bf16 \
  --kv-cache-dtype fp16 \
  --served-model-name xft \
  --port 8001 \
  --trust-remote-code
