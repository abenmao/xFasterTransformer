TOKEN_PATH=/home/mengchen/models/Qwen2.5-14B-Instruct/
MODEL_PATH=/home/mengchen/models/Qwen2.5-14B-Instruct-xft/
OMP_NUM_THREADS=48 numactl -C 0-47,96-143 -l python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --tokenizer ${TOKEN_PATH} \
        --dtype bf16 \
        --kv-cache-dtype fp16 \
        --served-model-name xft \
        --port 8001 \
        --trust-remote-code
