#ENABLE_KV_TRANS=0 \
#ENABLE_TUNED_COMM=0 \
#FLASH_ATTN_THRESHOLD=8192 \
OMP_NUM_THREADS=48 \
numactl -C 0-47 -l python demo.py \
        --token_path /data/models/deepseek-coder-33b-instruct/ \
        --model_path /data/models/deepseek-coder-33b-instruct-xft/ \
        --dtype bf16 \
        --output_len 512 \
        --chat False \
        --do_sample False \
        --rep_penalty 1.0 \
        --temperature 0.0 \
        --top_k 20 \
        --top_p 0.8 \
        #--streaming false \
        #--padding false \
        #--rep_penalty 1.2

        #--token_path /home/mengchen/models/Qwen2.5-14B-Instruct/ \
        #--model_path /home/mengchen/models/Qwen2.5-14B-Instruct-xft/ \
        #--token_path /home/mengchen/models/Llama-2-7b-chat-hf/ \
        #--model_path /home/mengchen/models/Llama-2-7b-chat-cpu/ \
        #--token_path /home/mengchen/models/Baichuan2-7B-Chat/ \
        #--model_path /home/mengchen/models/Baichuan2-7B-Chat-cpu/ \
        #--token_path /home/mengchen/models/Baichuan2-7B-Base/ \
        #--model_path /home/mengchen/models/Baichuan2-7B-Base-cpu/ \
        #--token_path /home/mengchen/models/Baichuan2-13B-Chat/ \
        #--model_path /home/mengchen/models/Baichuan2-13B-Chat-xft \
        #--token_path /home/mengchen/models/chatglm2-6b-hf/ \
        #--model_path /home/mengchen/models/chatglm2-6b-cpu/cpu \
        #--token_path /home/mengchen/models/chatglm3-6b-hf/ \
        #--model_path /home/mengchen/models/chatglm3-6b-cpu \
        #--token_path /home/mengchen/models/gemma-7b-it/ \
        #--model_path /home/mengchen/models/gemma-7b-it-xft \
        #--token_path /home/mengchen/models/Llama3-8B-Chinese-Chat/ \
        #--model_path /home/mengchen/models/Llama3-8B-Chinese-Chat-cpu \

