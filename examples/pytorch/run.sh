OMP_NUM_THREADS=48 \
numactl -C 0-47 -l python demo.py \
        --token_path /home/mengchen/models/Baichuan2-7B-Chat/ \
        --model_path /home/mengchen/models/Baichuan2-7B-Chat-cpu/ \
        --dtype int8 \
        --output_len 1024 \
        --chat False \
        --padding false

        #--token_path /home/mengchen/models/Llama-2-7b-chat-hf/ \
        #--model_path /home/mengchen/models/Llama-2-7b-chat-cpu/ \
        #--token_path /home/mengchen/models/Baichuan2-7B-Chat/ \
        #--model_path /home/mengchen/models/Baichuan2-7B-Chat-cpu/ \
        #--token_path /home/mengchen/models/Baichuan2-7B-Base/ \
        #--model_path /home/mengchen/models/Baichuan2-7B-Base-cpu/ \
        #--token_path /home/mengchen/models/chatglm2-6b-hf/ \
        #--model_path /home/mengchen/models/chatglm2-6b-cpu/cpu \
