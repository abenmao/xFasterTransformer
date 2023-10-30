MODEL_NAME=llama-2-emr-dummy/7b-chat/
MODEL_NAME=llama-2-emr-dummy/13b-chat/
#MODEL_NAME=llama-2-emr-dummy/70b-chat/

TOKEN_NAME=llama-2-emr-dummy/7b-chat/
TOKEN_NAME=llama-2-emr-dummy/13b-chat/
#TOKEN_NAME=llama-2-emr-dummy/70b-chat/

LOOP=3
DTYPE=bf16
#DTYPE=bf16_fp16
SeqLen=64
OutLen=100
BSIZE=16

export ENABLE_FLASH_ATTN=0
export ENABLE_FAKE_MODEL=1

node=$1
nth=$2

if [ "$node" -eq 1 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 numactl -C  0-`expr $nth - 1` -m 0 ./example -m /home/mengchen/models/${MODEL_NAME} -t /home/mengchen/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --no_stream 

#12 -i "Simply put, the theory of relativity states that" 

#18 -i "Once upon a time, there existed a little girl who liked to have adventures."

elif [ "$node" -eq 2 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 taskset -c 0-`expr $nth - 1` numactl -C  0-`expr $nth - 1` -m 0 ./example -m /home/mengchen/models/${MODEL_NAME} -t /home/mengchen/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --no_stream : \
        -n 1 taskset -c `expr $nth`-`expr $nth \* 2 - 1` numactl -C  `expr $nth`-`expr $nth \* 2 - 1` -m 0 ./example -m /home/mengchen/models/${MODEL_NAME} -t /home/mengchen/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --no_stream

elif [ "$node" -eq 4 ];then

OMP_NUM_THREADS=${nth} mpirun -n 1 numactl -C 0-`expr $nth - 1` -m 0 ./example -m /home/mengchen/models/${MODEL_NAME} -t /home/mengchen/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --no_stream : \
        -n 1 numactl -C  `expr $nth`-`expr $nth \* 2 - 1` -m 0 ./example -m /home/mengchen/models/${MODEL_NAME} -t /home/mengchen/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --no_stream : \
        -n 1 numactl -C  `expr $nth \* 2`-`expr $nth \* 3 - 1` -m 1 ./example -m /home/mengchen/models/${MODEL_NAME} -t /home/mengchen/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --no_stream : \
        -n 1 numactl -C  `expr $nth \* 3`-`expr $nth \* 4 - 1` -m 1 ./example -m /home/mengchen/models/${MODEL_NAME} -t /home/mengchen/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --no_stream

fi
