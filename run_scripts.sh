#BF16
export XFT_FAKE_MODEL=1
SCRIPT=run.sh.mulinst #run.sh.3ranks #run.sh.mulinst
dtype=fp16
kv_dtype=fp16
#sh ${SCRIPT} 1 40 ${dtype} ${kv_dtype} 2>&1 | tee h${dtype}_${kv_dtype}kv_bs1_3ranks
#sh ${SCRIPT} 8 40 ${dtype} ${kv_dtype} 2>&1 | tee h${dtype}_${kv_dtype}kv_bs8_3ranks
#sh ${SCRIPT} 16 40 ${dtype} ${kv_dtype} 2>&1 | tee h${dtype}_${kv_dtype}kv_bs16_3ranks
#sh ${SCRIPT} 32 40 ${dtype} ${kv_dtype} 2>&1 | tee h${dtype}_${kv_dtype}kv_bs32_3ranks
#sh ${SCRIPT} 64 40 ${dtype} ${kv_dtype} 2>&1 | tee h${dtype}_${kv_dtype}kv_bs64_3ranks

sh ${SCRIPT} 1 40 ${dtype} ${kv_dtype} "0 40 80" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs1_40x3
sh ${SCRIPT} 8 40 ${dtype} ${kv_dtype} "0 40 80" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs8_40x3
sh ${SCRIPT} 16 40 ${dtype} ${kv_dtype} "0 40 80" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs16_40x3
sh ${SCRIPT} 32 40 ${dtype} ${kv_dtype} "0 40 80" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs32_40x3
sh ${SCRIPT} 64 40 ${dtype} ${kv_dtype} "0 40 80" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs64_40x3

sh ${SCRIPT} 1 20 ${dtype} ${kv_dtype} "0 20 40 60 80 100" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs1_20x6
sh ${SCRIPT} 8 20 ${dtype} ${kv_dtype} "0 20 40 60 80 100" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs8_20x6
sh ${SCRIPT} 16 20 ${dtype} ${kv_dtype} "0 20 40 60 80 100" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs16_20x6
sh ${SCRIPT} 32 20 ${dtype} ${kv_dtype} "0 20 40 60 80 100" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs32_20x6
sh ${SCRIPT} 64 20 ${dtype} ${kv_dtype} "0 20 40 60 80 100" 2>&1 | tee ${dtype}_${kv_dtype}kv_bs64_20x6
