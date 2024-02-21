import sys

def embedding(model_config, task_config, seqLen, run_perf=None):
    hiddenSize = model_config["hidden_size"]
    vocabSize = model_config["vocab_size"]

    batchSize = task_config["batch_size"]
    numBytes = task_config["num_bytes"]

    cost = dict()
    cost["flop_cnt"] = 0 # just memory copy
    cost["act_size"] = (batchSize * seqLen) * hiddenSize * numBytes
    cost["weight_size"] = min(cost["act_size"], vocabSize * hiddenSize * numBytes)

    if (run_perf is not None):
        effi = efficiency(model_config, cost, run_perf, sys._getframe().f_code.co_name)
        print("gen token ", run_perf["token_id"], " ", sys._getframe().f_code.co_name, " cost: ", cost, "\n    ", effi)

    return cost


def attn_linear(model_config, task_config, inptLen, run_perf=None):
    hiddenSize = model_config["hidden_size"]
    numQHeads = model_config["num_qheads"]
    numKVHeads = model_config["num_kvheads"]
    headSize = model_config["head_size"]

    batchSize = task_config["batch_size"]
    numBytes = task_config["num_bytes"]

    qkvCols = (numQHeads + 2 * numKVHeads) * headSize

    # qkv MatMul
    qkv_flop = (batchSize * inptLen) * hiddenSize * qkvCols * 2
    qkv_act = (batchSize * inptLen) * hiddenSize * numBytes 
    qkv_wei = hiddenSize * qkvCols * numBytes

    # out MatMul
    out_flop = (batchSize * inptLen) * hiddenSize * hiddenSize * 2 
    out_act = (batchSize * inptLen) * hiddenSize * numBytes 
    out_wei = hiddenSize * hiddenSize * numBytes
    
    cost = dict()
    cost["flop_cnt"] = qkv_flop + out_flop
    cost["act_size"] = qkv_act + out_act
    cost["weight_size"] = qkv_wei + out_wei

    if (run_perf is not None):
        effi = efficiency(model_config, cost, run_perf, sys._getframe().f_code.co_name)
        print("gen token ", run_perf["token_id"], " ", sys._getframe().f_code.co_name, " cost: ", cost, "\n    ", effi)

    return cost


def sdpa(model_config, task_config, inptLen, accSeqLen, run_perf=None):
    hiddenSize = model_config["hidden_size"]
    numQHeads = model_config["num_qheads"]
    numKVHeads = model_config["num_kvheads"]
    headSize = model_config["head_size"]

    batchSize = task_config["batch_size"]
    numBytes = task_config["num_bytes"]
    # scaledDpAttn
    sdpa_flop = (batchSize * numQHeads) * (inptLen * headSize * accSeqLen * 2) * 2
    if (task_config["flash_attn"]):
        # bmm1 input act
        sdpa_act = (batchSize * numQHeads) * (inptLen * headSize) * numBytes
        # bmm2 input act
        sdpa_act += (batchSize * numQHeads) * (inptLen * accSeqLen) * numBytes
    else:
        # bmm1 input act
        sdpa_act = (batchSize * numQHeads) * (inptLen * headSize) * numBytes
        # bmm2 input act
        sdpa_act += 0 # ideally in cache(?)
    sdpa_wei = 0

    cost = dict()
    cost["flop_cnt"] = sdpa_flop
    cost["act_size"] = sdpa_act
    cost["weight_size"] = sdpa_wei

    if (run_perf is not None):
        effi = efficiency(model_config, cost, run_perf, sys._getframe().f_code.co_name)
        print("gen token ", run_perf["token_id"], " ", sys._getframe().f_code.co_name, " cost: ", cost, "\n    ", effi)

    return cost


def attention(model_config, task_config, inptLen, accSeqLen, run_perf=None):

    # qkv MatMul / out MatMul
    linear_cost = attn_linear(model_config, task_config, inptLen, run_perf)
    # scaledDPAttn
    sdpa_cost = sdpa(model_config, task_config, inptLen, accSeqLen, run_perf)

    cost = dict()
    cost["flop_cnt"] = linear_cost["flop_cnt"] + sdpa_cost["flop_cnt"]
    cost["act_size"] = linear_cost["act_size"] + sdpa_cost["act_size"]
    cost["weight_size"] = linear_cost["weight_size"] + sdpa_cost["weight_size"]

    if (run_perf is not None):
        effi = efficiency(model_config, cost, run_perf, sys._getframe().f_code.co_name)
        print("gen token ", run_perf["token_id"], " ", sys._getframe().f_code.co_name, " cost: ", cost, "\n    ", effi)

    return cost

def mlp(model_config, task_config, inptLen, run_perf=None):
    hiddenSize = model_config["hidden_size"]
    interSize = model_config["intermediate_size"]

    batchSize = task_config["batch_size"]
    numBytes = task_config["num_bytes"]
   
    up_flop = (batchSize * inptLen) * hiddenSize * interSize * 2 
    up_act = (batchSize * inptLen) * hiddenSize * numBytes
    up_wei = hiddenSize * interSize * numBytes
    if (model_config["mlp_gate"]): 
        #plus gate MatMul
        up_flop *= 2
        up_act *= 2
        up_wei *= 2
        
    down_flop = (batchSize * inptLen) * interSize * hiddenSize * 2 
    down_act = (batchSize * inptLen) * interSize * numBytes
    down_wei = interSize * hiddenSize * numBytes
    
    cost = dict()
    cost["flop_cnt"] = up_flop + down_flop
    cost["act_size"] = up_act + down_act
    cost["weight_size"] = up_wei + down_wei

    if (run_perf is not None):
        effi = efficiency(model_config, cost, run_perf, sys._getframe().f_code.co_name)
        print("gen token ", run_perf["token_id"], " ", sys._getframe().f_code.co_name, " cost: ", cost, "\n    ", effi)

    return cost

def transformer(model_config, task_config, inptLen, accSeqLen, run_perf=None):
    attn_cost = attention(model_config, task_config, inptLen, accSeqLen, run_perf)
    mlp_cost = mlp(model_config, task_config, inptLen, run_perf)
    tf_flop = attn_cost["flop_cnt"] + mlp_cost["flop_cnt"] 
    tf_act = attn_cost["act_size"] + mlp_cost["act_size"] 
    tf_wei = attn_cost["weight_size"] + mlp_cost["weight_size"] 

    cost = dict()
    cost["flop_cnt"] = tf_flop
    cost["act_size"] = tf_act
    cost["weight_size"] = tf_wei

    if (run_perf is not None):
        effi = efficiency(model_config, cost, run_perf, sys._getframe().f_code.co_name)
        print("gen token ", run_perf["token_id"], " ", sys._getframe().f_code.co_name, " cost: ", cost, "\n    ", effi)

    return cost
    

def predictor(model_config, task_config, inptLen, run_perf=None):
    hiddenSize = model_config["hidden_size"]
    vocabSize = model_config["vocab_size"]

    batchSize = task_config["batch_size"]
    numBytes = task_config["num_bytes"]

    cost = dict()

    if (task_config["all_logits"]):
        outLen = inptLen
    else:
        outLen = 1
    cost["flop_cnt"] = (batchSize * outLen) * hiddenSize * vocabSize * 2 
    cost["act_size"] = (batchSize * outLen) * vocabSize * numBytes
    cost["weight_size"] = hiddenSize * vocabSize * numBytes

    if (run_perf is not None):
        effi = efficiency(model_config, cost, run_perf, sys._getframe().f_code.co_name)
        print("gen token ", run_perf["token_id"], " ", sys._getframe().f_code.co_name, " cost: ", cost, "\n    ", effi)

    return cost

def gen_token(model_config, task_config, gen_token_id, run_perf=None):
    numLayers = model_config["num_layers"]
    if (gen_token_id == 0):
        inptLen = task_config["prompt_len"]
    else:
        inptLen = 1
    accSeqLen = task_config["past_len"] + task_config["prompt_len"] + gen_token_id

    embedding_cost = embedding(model_config, task_config, inptLen, run_perf)
    transformer_cost = transformer(model_config, task_config, inptLen, accSeqLen, run_perf)
    predictor_cost = predictor(model_config, task_config, inptLen, run_perf)

    first_flop = embedding_cost["flop_cnt"] + numLayers * transformer_cost["flop_cnt"] + predictor_cost["flop_cnt"]
    first_act = embedding_cost["act_size"] + numLayers * transformer_cost["act_size"] + predictor_cost["act_size"]
    first_wei = embedding_cost["weight_size"] + numLayers * transformer_cost["weight_size"] + predictor_cost["weight_size"]

    cost = dict()
    cost["flop_cnt"] = first_flop
    cost["act_size"] = first_act
    cost["weight_size"] = first_wei
    
    if (run_perf is not None):
        assert(gen_token_id == run_perf["token_id"])
        effi = efficiency(model_config, cost, run_perf, sys._getframe().f_code.co_name)
        print("gen token ", gen_token_id, " ", sys._getframe().f_code.co_name, " cost: ", cost, "\n    ", effi, "\n")

    return cost

def efficiency(model_config, cost, run_perf, name):

    flop_cnt = (float)(cost["flop_cnt"]) / 1.0e9 # Gflop since latency is ms
    mem_size = (float)(cost["act_size"] + cost["weight_size"]) / 1.0e6 # MB since latency is ms
    latency = run_perf[name]

    eff = dict()
    eff["flops"] = flop_cnt / latency
    eff["membw"] = mem_size / latency
    if (name == "embedding" or name == "predictor" or name == "gen_token"):
        eff["ratio"] = latency / run_perf["gen_token"]
    else:
        eff["ratio"] = model_config["num_layers"] * latency / run_perf["gen_token"]

    return eff
    

if __name__ == "__main__":
    ##############################ConfigBegin######################################
    model_config = dict()
    model_config["hidden_size"] = 4096
    model_config["intermediate_size"] = 11008
    model_config["num_layers"] = 32
    model_config["num_qheads"] = 32
    model_config["num_kvheads"] = 32
    model_config["head_size"] = 128
    model_config["vocab_size"] = 32000
    model_config["mlp_gate"] = True # standard: false

    task_config = dict()
    task_config["batch_size"] = 1
    task_config["prompt_len"] = 1024
    task_config["past_len"] = 0
    task_config["output_len"] = 128
    task_config["beam_size"] = 1
    task_config["flash_attn"] = True # non-opt: false
    task_config["all_logits"] = False # output all logits: true
    task_config["num_bytes"] = 2 #fp32:4, fp16/bf16:2 int8:1 int4:0.5(?)

    run_perf = dict()
    run_perf[0] = dict()
    run_perf[0]["token_id"] = 0
    run_perf[0]["gen_token"] = 764 # ms
    run_perf[0]["embedding"] = 1.509 # ms
    run_perf[0]["sdpa"] = 4.42 # 3.75 ms
    run_perf[0]["attention"] = 11.5 # ms
    run_perf[0]["attn_linear"] = run_perf[0]["attention"] - run_perf[0]["sdpa"] # ms
    run_perf[0]["mlp"] = 12.3 # ms
    run_perf[0]["transformer"] = 23.5 # ms
    run_perf[0]["predictor"] = 1.185 # ms
    run_perf[1] = dict()
    run_perf[1]["token_id"] = 1
    run_perf[1]["gen_token"] = 67 # ms
    run_perf[1]["embedding"] = 0.003 # ms
    run_perf[1]["sdpa"] = 0.2 # ms
    run_perf[1]["attention"] = 0.85 # ms
    run_perf[1]["attn_linear"] = run_perf[1]["attention"] - run_perf[1]["sdpa"] # ms
    run_perf[1]["mlp"] = 1.33 # ms
    run_perf[1]["transformer"] = 2.2 # ms
    run_perf[1]["predictor"] = 1.139 # ms
    ##############################ConfigEnd########################################

    print("Hardware Limit--1 SPR socket (hflops: 122T/s, bandwidth :300GB/s)")
    print("Hardware Limit--1 SPR+HBM socket (hflops: 122T/s, bandwidth :1000GB/s)")
    print("Run Perf: ")
    
    #first_cost = gen_token(model_config, task_config, 0)
    #second_cost = gen_token(model_config, task_config, 1)
    first_cost = gen_token(model_config, task_config, 0, run_perf[0])
    second_cost = gen_token(model_config, task_config, 1, run_perf[1])
