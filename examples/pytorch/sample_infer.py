# -*- coding: utf-8 -*-
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import pdb
import numpy as np
import time

# set the path of the model directory
#model_dir = "/home/mengchen/models/Llama-2-7b-chat-hf"
model_dir = "/home/mengchen/models/llama-2-13b-chat-hf"
#model_dir = "/home/mengchen/models/TinyLlama-1.1B-Chat-v1.0"
draft_model_dir = "/home/mengchen/models/TinyLlama-1.1B-Chat-v1.0"
#draft_model_dir = "/home/mengchen/models/Llama-2-7b-chat-hf"

lookahead_K = 8
AJUST_K = True #False dynamic lookahead_k or not
accepted_P = 0.85 # when less than 1, it would be more tolerance
batch_S = 24
max_gen_tokens = 512

prompt = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level"

def load(model_dir):
    device = torch.device('cpu:0')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir).bfloat16().to(device)
    return tokenizer, model

def encode(tokenizer, prompt):
    return tokenizer.encode(prompt, return_tensors="pt")

def draft_lookahead(model, inputs, past_key_values=None, lookahead_k=16):
    #outputs = model.generate(inputs, max_new_tokens=lookahead_k,
    #                output_scores=True,
    #                return_dict_in_generate=True, do_sample=False)
    scores = []
    sequences = inputs

    with torch.no_grad():
        accu_tokens = 0
        while accu_tokens < lookahead_k:
            output = model.forward(inputs,
                    past_key_values=past_key_values if past_key_values is not None else ())
            next_tokens_scores = output.logits[:, -1, :]
            scores.append(next_tokens_scores)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)
            inputs = sequences[:, -1:]
            past_key_values = output.past_key_values
            accu_tokens += 1

    scores = torch.stack(scores).view(-1, lookahead_k, scores[0].shape[-1])
    outputs = {'sequences' : sequences, 'scores' : scores}
    return outputs, output.past_key_values

def validate(model, lookaheads, past_key_values=None, accepted_p=0.95):
    #outputs = model.generate(inputs, max_new_tokens=1,
    #                output_scores=True,
    #                output_attentions=True,
    #                output_hidden_states=True,
    #                return_dict_in_generate=True, do_sample=False)
    inputs = lookaheads['sequences']
    logits_draft = lookaheads['scores']
    lookahead_k = logits_draft.shape[1]

    with torch.no_grad():
        output = model.forward(inputs,
            past_key_values=past_key_values if past_key_values is not None else ())
        # logits for lookahead tokens
        logits = output.logits[:, -lookahead_k-1:, :]
        next_tokens = torch.argmax(logits, dim=-1)

    batch_size = logits.shape[0]

    # validate if logits are accepted based on accepted_p
    accepted_k = lookahead_k
    for b in range(batch_size):
        for i in range(lookahead_k):
            #tidx = inputs[b][-lookahead_k + i]
            #p = logits[b][i][tidx]
            #q = logits_draft[b][i][tidx]
            # p / q < accepted_p ?
            if next_tokens[b][i] != inputs[b][i - lookahead_k]:
                accepted_k = min(i, lookahead_k)
                break
            
 
    if (accepted_k < lookahead_k):
        next_tokens_scores = logits[:, accepted_k]
        sequences = inputs[:, : accepted_k - lookahead_k]
    else:
        next_tokens_scores = logits[:, -1]
        sequences = inputs

    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

    tmp = torch.argmax(logits, dim=-1)
    print("model: ", len(tmp[0]), tokenizer.batch_decode(tmp[0], skip_special_tokens=True))
    print("draft: ", len(inputs[0]), tokenizer.batch_decode(inputs[0], skip_special_tokens=True))
    print(len(sequences[0]), tokenizer.batch_decode(sequences[0], skip_special_tokens=True))

    return accepted_k, sequences, output.past_key_values

def update_inputs_and_kvs(accepted_k, lookahead_k, seqs, draft_past_kvs, past_kvs, draft_model):
    inputs = seqs
    kvs_len = past_kvs[0][0].shape[-2] - (lookahead_k - accepted_k)

    past_kvs = tuple(
                tuple(past_tensor[:, :, : kvs_len, :] for past_tensor in past_kv) 
                        for past_kv in past_kvs
                )
    # fill draft past kv
    if accepted_k == lookahead_k:
        draft_last = draft_model.forward(inputs[:, -2:-1],
                past_key_values=draft_past_kvs)
        draft_past_kvs = draft_last.past_key_values

    draft_past_kvs = tuple(
                tuple(past_tensor[:, :, : kvs_len, :] for past_tensor in past_kv) 
                        for past_kv in draft_past_kvs
                )
 
    inputs = inputs[:, -1:]
    return inputs, draft_past_kvs, past_kvs

def dynamic_adjust_lookahead(accepted_k, acpt_k_history, lookahead_k):
    acpt_k_history.append(accepted_k)

    if (not AJUST_K):
        lookahead_k = lookahead_k
    elif (accepted_k <= lookahead_k / 2):
        lookahead_k = max(4, lookahead_k / 2)
    elif (accepted_k == lookahead_k):
        lookahead_k = min(16, lookahead_k * 2)

    print("accepted_k: ", accepted_k, "mean: ", np.mean(acpt_k_history), "lookahead: ", lookahead_k)
    return lookahead_k, acpt_k_history 

if __name__ == "__main__":

    print(f'prompt: {prompt}')
    tokenizer, model = load(model_dir)
    draft_tokenizer, draft_model = load(draft_model_dir)

    input = encode(tokenizer, prompt)
    inputs = input
    for i in range(batch_S - 1):
        inputs = torch.cat([inputs, input])
    pdb.set_trace()
    print(f'inputs: {inputs.shape}')

    accu_gen_tokens = 0
    accepted_k = 0
    draft_past_kvs = None
    past_kvs = None
    acpt_k_history = []
   
    t1 = time.time() 
    outputs = model.generate(inputs, max_new_tokens=max_gen_tokens,
                    output_scores=True,
                    return_dict_in_generate=True, do_sample=False)
    t2 = time.time()
    print("generate: ", t2-t1)
    print(tokenizer.batch_decode(outputs.sequences[0], skip_special_tokens=True))

    tt1 = time.time()
    while accu_gen_tokens < max_gen_tokens:
        # generate lookahead_k tokens using draft model
        #pdb.set_trace()
        output_k, draft_past_kvs = draft_lookahead(draft_model, inputs, draft_past_kvs, lookahead_K)
        
        # validate the lookahead_k tokens using the large model
        #pdb.set_trace()
        accepted_k, outputs, past_kvs = validate(model, output_k, past_kvs, accepted_P)

        # update the num of gen tokens
        accu_gen_tokens += accepted_k + 1

        #pdb.set_trace()
        inputs, draft_past_kvs, past_kvs = update_inputs_and_kvs(accepted_k, lookahead_K, outputs,
                                            draft_past_kvs, past_kvs, draft_model)
        # dynamic adjust lookahead_K
        lookahead_K, acpt_k_history = dynamic_adjust_lookahead(accepted_k, acpt_k_history, lookahead_K)
    
    tt2 = time.time()
    print("specInfer: ", tt2-tt1)
