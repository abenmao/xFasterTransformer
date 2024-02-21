#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
@Author  : diaorenyan (diaorenyan@baidu.com)
@Date    : 2024/1/31 
@Brief   :
"""
import argparse
import torch
import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("../../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")
import xfastertransformer
from transformers import (
    AutoTokenizer,
)


model_dir=""
token_dir=""
model_name=""
argparser = argparse.ArgumentParser()
argparser.add_argument("--token_dir", required=True, type=str, default="", help="token dir")
argparser.add_argument("--model_dir", required=True, type=str, default="", help="model dir")
argparser.add_argument("--model_name", required=False, type=str, help="llm to load")
argparser.add_argument("--max_new_tokens", required=False, type=int, default=50)


args = argparser.parse_args()
model_dir = args.model_dir
token_dir = args.token_dir
model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(token_dir, use_fast=False, padding_side='left', trust_remote_code=True)
#model = xfastertransformer.AutoModel.from_pretrained(model_dir, dtype="fp16")
model = xfastertransformer.AutoModel.from_pretrained(model_dir, dtype="int8")
#model = xfastertransformer.AutoModel.from_pretrained(model_dir, dtype="bf16")

prompt = "你是一辆汽车"
inputs = tokenizer.encode(prompt, return_tensors="pt")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(inputs)
print(input_ids)
input_token_num = int(torch.numel(input_ids))

import pdb
pdb.set_trace()
outputs = model.generate(input_ids, max_length=args.max_new_tokens+input_token_num, repetition_penalty=1.05)
print(tokenizer.decode(outputs[0]))
