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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


model_dir=""
model_name=""
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_dir", required=True, type=str, default="", help="model dir")
argparser.add_argument("--model_name", required=False, type=str, help="llm to load")
argparser.add_argument("--max_new_tokens", required=False, type=int, default=50)

args = argparser.parse_args()
model_dir = args.model_dir
model_name = args.model_name

device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, padding_side='left', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).half().to(device)

prompt = "你是一辆汽车"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
print(inputs)
outputs = model.generate(inputs, max_new_tokens=args.max_new_tokens)
print(outputs)
print(tokenizer.decode(outputs[0]))