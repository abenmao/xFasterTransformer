import tornado.ioloop
import tornado.web

import argparse

import logging

import torch
import time
import uuid
import json
import traceback
import transformers
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers import PreTrainedTokenizer

import os
import sys
from typing import Tuple, List
import string
from threading import Thread

import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

import xfastertransformer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


DTYPE_LIST = [
    "fp16",
    "bf16",
    "int8",
    "w8a8",
    "int4",
    "nf4",
    "bf16_fp16",
    "bf16_int8",
    "bf16_w8a8",
    "bf16_int4",
    "bf16_nf4",
    "w8a8_int8",
    "w8a8_int4",
    "w8a8_nf4",
]

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str,
                    default="/root/xFasterTransformer/examples/model_config/chatglm3-6b/",
                    help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/mnt/data/chatglm3-6b-xft/", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=DTYPE_LIST, default="bf16", help="Data type")
parser.add_argument("--padding", help="Enable padding, Default to False.", type=boolean_string, default=False)
parser.add_argument("--streaming", help="Streaming output, Default to True.", type=boolean_string, default=True)
parser.add_argument("--num_beams", help="Num of beams, default to 1 which is greedy search.", type=int, default=1)
parser.add_argument("--output_len", help="max tokens can generate excluded input.", type=int, default=500)
parser.add_argument("--chat", help="Enable chat mode, Default to False.", type=boolean_string, default=True)
parser.add_argument("--do_sample", help="Enable sampling search, Default to False.", type=boolean_string, default=True)
parser.add_argument("--temperature", help="value used to modulate next token probabilities.", type=float, default=1.0)
parser.add_argument("--top_p", help="retain minimal tokens above topP threshold.", type=float, default=1.0)
parser.add_argument("--top_k", help="num of highest probability tokens to keep for generation", type=int, default=50)
parser.add_argument("--rep_penalty", help="param for repetition penalty. 1.0 means no penalty", type=float, default=1.0)
parser.add_argument("--port", help="service port", type=int, default=8888)

model = None
tokenizer = None

args = parser.parse_args()

logging.basicConfig(filename='./log-' + str(args.port) + '.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def check_transformers_version_compatibility(token_path):
    config_path = os.path.join(token_path, "config.json")
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)

        transformers_version = config_data.get("transformers_version")
    except Exception as e:
        pass
    else:
        if transformers_version:
            if transformers.__version__ != transformers_version:
                print(
                    f"[Warning] The version of `transformers` in model configuration is {transformers_version}, and version installed is {transformers.__version__}. "
                    + "This tokenizer loading error may be caused by transformers version compatibility. "
                    + f"You can downgrade or reinstall transformers by `pip install transformers=={transformers_version} --force-reinstall` and try again."
                )


def load():
    global model
    global tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.token_path, use_fast=False, padding_side="left", trust_remote_code=True
        )
    except Exception as e:
        traceback.print_exc()
        print("[ERROR] An exception occurred during the tokenizer loading process.\t" + str(e))
        check_transformers_version_compatibility(args.token_path)
        sys.exit(-1)
    model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    streamer = None
    stop_words_ids = None
    # if model.rank == 0 and args.streaming and args.num_beams == 1:
    #     streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=args.chat)
    print("[INFO] xfastertransformer is installed, using pip installed package.")
    logging.info("xfastertransformer is installed, using pip installed package.")


class CdssHandler(tornado.web.RequestHandler):
    def post(self):
        body = self.request.body
        data = json.loads(body)
        self.set_header("Content-Type", "text/plain; charset=utf-8")
        # self.set_header("Transfer-Encoding", "chunked")
        try:
            prompt = data["messages"][0]["content"]
        except Exception as e:
            print("[ERROR] get content error.\t" + str(e))
            response = {
                "error_code": 336003,
                "error_msg": "message content can not be empty"
            }
            self.write(response)
            return

        if len(prompt) > 8000:
            logging.info("content to long")
            response = {
                "error_code": 336003,
                "error_msg": "message content to lang"
            }
            self.write(response)
            return
        output_len = 300
        output_len_max = 300
        logging.info("prompt info \t" + prompt)
        try:
            output_len_max = data["max_output_tokens"]
        except Exception as e:
            logging.info("get output_len error use default\t" + str(e))

        if isinstance(output_len_max, int):
            output_len = output_len_max
        temperature = 1.0
        try:
            temperature = data["temperature"]
        except Exception as e:
            logging.info("get temperature error use default\t" + str(e))
            temperature = 1.0

        if not isinstance(temperature, float):
            logging.info("get temperature error use default")
            temperature = 1.0
        top_p = 1.0
        try:
            top_p = data["top_p"]
        except Exception as e:
            logging.info("get top_p error use default\t" + str(e))
            top_p = 1.0

        if not isinstance(top_p, float):
            logging.info("get top_p error use default")
            top_p = 1.0

        # input_ids = build_inputs_chatglm(tokenizer, prompt, args.padding)
        # input_ids = tokenizer(prompt, return_tensors="pt", padding=args.padding).input_ids
        input_ids = tokenizer.build_chat_input(prompt, history=[], role="user").input_ids
        # print("=" * 50)
        start_time = time.perf_counter()
        created = time.time()
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=args.chat, )
        stop_words_ids = None
        logging.info("do_sample:")
        logging.info(args.do_sample)
        t = Thread(target=model.generate, kwargs={
            "input_ids": input_ids,
            "max_length": input_ids.shape[-1] + output_len,
            "streamer": streamer,
            "num_beams": args.num_beams,
            "stop_words_ids": stop_words_ids,
            "do_sample": args.do_sample,
            "temperature": temperature,
            "top_k": args.top_k,
            "top_p": top_p,
            "repetition_penalty": args.rep_penalty,
        })
        t.start()

        uid = str(uuid.uuid4())

        resultText = ""
        is_answer = False
        first_time = 0
        for new_text in streamer:
            # print("*" * 50)
            print(new_text)
            if new_text == "" or new_text == "\n" or new_text == "<|assistant|>" or new_text == "<|assistant|> \n" or new_text == "<|assistant|> ":
                continue
            if first_time == 0:
                first_time = time.perf_counter()
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            new_text = new_text.replace("<|assistant|> \n", "")
            resultText = resultText + new_text

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        latency_time = end_time - start_time
        # print("=" * 20 + "Performance" + "=" * 20)
        logging.info("=" * 20 + "Performance" + "=" * 20)
        input_len = torch.numel(input_ids)
        # print(f"Input:\t{input_len:.2f} s")
        logging.info(f"Input:\t{input_len:.2f} s")
        # print(f"Execution time:\t{execution_time:.2f} s")
        logging.info(f"Execution time:\t{execution_time:.2f} s")
        first_time = first_time - start_time
        # print(f"First time:\t{first_time:.2f} s")
        logging.info(f"First time:\t{first_time:.2f} s")
        # input_token_nums = torch.numel(input_ids)
        output_token_nums = len(resultText)
        latency = latency_time * 1000 / output_token_nums
        througput = (output_token_nums + input_len) / execution_time
        # print(f"Latency:\t{latency:.2f} ms/token")
        logging.info(f"Latency:\t{latency:.2f} ms/token")
        # print(f"Througput:\t{througput:.2f} tokens/s")
        logging.info(f"Througput:\t{througput:.2f} tokens/s")
        logging.info(f"Output:\t{output_token_nums:.2f}")

        result = {}
        usage = {}
        result["id"] = uid
        result["object"] = "chat.completion"
        result["created"] = created
        result["runtime"] = execution_time
        result["result"] = resultText
        result["first_time"] = first_time
        result["latency"] = latency
        result["througput"] = througput
        result["is_truncated"] = False
        result["need_clear_history"] = False
        result["finish_reason"] = "normal"
        result["usage"] = usage
        response = json.dumps(result, ensure_ascii=False)
        # print(response)
        self.write(response)


def build_inputs_chatglm(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    inputs = tokenizer(prompt, return_tensors="pt", padding=padding).input_ids
    return inputs


def make_app():
    return tornado.web.Application([
        (r"/llm-chatglm3", CdssHandler),
    ])


if __name__ == "__main__":
    load()
    app = make_app()
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()

