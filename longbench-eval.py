import os
import sys
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
# from vllm import LLM, SamplingParams

from replace_atten_cpe_chat import replace_with_cpellama
import math
import textwrap
from transformers import GenerationConfig, TextStreamer

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama2-7b-bsrope-16k-v3")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    prompt = f"[INST]{prompt}[/INST]"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # chat models are better off without build prompts on these tasks
        if dataset not in ["trec", "triviaqa", "samsum", "lsht"]: 
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=0.6,
                top_p=0.9,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=0.6,
                top_p =0.9,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    replace_with_cpellama(4096)
    tokenizer = LlamaTokenizer.from_pretrained("/home/s1023244019/models/llama-2-7b-chat-hf")
    model = LlamaForCausalLM.from_pretrained("/home/s1023244019/models/cpe-llama2-7b-chat-hf-v0/", \
                                        attn_implementation="flash_attention_2", \
                                        trust_remote_code=True, \
                                        torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("/home/s1023244019/LongLoRA/test_config/model2path.json", "r"))
    model2maxlen = json.load(open("/home/s1023244019/LongLoRA/test_config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    max_length = model2maxlen[model_name]
    
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
          "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
           "passage_count", "passage_retrieval_en", "lcc", "repobench-p"] 
    # datasets = ["lcc", "passage_count", "passage_retrieval_en", "trec", "triviaqa", "samsum"]
    # datasets = ["triviaqa", "samsum", "lcc", "repobench-p"]
    dataset2prompt = json.load(open("/home/s1023244019/LongLoRA/test_config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("/home/s1023244019/LongLoRA/test_config/dataset2maxlen.json", "r"))
    
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    dataset_path = "/home/s1023244019/LongLoRA/benchmarks/LongBench"
    
    for dataset in datasets:
        if args.e:
            data = load_dataset(dataset_path, f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('json', data_files=os.path.join(dataset_path, f"{dataset}.jsonl"), split='train')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        # Add the script on data process.
        
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()