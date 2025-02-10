"""
Install:
python 3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
pip install transformers==4.37.0
"""

import re
import sys
import json
import torch

import argparse

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import transformers


import json, os, pickle
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

from pathlib import Path
import json
from pathlib import Path
from collections import Counter
import pickle, os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluate():
    def __init__(self, args):

        self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
        self.model = self.model.to(device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if 'gpt' in args.model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = 100
        self.batch_size = args.batch_size
        self.data = args.data


    def perplexity(self):

        ### dataload
        data = self.data

        encodings = self.tokenizer(
            data,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True
        ).to(device)
        input_ids = encodings.input_ids
        attn_masks = encodings.attention_mask

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in (range(0, len(input_ids), self.batch_size)):
            end_index = min(start_index + self.batch_size, len(input_ids))
            encoded_batch = input_ids[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]
            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits
            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='null')
    parser.add_argument('--model_path', type=str, default="bigcode/starcoderbase-1b", help='generation model path')
    parser.add_argument('--batch_size', '-bs', type=int, default=1, help='batch_size')

    args = parser.parse_args()

    with open('./auto-leak-detect-bench-balanced.pkl', 'rb') as fr:
        all_ids, all_labels, all_inputs = pickle.load(fr)

    all_ppls = []


    for start_index_ in tqdm((range(0, len(all_inputs), 10))):
        end_index_ = min(start_index_ + 10, len(all_inputs))
        args.data = all_inputs[start_index_:end_index_]
        evaluate = Evaluate(args)
        perplexity = evaluate.perplexity()
        all_ppls.append(perplexity)

        print(perplexity)

    print(len(all_ppls), all_ppls[0])
    print(len(all_inputs), len(all_ids), len(all_data))

    new_all_ppls = []
    for l in all_ppls:
        new_all_ppls.extend(l['perplexities'])
    del all_ppls
    all_ppls = new_all_ppls

    os.mkdir('ppl_results')
    with open('ppl_results/all_ppls_'+args.model_path.split('/')[-1]+'-balance.json', 'wb') as f:
        pickle.dump( (all_ppls, all_inputs, all_labels, all_ids), f)


