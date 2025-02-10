
"""
From: https://github.com/chaemino/perplexity-for-local-model


实验的位置：xinzhou@10.0.104.96 (Saturn)
/mnt/hdd2/xinzhou/UER-py-master/Data_Contamination/20040819

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


import json, os
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

from pathlib import Path
import json
from pathlib import Path
from collections import Counter
import pickle, os
import matplotlib.pyplot as plt
import random


def list_files_in_directory(path):
    return [f.resolve() for f in Path(path).rglob('*') if f.is_file()]

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
    parser = argparse.ArgumentParser(description='Sem Sim')
    parser.add_argument('--model_path', type=str, default="bigcode/starcoderbase-7b", help='generation model path')
    parser.add_argument('--batch_size', '-bs', type=int, default=1, help='batch_size')

    args = parser.parse_args()



    with open('ppl_results/all_ppls_'+args.model_path.split('/')[-1]+'-balance.json', 'rb') as f:
        all_ppls_, all_inputs, all_labels, all_ids = pickle.load(f)


    all_ppls = all_ppls_

    duplicate_group, non_duplicate_group = [],[]
    for i in range(len(all_ids)):
            if all_labels[i] in ['2', '3']:
                duplicate_group.append(all_ppls[i])
            elif all_labels[i] in ['0', '1']:
                non_duplicate_group.append(all_ppls[i])

    # 示例数据
    random.seed(0)
    list1 = duplicate_group
    list2 = non_duplicate_group

    def remove_outliers(data, percentile=4):
        # 计算数据的下百分位数和上百分位数
        lower_bound = np.percentile(data, percentile / 2)
        upper_bound = np.percentile(data, 100 - percentile / 2)
        return [x for x in data if x <= upper_bound]


    # 去除离群值
    filtered_list1 = remove_outliers(list1)
    filtered_list2 = remove_outliers(list2)
    print(len(filtered_list1), len(filtered_list2))


    # 绘制直方图
    plt.hist(filtered_list1, bins=120, color='blue', alpha=0.5, label='Leaked Samples')
    plt.hist(filtered_list2, bins=120, color='orange', alpha=0.5, label='Non-leaked Samples')

    # 设置 Y 轴为对数刻度
    # plt.yscale('log')

    # 添加标签和图例
    plt.xlabel('Perplexity Scores')
    plt.ylabel('#Sample Count')
    plt.legend()

    # 显示图形
    plt.show()
