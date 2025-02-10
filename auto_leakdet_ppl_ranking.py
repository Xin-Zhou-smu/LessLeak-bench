
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

import matplotlib.pyplot as plt


def remove_outliers(data, percentile=5):
    # 计算数据的下百分位数和上百分位数
    lower_bound = np.percentile(data, percentile / 2)  # 2.5%
    upper_bound = np.percentile(data, 100 - percentile / 2)  # 97.5%
    return [x for x in data if x <= upper_bound]


def obtain_detect_results (model_path):
    with open('ppl_results/all_ppls_' + model_path.split('/')[-1] + '-balance.json', 'rb') as f:
        all_ppls_, all_inputs, all_labels, all_ids = pickle.load(f)

    all_ppls = all_ppls_

    duplicate_group, non_duplicate_group = [], []
    for i in range(len(all_ids)):
        if all_labels[i] in ['2', '3']:
            duplicate_group.append(all_ppls[i])
        elif all_labels[i] in ['0', '1']:
            non_duplicate_group.append(all_ppls[i])

    ppl_data = duplicate_group + non_duplicate_group
    label_data = [1] * len(duplicate_group) + [0] * len(non_duplicate_group) ## Here 1--> leak; 0-->non leak

    k_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    print('k_values:', k_values)

    def compute_ratios(data_list, labels, k_values):
        """对单个list和label计算比例"""
        sorted_indices = sorted(range(len(data_list)), key=lambda i: data_list[i])
        sorted_labels = [labels[i] for i in sorted_indices]
        return [sum(sorted_labels[:k]) / k for k in k_values]

    ratios = compute_ratios(ppl_data, label_data, k_values)
    return ratios, k_values


model_path = 'bigcode/starcoderbase-1b'
ratios_1b, k_values = obtain_detect_results (model_path)
model_path = 'bigcode/starcoderbase-3b'
ratios_3b, _ = obtain_detect_results (model_path)
model_path = 'bigcode/starcoderbase-7b'
ratios_7b, _ = obtain_detect_results (model_path)



# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制每条线
plt.plot(k_values, ratios_7b, label='StarCoder-7B', marker='o', linestyle='-', linewidth=2, markersize=8, color='teal')
plt.plot(k_values, ratios_3b, label='StarCoder-3B', marker='s', linestyle='--', linewidth=2, markersize=8, color='orange')
plt.plot(k_values, ratios_1b, label='StarCoder-1B', marker='^', linestyle='-.', linewidth=2, markersize=8, color='purple')

# 设置背景样式
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gca().set_facecolor('#f5f5f5')

# x轴和y轴美化
plt.xticks(
    [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    fontsize=12,
    rotation=0
)
plt.yticks(fontsize=12)
plt.ylim(0.25, 0.6)  # 确保比例值显示完全

# 添加标签和标题
plt.xlabel("Top-k", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
plt.title("Leakage Detection via Ranking Perplexity From Low to High", fontsize=16, fontweight='bold', color='black')

# 添加图例
plt.legend(fontsize=12, loc='upper right')

# 显示数据点的数值
for x, y in zip(k_values, ratios_7b):
    plt.text(x, y + 0.02, f'{y:.2f}', ha='center', fontsize=12, color='teal', alpha=0.8)
for x, y in zip(k_values[0:8], ratios_3b[0:8]):
    plt.text(x, y - 0.05, f'{y:.2f}', ha='center', fontsize=12, color='orange', alpha=0.9)
for x, y in zip(k_values, ratios_1b):
    plt.text(x, y - 0.06, f'{y:.2f}', ha='center', fontsize=12, color='purple', alpha=0.7)

# 显示图表
plt.show()