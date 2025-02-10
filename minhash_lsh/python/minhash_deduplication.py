#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations
import json
import gc
import hashlib
import logging
import multiprocessing as mp
import os
import random
import re
import struct
import time
import warnings
from collections import defaultdict
from itertools import tee
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
import argparse
from datasets import load_dataset, concatenate_datasets
import numpy as np
from joblib import Parallel, delayed


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import datasets
    import numpy as np
    import typer
    from datasets import load_dataset
    from scipy.integrate import quad as integrate
    from tqdm import tqdm


SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
datasets.logging.set_verbosity_error()


def ngrams(sequence: List[str], n: int, min_ngram_size: int) -> Iterable:
    """
    Directly taken from nltk package to avoid dependency.

    Parameters
    ----------
    sequence : list
        The sequence of items to be n-grammed.
    n : int
        The order of the n-grams to be extracted.
    min_ngram_size : int
        The minimum size of n-grams.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.
    """
    if len(sequence) < min_ngram_size:
        return []
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int = 5,
) -> Dict[str, Any]:
    """
    Combined with some datasketch code to better parallelize computation.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    min_ngram_size : int
        The minimum size of n-grams.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.
    """
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)}
    hv = np.array([sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64)  # noqa: E501
    a, b = permutations
    phv = np.bitwise_and(((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH)  # noqa: E501
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.
    """

    def false_positive_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)

def jaccard_similarity(code1: str, code2: str) -> float:
    """
    Calculate the jaccard similarity between two code snippets.

    Parameters
    ----------
    code1 : str
        The first code snippet.
    code2 : str
        The second code snippet.

    Returns
    -------
    float
        The jaccard similarity between the two code snippets.

    Examples
    --------
    # >>> jaccard_similarity("a = 1", "a = 2")
    # 0.3333333333333333
    # >>> jaccard_similarity("a = 1", "a = 1")
    # 1.0
    """
    tokens1 = set([t for t in NON_ALPHA.split(code1) if t.strip()])
    tokens2 = set([t for t in NON_ALPHA.split(code2) if t.strip()])
    return len(tokens1 & tokens2) / max(1, len(tokens1 | tokens2))

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xin1997/bugsinpy_all_only_input", type=str, help="Name or path of the HF dataset to decontaminate")
    parser.add_argument("--pretrain_dataset_id", default="00001", type=str, help="Name or path of the HF dataset to decontaminate")
    parser.add_argument("--config", default="default", type=str,help="Dataset config")
    parser.add_argument("--data_dir", default=None, type=str, help="Dataset data directory")
    parser.add_argument("--split", default="train", type=str, help="Dataset split")
    parser.add_argument("--revision", default="main", type=str, help="Dataset revision")
    parser.add_argument("--column", default="content", type=str, help="Dataset column")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="Cache directory")
    parser.add_argument("--ngram_size", type=int, default=5, help="Number of processes")
    parser.add_argument("--num_perm", default=256, type=int, help="Number of permutations")
    parser.add_argument("--min_ngram_size", type=int, default=5, help="Shorter documents will be removed")
    parser.add_argument("--threshold", default=0.7, type=float, help="Minhash threshold")
    parser.add_argument("--output", default='out_near_dup', type=str, help="Store the deduplicated dataset")
    parser.add_argument("--num_proc", type=int, default=100, help="Number of processes")
    return parser.parse_args()


if __name__ == "__main__":

    def run(dataset, pretrain_dataset_id, config, split, data_dir, revision, column, cache_dir, ngram_size, num_perm, threshold, min_ngram_size, output, num_proc):
        global uf
        OUTPUT_BASE = Path(output or "output")
        OUTPUT_BASE.mkdir(exist_ok=True, parents=True)
        output = OUTPUT_BASE / "deduplicated"

        logging.basicConfig(level=logging.INFO)

        time_measures = {}
        start_time = time.time()

        B, R = optimal_param(threshold, num_perm)
        HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
        HASH_TABLES = [defaultdict(set) for _ in range(B)]

        time_measures["load_dataset"] = time.time()

        ### Load pretraining dataset from name
        ds = load_dataset(
            dataset,
            config,
            data_dir=data_dir,
            split=split,
            use_auth_token=True,
            # cache_dir=cache_dir,
            revision=revision,
            num_proc=num_proc,
        )


        pretrain_ds = load_dataset(
            "parquet",
            data_files="../../1.starcoder_data/python_data/train-"+pretrain_dataset_id+"-of-00059.parquet",
            # Use data_files to specify the local file(s)
            split="train",  # Specify the split; can be "train", "test", etc.
            num_proc=num_proc,  # Parallelize processing if necessary
            keep_in_memory=True
        )


        print()

        def add_pretraining_dagta_prefix_to_id(example):
            # Prefix 'data_' to the 'id' field
            example['id'] = f"pretrain_python_data_{example['id']}"
            return example

        def int2str(example):
            # Prefix 'data_' to the 'id' field
            example['id'] = f"test_{example['id']}"
            return example

        # add prefix on ids of pretraining data
        pretrain_ds = pretrain_ds.map(add_pretraining_dagta_prefix_to_id, num_proc=num_proc, desc="add prefix on ids...",)
        ds = ds.map(int2str, num_proc=num_proc, desc="turn int ids into strings's ids...",)


        columns1 = set(ds.column_names)
        columns2 = set(pretrain_ds.column_names)
        missing_columns = columns2 - columns1  # ds 缺少的列
        # 给 ds 增加缺失的列，并填充 None
        for col in missing_columns:
            if col != 'max_stars_count':
                ds = ds.add_column(col, ['NA'] * len(ds))
            else:
                ds = ds.add_column(col, [0] * len(ds))

        # pretrain_ds_slice = pretrain_ds.select(range(1000))

        # ds = concatenate_datasets([ds, pretrain_ds_slice])

        ds = concatenate_datasets([ds, pretrain_ds])







        time_measures["load_dataset"] = time.time() - time_measures["load_dataset"]
        DATA_SIZE = len(ds)
        PERMUTATIONS = np.array(
            [
                (
                    RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

        time_measures["minhash"] = time.time()
        embedded = ds.map(
            function=embed_func,
            fn_kwargs={
                "num_perm": num_perm,
                "hashranges": HASH_RANGES,
                "ngram_size": ngram_size,
                "permutations": PERMUTATIONS,
                "min_ngram_size": min_ngram_size,
            },
            input_columns=[column],
            remove_columns=ds.column_names,
            num_proc=num_proc,
            with_indices=True,
            desc="Fingerprinting...",
        )
        time_measures["minhash"] = time.time() - time_measures["minhash"]

        time_measures["clustering"] = time.time()
        batch_size: int = 10000
        for i in tqdm(
            range(0, len(embedded), batch_size), dynamic_ncols=True, desc="Iterating MinHashes..."  # noqa: E501
        ):
            batch = embedded[i : i + batch_size]
            for key, Hs in zip(batch["__id__"], batch["__signatures__"]):
                for H, hashtable in zip(Hs, HASH_TABLES):
                    ## those data with the same segment in signatures are put into the same bucket by 'add(key)' where key is the id of the data
                    hashtable[H].add(key)

        for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
            for cluster in table.values():
                if len(cluster) <= 1:
                    continue
                idx = min(cluster)
                for x in cluster:
                    uf.union(x, idx) ## use the Union-Find Set Strcuture to build the tree-like near-duplcation relations between all samples; a unique sample will be a single tree; near duplicated samples will be in a tree.


        time_measures["clustering"] = time.time() - time_measures["clustering"]

        time_measures["filtering"] = time.time()
        gc.freeze()
        gc.disable()
        ds = ds.map(
            function=lambda _, idx: {"__cluster__": uf.find(idx)}, ## use the smallest id in the tree as the name for the cluster, and all leafs of the tree is the memebers of the duplicated cluster
            with_indices=True,
            num_proc=num_proc,
            new_fingerprint=str(random.getrandbits(128)),
            desc="Finding clusters...",
        )
        gc.enable()
        gc.collect()

        def generate_parent_child_list(uf):
            # Step 1: Find all root nodes
            root_to_children = {}
            for node in uf.parent.keys():
                root = uf.find(node)
                if root not in root_to_children:
                    root_to_children[root] = []
                root_to_children[root].append(node)

            # Step 2: Convert the mapping to the desired list format
            parent_child_list = []
            for root, children in root_to_children.items():
                if len(children) > 1:
                    parent_child_list.append(children)

            return parent_child_list

        ## obtain a list of cluster, where each cluster has the duplciated sample id
        parent_child_list = generate_parent_child_list(uf)

        print('here')

        ## save duplicated pair and check the jaccard similarity
        def find_duplicates_in_cluster(cluster, ds, threshold):
            duplicated = []
            neighbors = [ds[cluster_id] for cluster_id in cluster]

            # Convert to NumPy array for faster access (optional)
            neighbors = np.array(neighbors)

            # Use a set to track checked pairs by their indices or unique IDs
            seen_pairs = set()

            # Iterate over pairs, but only consider one order
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):  # Avoid duplicate comparisons
                    curr_ = neighbors[i]
                    reference = neighbors[j]

                    # Use a tuple of unique identifiers (e.g., indices) to track pairs
                    pair = (i, j)

                    if pair not in seen_pairs:
                        if curr_['id'].split('_')[0] != reference['id'].split('_')[0] and 'pretrain' in [
                            curr_['id'].split('_')[0], reference['id'].split('_')[0]]:
                            reference_text = reference["content"]
                            curr_text = curr_["content"]
                            if jaccard_similarity(curr_text, reference_text) >= threshold:
                                duplicated.append((curr_, reference))
                                seen_pairs.add(pair)
                                seen_pairs.add((j, i))  # Add both orders to avoid future checks

            return duplicated

        # Parallel processing of clusters
        duplicated_data = Parallel(n_jobs=100)(delayed(find_duplicates_in_cluster)(cluster, ds, args.threshold) for cluster in tqdm(parent_child_list, desc="Checking for false positives..."))
        print('duplicate pair:', len(duplicated_data))

        # Save the list to a JSON file
        with open(args.output+"/duplicated.json", "w") as json_file:
            json.dump(duplicated_data, json_file, indent=4)


    args = arguments()
    all_pretrain_dataset_ids = ['0000' + str(i) for i in range(10)] #+ ['000' + str(i) for i in range(10, 59)]

    for args.pretrain_dataset_id in all_pretrain_dataset_ids:
        print("=====================")
        print("==== Benchmark:", args.dataset)
        print("==== Pretraining Subset:", args.pretrain_dataset_id)

        output_folder_name = './'+args.dataset.split('/')[-1].split('_')[0]+'/out_near_dup_python_'+args.pretrain_dataset_id
        args.output = output_folder_name

        print(output_folder_name)
        mp.set_start_method("fork", force=True)
        uf = UnionFind()
        run(args.dataset, args.pretrain_dataset_id,  args.config, args.split, args.data_dir, args.revision, args.column, args.cache_dir, args.ngram_size, args.num_perm, args.threshold, args.min_ngram_size, args.output, args.num_proc)

        import subprocess
        result = subprocess.run(['rm', '-r', '/mnt/hdd1/xinzhou/Huggingface_CACHE/datasets'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.run(['mkdir', '/mnt/hdd1/xinzhou/Huggingface_CACHE/datasets'], stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
        print("=====================")
