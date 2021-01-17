import os
import numpy as np
import time
import pandas as pd
import torch
from torch.utils.data.dataset import TensorDataset
import collections

def get_dataloader(data_file):
    all_examples, item_vocab, item2id, id2item, item_weight, max_length = data_preprocess(args.data_file)
    # Split train/test set
    eval_examples_index = -1 * int(args.eval_percentage * float(len(all_examples)))
    train_examples, eval_examples = all_examples[:eval_examples_index], all_examples[eval_examples_index:]
    batch_num = len(train_examples) // args.batch_size + 1

    train_data = get_tensor_data(train_examples, "train")
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    eval_data = get_tensor_data(eval_examples, "eval")
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    return train_dataloader, eval_dataloader


def data_preprocess(data_file):
    pad = "<PAD>"
    examples = open(data_file, "r").readlines()
    examples = [s for s in examples]
    max_length = max([len(x.strip().split(",")) for x in examples])
    item_freq = {pad: 0}

    for _example in examples:
        items_list = _example.strip().split(",")
        for item in items_list:
            if item in item_freq.keys(): 
                item_freq[item] += 1
            else:
                item_freq[item] = 1
        item_freq[pad] += max_length - len(items_list)
    
    count_pairs = sorted(item_freq.items(), key=lambda x: (-x[1], x[0]))
    item_vocab, _ = list(zip(*count_pairs))
    item2id = dict(zip(item_vocab, range(len(item_vocab))))
    id2item = {value:key for key, value in item2id.items()}

    # item_freq = {item2id[key]:value for key, value in item_freq.items()}
    pad_id = item2id[pad]
    examples2id = []
    
    s = set()
    item_freq = collections.defaultdict(lambda : 0)
    for _example in examples: 
        _example2id = []
        s.clear()
        for item in _example.strip().split(','):
            t = item2id[item]
            _example2id.append(t)
            s.add(t)
        _example2id = ([pad_id] * (max_length - len(_example2id))) + _example2id
        for _id in s:
            item_freq[_id] += 1

        examples2id.append(_example2id)
        
    examples = np.array(examples2id)
    t = len(examples2id)
    min_val = 10000000
    for _key in item_freq:
        t = np.log(item_freq[_key])
        if t < min_val:
            min_val = t
        item_freq[_key] = t

    # print (item_freq[pad_id])
      
    item_freq[pad_id] = min_val

    return examples, item_vocab, item2id, id2item, item_freq, max_length

def get_tensor_data(examples, data_type):
    assert data_type in ["train", "eval"]
    
    all_input_ids = torch.tensor(examples[:, :-1], dtype=torch.long)

    if data_type == "train":
        all_target_ids = torch.tensor(examples[:, 1:], dtype=torch.long)
    else:
        all_target_ids = torch.tensor(examples[:, -1], dtype=torch.long)
    
    tensor_data = TensorDataset(all_input_ids, all_target_ids)

    return tensor_data