import torch.nn as nn
import time
import math
import numpy as np
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import sys
from tqdm import tqdm, trange
from pyemd import emd_with_flow
import collections
import torch

def train(model, config,  train_dataloader, eval_dataloader):

    log_format = '%(asctime)s   %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(config.log_path, mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger = logging.getLogger()

    config.log(logger)

    others_p = []
    decay_p = []
    for name, p in model.named_parameters():
        if 'bias' in name:
            others_p += [p]
        else:
            decay_p += [p]
    params = [{'params': decay_p, 'weight_decay':config.reg},\
                    {'params': others_p, 'weight_decay':0}]
    optimizer = torch.optim.Adam(params, lr=config.lr)

    batch_num = len(train_dataloader)

    total_train_time = 0.0

    best_scores = None

    no_imporve_epoch = 0

    if config.kd_method == "scratch":
        loss_func = nn.CrossEntropyLoss()
    
    for _epoch in trange(int(config.max_epoch), desc="Epoch"):
        model.train()
        train_loss = 0
        correct = 0
        total = 0 

        logger.info("------------------------train-----------------------------")
        start = time.time()
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            inputs, targets = batch
            inputs = inputs.to(config.device)
            targets = targets.reshape(-1).to(config.device)
            optimizer.zero_grad()
            
            if config.kd_method == "scratch":
                logits, _ = model(inputs)
                loss = loss_func(logits, targets)
            else:
                loss, logits = model(inputs) 

            loss.backward()
        
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx==0 or (batch_idx+1) % max(10, batch_num//10)  == 0:
                logger.info("epoch: {}\t {}/{}".format(_epoch+1, batch_idx+1, batch_num))
                logger.info('Loss: %.3f | Acc(hit@1): %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
             
            # break
                    

        end = time.time()
        total_train_time += end - start
        logger.info("Time for {}'th epoch: {} mins, time for {} epoches: {} hours".\
            format(_epoch+1, round((end - start) / 60, 2), _epoch+1, round(total_train_time / 3600, 2)))
        

        if _epoch >= config.eval_begin_epochs or _epoch % config.eval_per_epochs == 0:
            scores = do_eval(model, config, eval_dataloader,logger, _epoch+1)
            no_imporve_epoch += 1
            if is_better(scores, best_scores):
                state = {
                    'net': model.state_dict(),
                    'scores': scores,
                }
                best_scores = scores
                torch.save(state, config.save_path)
                no_imporve_epoch = 0
            if no_imporve_epoch > config.early_stop:
                return
    

def do_eval(model, config, eval_dataloader, logger, epoch):
    model.eval()

    scores_list = {}
    scores_list['curr_preds_5'] = []
    scores_list['rec_preds_5'] = []
    scores_list['ndcg_preds_5'] = []
    scores_list['curr_preds_20'] = []
    scores_list['rec_preds_20'] = []
    scores_list['ndcg_preds_20'] = []

    correct = 0
    total = 0

    logger.info("------------------------eval-----------------------------")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", ascii=True)):
            inputs, targets = batch
            inputs = inputs.to(config.device)

            logits, _ = model(inputs, onecall=True) # [batch_size, item_size] only predict the last position

            logits = logits.cpu()

            accuracy(logits.numpy(), targets.numpy(), scores_list)

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # break

        end = time.time()
       
    acc = 100. * correct / total
    scores = {}
    scores['mrr_5'] = sum(scores_list['curr_preds_5']) / float(len(scores_list['curr_preds_5']))
    scores['mrr_20'] = sum(scores_list['curr_preds_20']) / float(len(scores_list['curr_preds_20']))
    scores['hit_5'] = sum(scores_list['rec_preds_5']) / float(len(scores_list['rec_preds_5']))
    scores['hit_20'] = sum(scores_list['rec_preds_20']) / float(len(scores_list['rec_preds_20']))
    scores['ndcg_5'] = sum(scores_list['ndcg_preds_5']) / float(len(scores_list['ndcg_preds_5']))
    scores['ndcg_20'] = sum(scores_list['ndcg_preds_20']) / float(len(scores_list['ndcg_preds_20']))

    logger.info("Time for eval: {} mins".format(round((end - start) / 60)))
    logger.info("Acc(hit@1): %.3f%%" % (acc))
    logger.info("Accuracy mrr_5: {}".format(scores['mrr_5']))
    logger.info("Accuracy mrr_20: {}".format(scores['mrr_20']))
    logger.info("Accuracy hit_5: {}".format(scores['hit_5']))
    logger.info("Accuracy hit_20: {}".format(scores['hit_20']))
    logger.info("Accuracy ndcg_5: {}".format(scores['ndcg_5']))
    logger.info("Accuracy ndcg_20: {}\n".format(scores['ndcg_20']))
    
    return  scores

def sample_top_k(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    return idx

def is_better(scores, best_scores):
    if best_scores == None:
        return True
    else:
        cnt = 0

        total_cur = 0.0
        total_best = 0.0

        for key in scores:
            total_best += round(best_scores[key], 4)
            total_cur += round(scores[key], 4)
            if round(scores[key], 4) >= round(best_scores[key], 4):
                cnt += 1
        
        if cnt > len(scores.keys()) // 2 + 1:
            return True
        elif cnt == len(scores.keys) // 2 and total_cur > total_best:
            return True
        else:
            return False

def accuracy(output, target, scores_list, topk=(5, 20)): # output: [batch_size, item_size] target: [batch_size]
    """Computes the accuracy over the k top predictions for the specified values of k"""
    for bi in range(output.shape[0]):
        pred_items_5 = sample_top_k(output[bi], top_k=topk[0])  # top_k=5
        pred_items_20 = sample_top_k(output[bi], top_k=topk[1])

        true_item=target[bi]
        predictmap_5={ch : i for i, ch in enumerate(pred_items_5)}
        pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

        rank_5 = predictmap_5.get(true_item)
        rank_20 = pred_items_20.get(true_item)
        if rank_5 == None:
            scores_list['curr_preds_5'].append(0.0)
            scores_list['rec_preds_5'].append(0.0)
            scores_list['ndcg_preds_5'].append(0.0)
        else:
            MRR_5 = 1.0/(rank_5+1)
            Rec_5 = 1.0#3
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
            scores_list['curr_preds_5'].append(MRR_5)
            scores_list['rec_preds_5'].append(Rec_5)#4
            scores_list['ndcg_preds_5'].append(ndcg_5)  # 4

        if rank_20 == None:
            scores_list['curr_preds_20'].append(0.0)
            scores_list['rec_preds_20'].append(0.0)#2
            scores_list['ndcg_preds_20'].append(0.0)#2
        else:
            MRR_20 = 1.0/(rank_20+1)
            Rec_20 = 1.0#3
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
            scores_list['curr_preds_20'].append(MRR_20)
            scores_list['rec_preds_20'].append(Rec_20)#4
            scores_list['ndcg_preds_20'].append(ndcg_20)  # 4

def get_dataloader(config):
    pad = "<PAD>"
    examples = open(config.dataset_path, "r").readlines()
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


      
    item_freq[pad_id] = min_val

    eval_examples_index = -1 * int(config.eval_percentage * float(len(examples)))
    train_examples, eval_examples = examples[:eval_examples_index], examples[eval_examples_index:]

    train_data = get_tensor_data(train_examples, "train")
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)

    eval_data = get_tensor_data(eval_examples, "eval")
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=config.batch_size)

    return train_dataloader, eval_dataloader, len(item_vocab)

def get_tensor_data(examples, data_type):
    assert data_type in ["train", "eval"]
    
    all_input_ids = torch.tensor(examples[:, :-1], dtype=torch.long)

    if data_type == "train":
        all_target_ids = torch.tensor(examples[:, 1:], dtype=torch.long)
    else:
        all_target_ids = torch.tensor(examples[:, -1], dtype=torch.long)
    
    tensor_data = TensorDataset(all_input_ids, all_target_ids)

    return tensor_data