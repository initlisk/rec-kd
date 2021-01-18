import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.dataset import TensorDataset
import collections
import torch

def train(model, config,  train_dataloader, eval_dataloader):
    total_train_time = 0.

    loss_func = get_loss()

    for _epoch in trange(int(config['epochs']), desc="Epoch"):
        model.train()
        train_loss = 0
        correct = 0
        total = 0 

        logger.info("------------------------train-----------------------------")
        start = time.time()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            inputs, targets = batch
            inputs = inputs.to(args.device)
            targets = targets.reshape(-1).to(args.device)
            optimizer.zero_grad()
            
            loss = model(inputs)

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
        

        if _epoch >= args.eval_begin_epochs or _epoch % args.eval_per_epochs == 0:
            do_eval(model, eval_dataloader, args, _epoch+1)

        if args.shrink_lr:
            lr_scheduler.step()


def accuracy(output, target, topk=(5, 20)): # output: [batch_size, item_size] target: [batch_size]
    """Computes the accuracy over the k top predictions for the specified values of k"""
    global curr_preds_5
    global rec_preds_5
    global ndcg_preds_5
    global curr_preds_20
    global rec_preds_20
    global ndcg_preds_20

    for bi in range(output.shape[0]):
        pred_items_5 = utils.sample_top_k(output[bi], top_k=topk[0])  # top_k=5
        pred_items_20 = utils.sample_top_k(output[bi], top_k=topk[1])

        true_item=target[bi]
        predictmap_5={ch : i for i, ch in enumerate(pred_items_5)}
        pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

        rank_5 = predictmap_5.get(true_item)
        rank_20 = pred_items_20.get(true_item)
        if rank_5 == None:
            curr_preds_5.append(0.0)
            rec_preds_5.append(0.0)
            ndcg_preds_5.append(0.0)
        else:
            MRR_5 = 1.0/(rank_5+1)
            Rec_5 = 1.0#3
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
            curr_preds_5.append(MRR_5)
            rec_preds_5.append(Rec_5)#4
            ndcg_preds_5.append(ndcg_5)  # 4
        if rank_20 == None:
            curr_preds_20.append(0.0)
            rec_preds_20.append(0.0)#2
            ndcg_preds_20.append(0.0)#2
        else:
            MRR_20 = 1.0/(rank_20+1)
            Rec_20 = 1.0#3
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
            curr_preds_20.append(MRR_20)
            rec_preds_20.append(Rec_20)#4
            ndcg_preds_20.append(ndcg_20)  # 4

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