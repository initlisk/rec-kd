import torch
import torch.nn as nn
from models import NextItNet_Decoder, SASRec
import shutil
import time
import math
import numpy as np
import argparse
from data_preprocess import data_preprocess, get_tensor_data
import os
import random
import torch.nn.functional as F
import utils
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import sys
from tqdm import tqdm, trange
from pyemd import emd_with_flow


def is_better(scores, better_scores):
    if better_scores == None:
        return True
    else:
        cnt = 0
        for key in scores:
            if round(scores[key], 4) >= round(best_scores[key], 4):
                cnt += 1
        
        if cnt > len(scores.keys()) // 2 + 1:
            return True
        else:
            return False

# generate models(stduent model and teacher model)
def get_models(model_para, args):
    if args.network_type == "nextitnet":
        model = NextItNet_Decoder(model_para).to(args.device)
    elif args.network_type == "sasrec":
        model = SASRec(model_para).to(args.device)
    else:
        model = None
    
    teacher_model = None
    if args.is_distilling:

        model_para['dilations'] = model_para['base_block'] * args.teacher_block_num
        model_para['hidden_size'] = args.teacher_hidden_size
        model_para['is_student'] = False
        model_para['block_num'] = args.teacher_block_num

        if args.network_type == "nextitnet":
            teacher_model = NextItNet_Decoder(model_para).to(args.device)
        elif args.network_type == "sasrec":
            teacher_model = SASRec(model_para).to(args.device)
        else:
            teacher_model = None

        T_checkpoint = torch.load(args.teacher_model_path, map_location=args.device)
    
        teacher_model.load_state_dict(T_checkpoint['net'], strict=False)  
        teacher_model.eval()

        model_para['dilations'] = model_para['base_block'] * args.block_num
        model_para['hidden_size'] = args.hidden_size
        model_para['is_student'] = True
        model_para['block_num'] = args.block_num
    
    scores = None
    if not args.train_from_scratch:
        checkpoint = torch.load(args.pretrain_model_path, map_location=args.device)
        model.load_state_dict(checkpoint['net'], strict=False)
        if 'scores' in checkpoint:
            scores = checkpoint['scores']

    return model, scores, teacher_model

def get_new_layer_weight(student_layer_weight, teacher_layer_weight, trans_matrix, distance_matrix, stu_layer_num, tea_layer_num, args, T):

    distance_matrix = distance_matrix.detach().cpu().numpy()
    trans_weight = np.sum(trans_matrix * distance_matrix, -1)

    for i in range(stu_layer_num):
        student_layer_weight[i] = trans_weight[i] / student_layer_weight[i]
    weight_sum = np.sum(student_layer_weight)
    for i in range(stu_layer_num):
        if student_layer_weight[i] != 0:
            student_layer_weight[i] = weight_sum / student_layer_weight[i]

    trans_weight = np.sum(np.transpose(trans_matrix) * distance_matrix, -1)
    for j in range(tea_layer_num):
        teacher_layer_weight[j] = trans_weight[j + stu_layer_num] / teacher_layer_weight[j]
    weight_sum = np.sum(teacher_layer_weight)
    for i in range(tea_layer_num):
        if teacher_layer_weight[i] != 0:
            teacher_layer_weight[i] = weight_sum / teacher_layer_weight[i]
    
    student_layer_weight = utils.softmax(student_layer_weight / T)
    teacher_layer_weight = utils.softmax(teacher_layer_weight / T)

    student_layer_weight = utils.softmax(student_layer_weight)
    teacher_layer_weight = utils.softmax(teacher_layer_weight)
    if args.emd_type == "v4":
        for i in range(tea_layer_num):
            teacher_layer_weight[i] *= 1 + i * 0.1 / tea_layer_num
        for i in range(stu_layer_num):
            student_layer_weight[i] *= 1 + i * 0.1 / stu_layer_num
        student_layer_weight = student_layer_weight / np.sum(student_layer_weight)
        teacher_layer_weight = teacher_layer_weight / np.sum(teacher_layer_weight)
    
    #print (student_layer_weight, '\n', teacher_layer_weight)
 
    return student_layer_weight, teacher_layer_weight

def emd_loss(student_reps, teacher_reps, student_attns, teacher_attns, loss_mse, args, T=1):
    global rep_student_weight, rep_teacher_weight
    global attn_student_weight, attn_teacher_weight
    global v8_weight
    if args.emd_type == "v7":
        sum_student_reps = [torch.norm(t).detach().cpu().numpy().astype('float32') for t in student_reps]
        sum_teacher_reps = [torch.norm(t).detach().cpu().numpy().astype('float32') for t in teacher_reps]
        sum_student_attns = [torch.norm(t).detach().cpu().numpy().astype('float32') for t in student_attns]
        sum_teacher_attns = [torch.norm(t).detach().cpu().numpy().astype('float32') for t in teacher_attns]
        rep_student_weight = sum_student_reps / sum(sum_student_reps)
        rep_teacher_weight = sum_teacher_reps / sum(sum_teacher_reps)
        attn_student_weight = sum_student_attns / sum(sum_student_attns)
        attn_teacher_weight = sum_teacher_attns / sum(sum_teacher_attns)

    def emd_attn_loss(student_attns, teacher_attns, student_layer_weight, teacher_layer_weight,
                        stu_layer_num, tea_layer_num, loss_mse, args):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        total_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([total_num, total_num]).to(args.device)
        if args.emd_type == "v3":
            t_dis_matrix = torch.zeros([total_num, total_num]).to(args.device)
        
        for i in range(stu_layer_num):
            student_attn = student_attns[i]
            for j in range(tea_layer_num):
                teacher_attn = teacher_attns[j]
                if args.emd_type == "v6":
                    tmp_loss = 1 - torch.sum(teacher_attn * student_attn, dim=2)
                    tmp_loss = tmp_loss.mean()
                elif args.emd_type == "v8":
                    # weight = torch.norm(teacher_attn, dim=2, keepdim=True)
                    # weight = weight / torch.sum(weight, dim=1, keepdim=True)
                    tmp_loss = torch.sum(v8_weight * (student_attn - teacher_attn) ** 2, dim=1).mean()
                else:
                    tmp_loss = loss_mse(student_attn, teacher_attn) 
                if args.emd_type == "v3":
                    t_dis_matrix[i][j + stu_layer_num] = t_dis_matrix[j + stu_layer_num][i] = tmp_loss
                    tmp_loss *= (1+abs(i/stu_layer_num - j/tea_layer_num) / 5) 
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss
        
        if args.emd_type == "v10":
            tmp = distance_matrix.detach().cpu().numpy()
            student_layer_weight = tmp.mean(axis = 1)[:stu_layer_num]
            teacher_layer_weight = tmp.mean(axis = 0)[stu_layer_num:]
            student_layer_weight = sum(student_layer_weight) / student_layer_weight
            teacher_layer_weight = sum(teacher_layer_weight) / teacher_layer_weight
            student_layer_weight = utils.softmax(student_layer_weight / 10)
            teacher_layer_weight = utils.softmax(teacher_layer_weight / 10)
            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        
        if args.emd_type == "v3":
            attn_loss = torch.sum(torch.tensor(trans_matrix).to(args.device) * t_dis_matrix)
        else:
            attn_loss = torch.sum(torch.tensor(trans_matrix).to(args.device) * distance_matrix)
        
        return attn_loss, trans_matrix, distance_matrix

    def emd_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                        stu_layer_num, tea_layer_num, loss_mse, args):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        total_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([total_num, total_num]).to(args.device)
        
        for i in range(stu_layer_num):
            student_rep = student_reps[i]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j]
                if args.emd_type == "v6":
                    tmp_loss = 1 - torch.sum(student_rep * teacher_rep, dim=2)
                    tmp_loss = tmp_loss.mean()
                elif args.emd_type == "v8":
                    # weight = torch.norm(teacher_rep, dim=2, keepdim=True)
                    # weight = weight / torch.sum(weight, dim=1, keepdim=True)
                    tmp_loss = torch.sum(v8_weight * (student_rep - teacher_rep) ** 2, dim=1).mean()
                else:
                    tmp_loss = loss_mse(student_rep, teacher_rep) 
                distance_matrix[i][j + stu_layer_num] =  distance_matrix[j + stu_layer_num][i] = tmp_loss 
        
        if args.emd_type == "v10":
            tmp = distance_matrix.detach().cpu().numpy()
            student_layer_weight = tmp.mean(axis = 1)[:stu_layer_num]
            teacher_layer_weight = tmp.mean(axis = 0)[stu_layer_num:]
            student_layer_weight = sum(student_layer_weight) / student_layer_weight
            teacher_layer_weight = sum(teacher_layer_weight) / teacher_layer_weight
            student_layer_weight = utils.softmax(student_layer_weight / 20)
            teacher_layer_weight = utils.softmax(teacher_layer_weight / 20)
            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))

        rep_loss = torch.sum(torch.tensor(trans_matrix).to(args.device) * distance_matrix)

        tmp = distance_matrix.detach().cpu().numpy()

        return rep_loss, trans_matrix, distance_matrix

    if args.emd_type != "v2":
        stu_layer_num = len(student_reps)
        tea_layer_num = len(teacher_reps)

        rep_loss, rep_trans_matrix, rep_distance_matrix = \
            emd_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                            stu_layer_num, tea_layer_num, loss_mse, args)

        if args.update_weight:
            rep_student_weight, rep_teacher_weight = get_new_layer_weight(rep_student_weight, rep_teacher_weight, rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, args,T=T)

        if args.use_attn:
            attn_loss, attn_trans_matrix, attn_distance_matrix = \
            emd_attn_loss(student_attns, teacher_attns, attn_student_weight, attn_teacher_weight,
                            stu_layer_num, tea_layer_num, loss_mse, args)

            if args.update_weight:
                attn_student_weight, attn_teacher_weight = get_new_layer_weight(attn_student_weight, attn_teacher_weight, attn_trans_matrix, attn_distance_matrix, stu_layer_num, tea_layer_num, args,T=T)
    
        else:
            attn_loss = torch.tensor([0])

        loss = rep_loss + attn_loss
    
    else:
        loss = 0.0

        for i in range(args.block_num):
            stu_layer_num = len(student_reps[i])
            tea_layer_num = len(teacher_reps[i])

            rep_loss, rep_trans_matrix, rep_distance_matrix = \
                emd_rep_loss(student_reps[i], teacher_reps[i], rep_student_weight[i], rep_teacher_weight[i],
                                stu_layer_num, tea_layer_num, loss_mse, args)

            # print (1)
            if args.update_weight:
                rep_student_weight[i], rep_teacher_weight[i] = get_new_layer_weight(rep_student_weight[i], rep_teacher_weight[i], rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, args, T=T)

            if args.use_attn:
                attn_loss, attn_trans_matrix, attn_distance_matrix = \
                emd_attn_loss(student_attns[i], teacher_attns[i], attn_student_weight[i], attn_teacher_weight[i],
                                stu_layer_num, tea_layer_num, loss_mse, args)

                # print (1)
                if args.update_weight:
                    attn_student_weight[i], attn_teacher_weight[i] = get_new_layer_weight(attn_student_weight[i], attn_teacher_weight[i], attn_trans_matrix, attn_distance_matrix, stu_layer_num, tea_layer_num, args, T=T)
    
            else:
                attn_loss = torch.tensor([0])

            loss = loss + attn_loss + rep_loss
            # print (1)
        
    return loss

def do_eval(model, eval_dataloader, args, epoch):
    model.eval()

    global best_scores

    global curr_preds_5
    global rec_preds_5
    global ndcg_preds_5
    global curr_preds_20
    global rec_preds_20
    global ndcg_preds_20
    curr_preds_5 = []
    rec_preds_5 = []
    ndcg_preds_5 = []
    curr_preds_20 = []
    rec_preds_20 = []
    ndcg_preds_20 = []

    correct = 0
    total = 0

    logger.info("------------------------eval-----------------------------")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", ascii=True)):
            inputs, targets = batch
            inputs = inputs.to(args.device)

            logits, _ = model(inputs, onecall=True) # [batch_size, item_size] only predict the last position

            logits = logits.cpu()

            accuracy(logits.numpy(), targets.numpy())

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # break

        end = time.time()
       
    acc = 100. * correct / total
    res = 0.0
    scores = {}
    scores['mrr_5'] = sum(curr_preds_5) / float(len(curr_preds_5))
    res += scores['mrr_5']
    scores['mrr_20'] = sum(curr_preds_20) / float(len(curr_preds_20))
    res += scores['mrr_20']
    scores['hit_5'] = sum(rec_preds_5) / float(len(rec_preds_5))
    res += scores['hit_5']
    scores['hit_20'] = sum(rec_preds_20) / float(len(rec_preds_20))
    res += scores['hit_20']
    scores['ndcg_5'] = sum(ndcg_preds_5) / float(len(ndcg_preds_5))
    res += scores['ndcg_5']
    scores['ndcg_20'] = sum(ndcg_preds_20) / float(len(ndcg_preds_20))
    res += scores['ndcg_20']

    logger.info("Time for eval: {} mins".format(round((end - start) / 60)))
    logger.info("Acc(hit@1): %.3f%%" % (acc))
    logger.info("Accuracy mrr_5: {}".format(scores['mrr_5']))
    logger.info("Accuracy mrr_20: {}".format(scores['mrr_20']))
    logger.info("Accuracy hit_5: {}".format(scores['hit_5']))
    logger.info("Accuracy hit_20: {}".format(scores['hit_20']))
    logger.info("Accuracy ndcg_5: {}".format(scores['ndcg_5']))
    logger.info("Accuracy ndcg_20: {}\n".format(scores['ndcg_20']))
    
    state = {
        'net': model.state_dict(),
        'scores': scores,
    }

    if is_better(scores, best_scores):
        best_scores = scores
       
        if args.is_distilling:
            torch.save(state, '%s/best_distill_model_%d_%d.t7' % (args.output_dir, args.block_num*4,\
                args.hidden_size))
        else:
            torch.save(state, '%s/best_scratch_model_%d_%d.t7' % (args.output_dir, args.block_num*4, \
                args.hidden_size))
    
    return res / 6
 
def do_train(model, teacher_model, train_dataloader, eval_dataloader, args, batch_num, mse_criterion, ce_criterion, kd_criterion):
    total_train_time = 0.
    global v8_weight

    for _epoch in trange(int(args.epochs), desc="Epoch"):
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
            logits, hidden_outputs = model(inputs) # [batch_size*seq_len, item_size]
            if args.is_distilling:
                with torch.no_grad():
                    teacher_logits, teacher_hidden_outputs = teacher_model(inputs)
                
                embed_loss = 0.0
                hid_loss = 0.0
                if args.distill_type != "kd" and (args.one_step or not args.pred_distill):
                    if args.distill_type != "pkd":
                        embed_loss = mse_criterion(hidden_outputs[0], teacher_hidden_outputs[0])
                        embed_loss *= args.lambda_embed
    
                    if args.distill_type == "emd":
                        new_student_hidden = []
                        new_student_attn = []
                        new_teacher_hidden = []
                        new_teacher_attn = []

                        for i in range(1, len(hidden_outputs) // 2 + 1):
                            new_student_hidden.append(hidden_outputs[2*i])
                            new_student_attn.append(hidden_outputs[2*i-1])
                        for i in range(1, len(teacher_hidden_outputs) // 2 + 1):
                            new_teacher_hidden.append(teacher_hidden_outputs[2*i])
                            new_teacher_attn.append(teacher_hidden_outputs[2*i-1])
                        
                        if args.emd_type == "v2":
                            times = args.teacher_block_num // args.block_num
                            new_student_hidden_v1 = []
                            new_student_attn_v1 = []
                            new_teacher_hidden_v1 = []
                            new_teacher_attn_v1 = []
                            for i in range(args.block_num):
                                new_student_hidden_v1.append([new_student_hidden[2*i], new_student_hidden[2*i+1]])
                                new_student_attn_v1.append([new_student_attn[2*i], new_student_attn[2*i+1]])
                                ed = (i+1)*2*times if i < args.block_num-1 else len(new_teacher_hidden)
                                t_hid = []
                                t_attn = []
                                for j in range(i*2*times, ed):
                                    t_hid.append(new_teacher_hidden[j])
                                    t_attn.append(new_teacher_attn[j])
                                new_teacher_hidden_v1.append(t_hid)
                                new_teacher_attn_v1.append(t_attn)
                            
                            new_student_hidden = new_student_hidden_v1
                            new_student_attn = new_student_attn_v1
                            new_teacher_hidden = new_teacher_hidden_v1
                            new_teacher_attn = new_teacher_attn_v1
                        elif args.emd_type == "v5":
                            new_student_states = []
                            new_teacher_states = []
                            for i in range(len(new_student_hidden)):
                                new_student_states.append(new_student_attn[i])
                                new_student_states.append(new_student_hidden[i])

                            for i in range(len(new_teacher_hidden)):
                                new_teacher_states.append(new_teacher_attn[i])
                                new_teacher_states.append(new_teacher_hidden[i])
                            new_student_hidden = new_student_states
                            new_student_attn = []
                            new_teacher_hidden = new_teacher_states
                            new_teacher_attn = []
                        elif args.emd_type == "v6":
                            new_student_hidden = [F.normalize(s_hid, p=2, dim=2) for s_hid in new_student_hidden]
                            new_student_attn = [F.normalize(s_attn, p=2, dim=2) for s_attn in new_student_attn]
                            new_teacher_hidden = [F.normalize(t_hid, p=2, dim=2) for t_hid in new_teacher_hidden]
                            new_teacher_attn = [F.normalize(t_attn, p=2, dim=2) for t_attn in new_teacher_attn]
                        elif args.emd_type == "v8":
                            tmp = inputs.detach().cpu().numpy()
                            t_weight = np.random.rand(inputs.size(0), inputs.size(1), 1)
                            for i in range(t_weight.shape[0]):
                                for j in range(t_weight.shape[1]):
                                    t_weight[i][j][0] = item_weight[tmp[i][j]]
                            
                            # print (1)
                            v8_weight = torch.tensor(t_weight, dtype=torch.float32).to(args.device)
                            
                            # print (1)

                            v8_weight = F.softmax(v8_weight/20, dim=1)
                            # v8_weight /= torch.sum(v8_weight, dim=1, keepdim=True)

                        hid_loss = emd_loss(new_student_hidden, new_teacher_hidden, new_student_attn, new_teacher_attn, mse_criterion, args, T=50)
                        if args.emd_type == "v11":
                            hid_loss += mse_criterion(new_student_hidden[-1], new_teacher_hidden[-1]) / len(new_student_hidden)

                        '''
                        new_student_hidden = hidden_outputs[1:]
                        new_teacher_hidden = teacher_hidden_outputs[1:]
                        hid_loss = emd_loss(new_student_hidden, new_teacher_hidden, mse_criterion, args, T=1)
                        '''
                    else:
                        
                        new_student_hidden = hidden_outputs[1:]

                        if args.distill_type == "pkd":
                            new_student_hidden = []
                            for i in range(len(hidden_outputs)):
                                if i > 0 and i % 2 == 0:
                                    new_student_hidden.append(hidden_outputs[i])
                        
                        new_teacher_hidden = []
                        if args.network_type == "nextitnet":
                            for i in range(args.block_num):
                                idx = i*4*(args.teacher_block_num // args.block_num)
                                new_teacher_hidden.append(teacher_hidden_outputs[idx+1])
                                new_teacher_hidden.append(teacher_hidden_outputs[idx+2])
                                new_teacher_hidden.append(teacher_hidden_outputs[idx+3])
                                new_teacher_hidden.append(teacher_hidden_outputs[idx+4])
                        else:
                            for i in range(args.block_num):
                                idx = i*2*(args.teacher_block_num // args.block_num)
                                new_teacher_hidden.append(teacher_hidden_outputs[idx+1])
                                new_teacher_hidden.append(teacher_hidden_outputs[idx+2])

                        if args.distill_type == "pkd":
                            new_teacher_hidden_tmp = []
                            for i in range(len(new_teacher_hidden)):
                                if i % 2 != 0:
                                    new_teacher_hidden_tmp.append(new_teacher_hidden[i])
                            new_teacher_hidden = new_teacher_hidden_tmp
                        '''
                        new_student_hidden = []
                        new_teacher_hidden = []
                        for i in range(1, len(hidden_outputs) // 2 + 1):
                            new_student_hidden.append(hidden_outputs[2*i])
                        for i in range(args.block_num):
                            idx = i*4*(args.teacher_block_num // args.block_num)
                            new_teacher_hidden.append(teacher_hidden_outputs[idx+2])
                            new_teacher_hidden.append(teacher_hidden_outputs[idx+4])
                        '''


                        for s_hid, t_hid in zip(new_student_hidden, new_teacher_hidden):
                            # s_hid = torch.sum(torch.pow(torch.abs(s_hid), 2), dim=-1)
                            # t_hid = torch.sum(torch.pow(torch.abs(t_hid), 2), dim=-1)
                            #if args.distill_type == "pkd":
                            #    s_hid = F.normalize(s_hid, p=2, dim=2)
                            #    t_hid = F.normalize(t_hid, p=2, dim=2)
                            hid_loss += mse_criterion(s_hid, t_hid)
                
                    hid_loss *= args.lambda_hid
              
                cls_loss = 0.0
                if args.distill_type == "kd" or args.one_step or args.pred_distill:
                    kd_loss = kd_criterion(F.log_softmax(logits / args.T, dim=1), \
                        F.softmax(teacher_logits / args.T, dim=1)) * args.T * args.T
                    ce_loss = ce_criterion(logits, targets)
                    cls_loss = args.alpha * kd_loss + (1.0 - args.alpha) * ce_loss
        
                loss = embed_loss + hid_loss + cls_loss
            else:
                loss = ce_criterion(logits, targets)
            
            '''
            L2_loss = 0
            if args.L2 > 0.0:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        L2_loss += torch.norm(param, 2)
                loss += args.L2 * L2_loss
            '''

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
        
        if not args.is_distilling or args.one_step or args.pred_distill or args.distill_type == "kd":
            if _epoch >= args.eval_begin_epochs or _epoch % args.eval_per_epochs == 0:
                do_eval(model, eval_dataloader, args, _epoch+1)
        
        state = {
        'net': model.state_dict(),
        }   
        if args.is_distilling:
            torch.save(state, '%s/distill_ckpt_%d_%d_%d.t7' % (args.output_dir, args.block_num*4, \
                args.hidden_size, _epoch+1))
        else:
            torch.save(state, '%s/scratch_ckpt_%d_%d_%d.t7' % (args.output_dir, args.block_num*4, \
                args.hidden_size, _epoch+1))
        
        if args.shrink_lr:
            lr_scheduler.step()
            # logger.info("learning rate: {}\n".format(lr_scheduler.optimizer.state_dict()['param_groups'][0]['lr']))

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

if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_type', default="sasrec", type=str)

    parser.add_argument('--block_num', default=6, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--T_max', default=30, type=int)
    parser.add_argument('--shrink_lr', default="true", type=str2bool)
    # parser.add_argument('--L2', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    parser.add_argument("--train_from_scratch", default="false", type=str2bool)
    parser.add_argument('--pretrain_model_path', type=str, \
        default='/data1/libo/checkpoint/sas/ml100/pkd/ckpt_24_128_50.t7')

    parser.add_argument('--is_distilling', default="true", type=str2bool, help="distill or not")
    parser.add_argument('--distill_type', type=str, default="pkd", help="distill type")
    parser.add_argument('--emd_type', type=str, default="v1", help="bert emd type")
    parser.add_argument('--update_weight', default="true", type=str2bool)
    parser.add_argument('--pred_distill', default="false", type=str2bool)
    parser.add_argument('--one_step', default="true", type=str2bool)
    parser.add_argument('--use_attn', default="True", type=str2bool)
    parser.add_argument("--teacher_block_num", default=12, type=int)
    parser.add_argument("--teacher_hidden_size", default=256, type=int)
    parser.add_argument("--T", default=1, type=int, help="soft label Temperature")
    parser.add_argument("--lambda_hid", default=0.05, type=float, help="weight for loss between hidden states")
    parser.add_argument("--lambda_embed", default=20, type=float, help="weight for loss between hidden states")
    parser.add_argument("--alpha", default=0.7, type=float)
    parser.add_argument('--teacher_model_path', type=str, \
        default='Data/checkpoint/sas/ml100/best_scratch_model_48_256.t7')

    parser.add_argument('--log_path', type=str, default='sas_ml100_pkd_onestep.log')
    parser.add_argument("--eval_per_epochs", default=5, type=int)
    parser.add_argument("--eval_begin_epochs", default=25, type=int)

    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--data_file', type=str, default='Data/Session/mllatest_update_ls100gr3.csv',
                        help='data path')
    parser.add_argument('--epochs', default=55, type=int)
    parser.add_argument('--output_dir', default='/data1/libo/checkpoint/sas/ml100', type=str)
    parser.add_argument('--eval_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
        
    args = parser.parse_args()

    log_format = '%(asctime)s   %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(args.log_path, mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger = logging.getLogger()

    rep_student_weight = []
    rep_teacher_weight = []
    attn_student_weight = []
    attn_teacher_weight = []

    curr_preds_5 = []
    rec_preds_5 = []
    ndcg_preds_5 = []
    curr_preds_20 = []
    rec_preds_20 = []
    ndcg_preds_20 = []
    best_scores = None

    v8_weight = {}

    if args.is_distilling:
        args.output_dir += "/%s" % (args.distill_type)
        if args.distill_type == "emd":
            args.output_dir += "/%s" % (args.emd_type)
        if args.distill_type != "kd" and args.one_step:
            args.output_dir += "/%s" % ("one_step")


    assert args.network_type in ["nextitnet", "sasrec"]
    
    if args.is_distilling:
        assert args.distill_type in ["tinybert", "emd", "kd", "pkd"] 
    if args.distill_type == "emd":
        assert args.emd_type in ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"]
    if args.emd_type == "v5":
        args.use_attn = False
    if args.emd_type == "v7":
        args.update_weight = False
    if args.emd_type == "v9":
        args.update_weight = False

    logger.info(args)

    # for emb bert
    if args.network_type == "nextitnet":
        if args.emd_type == "v2":
            times = args.teacher_block_num // args.block_num
            for i in range(args.block_num):
                rep_student_weight.append(np.ones(2) / 2)
                attn_student_weight.append(np.ones(2) / 2)
                t_length = (i+1) * times if i < args.block_num-1 else args.teacher_block_num
                t_length -= i * times
                t_length *= 2
                rep_teacher_weight.append(np.ones(t_length) /  t_length)
                attn_teacher_weight.append(np.ones(t_length) / t_length)
        elif args.emd_type == "v5":
            s = args.block_num * 4
            t = args.teacher_block_num * 4
            rep_student_weight = np.ones(s) / s
            rep_teacher_weight = np.ones(t) / t
        elif args.emd_type == "v9":
            s = args.block_num * 2
            t = args.teacher_block_num * 2
            rep_student_weight = [np.exp((i-s) / 40)for i in range(s)]
            rep_student_weight /= sum(rep_student_weight)
            attn_student_weight = [np.exp((i-s) / 40)for i in range(s)]
            attn_student_weight /= sum(attn_student_weight)
            rep_teacher_weight = [np.exp((i-t) / 60)for i in range(t)]
            rep_teacher_weight /= sum(rep_teacher_weight)
            attn_teacher_weight = [np.exp((i-t) / 60)for i in range(t)]
            attn_teacher_weight /= sum(attn_teacher_weight)
        else:
            s = args.block_num * 2
            t = args.teacher_block_num * 2
            rep_student_weight = np.ones(s) / s
            rep_teacher_weight = np.ones(t) / t
            attn_student_weight = np.ones(s) / s
            attn_teacher_weight = np.ones(t) / t
    else:
        s = args.block_num
        t = args.teacher_block_num
        rep_student_weight = np.ones(s) / s
        rep_teacher_weight = np.ones(t) / t
        attn_student_weight = np.ones(s) / s
        attn_teacher_weight = np.ones(t) / t



    # ============ DATA PREPROCESSING ==================
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


    # =========== MODEL ================================
    model_para = {
        'item_size': len(item_vocab),
        'hidden_size': args.hidden_size,
        'block_num': args.block_num,
        'seq_len': len(all_examples[0]),
        'rezero': False,
        'dropout': args.dropout,

        # nexitnet
        'base_block': [1, 4],
        'dilations': [1, 4] * args.block_num,
        'kernel_size': 3,
        
        # sasrec
        'num_head': args.num_head,
        
        'fit_size': args.teacher_hidden_size if args.is_distilling else -1,
        'is_student': True if args.is_distilling else False,
        'device': args.device
    }

    ce_criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.KLDivLoss(reduction="batchmean")
    mse_criterion = nn.MSELoss()
    loss_cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    logger.info("shape: {} ".format(np.shape(all_examples)))
    logger.info("vocab size: {}".format(model_para['item_size']))
    logger.info("dilations: {}".format(args.block_num*model_para['base_block']))
    logger.info("hidden_size: {}".format(args.hidden_size))
    if args.is_distilling:
        logger.info("teacher's dilations: {}".format(args.teacher_block_num*model_para['base_block']))
        logger.info("teacher's hidden_size: {}".format(args.teacher_hidden_size))

    model, best_scores, teacher_model = get_models(model_para, args)
    t = model.state_dict()
    decay_p = []
    others_p = []

    if args.weight_decay > 0.0:
        for name, p in model.named_parameters():
            if 'bias' in name:
                others_p += [p]
            else:
                decay_p += [p]
        params = [{'params': decay_p, 'weight_decay':args.weight_decay},\
                        {'params': others_p, 'weight_decay':0}]

        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer =  torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    
    if args.shrink_lr == True:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=1e-5)

    do_train(model, teacher_model, train_dataloader, eval_dataloader, args, batch_num, mse_criterion, ce_criterion, kd_criterion)
    # do_eval(model, eval_dataloader, args, agrs.epoch)
