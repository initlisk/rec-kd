from time import sleep
from models.SASRec import SASRec
from models.NextItNet import NextItNet
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pyemd import emd_with_flow
import utils

class SRS_KD(nn.Module):
    
    def __init__(self, config):
        super(SRS_KD, self).__init__()
    
        if config.srs.lower() == 'nextitnet':
            SRS_Model = NextItNet
            s = config.student_config.block_num * 2
            t = s * 2
        elif config.srs.lower() == 'sasrec':
            SRS_Model = SASRec
            s = config.student_config.block_num
            t = s * 2

        self.rep_student_weight = np.ones(s) / s
        self.rep_teacher_weight = np.ones(t) / t
        self.attn_student_weight = np.ones(s) / s
        self.attn_teacher_weight = np.ones(t) / t

        self.teacher_model = SRS_Model(config.teacher_config)
        self.student_model = SRS_Model(config.student_config)
        T_checkpoint = torch.load(config.teacher_path, map_location=config.device)
        self.teacher_model.load_state_dict(T_checkpoint['net'], strict=False)  
        self.teacher_model.eval()

        self.device = config.device

        self.update_weight = config.update_weight
        self.use_attn = config.use_attn

        self.transform_block = nn.Linear(config.student_config.hidden_size, config.teacher_config.hidden_size)

        self.ce_criterion = nn.CrossEntropyLoss()
        self.kd_criterion = nn.KLDivLoss(reduction="batchmean")
        self.mse_criterion = nn.MSELoss()

        if config.kd_method == "vanilla_kd":
            self.loss_func = self.vanilla_kd_loss
        elif config.kd_method == "bertemd":
            self.loss_func = self.bertemd_loss

    def forward(self, x, labels, onecall=False): # inputs: [batch_size, seq_len]
        with torch.no_grad():
            teacher_logits, teacher_hiddens = self.teacher_model(x)
        student_logits, student_hiddens = self.student_model(x, onecall)
        for i in range(len(student_hiddens)):
            student_hiddens[i] = self.transform_block(student_hiddens[i])

        if onecall == True:
            loss = None
        else:
            loss = self.loss_func(student_logits, teacher_logits, labels, student_hiddens, teacher_hiddens)

        return student_logits, loss
    
    def vanilla_kd_loss(self, student_logits, teacher_logits, labels, student_hiddens, teacher_hiddens):
        T = 1
        alpha = 0.7 
        kd_loss = self.kd_criterion(F.log_softmax(student_logits / T, dim=1), \
            F.softmax(teacher_logits / T, dim=1)) * T * T
        ce_loss = self.ce_criterion(student_logits, labels)
        
        return alpha * kd_loss + (1.0 - alpha) * ce_loss

    def emd_loss(self, student_reps, teacher_reps, student_attns, teacher_attns, T=1):
        def emd_attn_loss(student_attns, teacher_attns, student_layer_weight, teacher_layer_weight,
                            stu_layer_num, tea_layer_num, loss_func, device):
            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
            total_num = stu_layer_num + tea_layer_num
            distance_matrix = torch.zeros([total_num, total_num]).to(device)
            
            for i in range(stu_layer_num):
                student_attn = student_attns[i]
                for j in range(tea_layer_num):
                    teacher_attn = teacher_attns[j]
                    tmp_loss = loss_func(student_attn, teacher_attn) 
                    distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

            _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                            distance_matrix.detach().cpu().numpy().astype('float64'))
            
            attn_loss = torch.sum(torch.tensor(trans_matrix).to(device) * distance_matrix)

            # tmp = distance_matrix.detach().cpu().numpy()
            
            return attn_loss, trans_matrix, distance_matrix

        def emd_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                            stu_layer_num, tea_layer_num, loss_func,device):
            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
            total_num = stu_layer_num + tea_layer_num
            distance_matrix = torch.zeros([total_num, total_num]).to(device)
            
            for i in range(stu_layer_num):
                student_rep = student_reps[i]
                for j in range(tea_layer_num):
                    teacher_rep = teacher_reps[j]
                    tmp_loss = loss_func(student_rep, teacher_rep) 
                    distance_matrix[i][j + stu_layer_num] =  distance_matrix[j + stu_layer_num][i] = tmp_loss 
                
            _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                            distance_matrix.detach().cpu().numpy().astype('float64'))

            rep_loss = torch.sum(torch.tensor(trans_matrix).to(device) * distance_matrix)

            # tmp = distance_matrix.detach().cpu().numpy()

            return rep_loss, trans_matrix, distance_matrix

        def get_new_layer_weight(student_layer_weight, teacher_layer_weight, trans_matrix, distance_matrix, stu_layer_num, tea_layer_num, T):

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
          
            return student_layer_weight, teacher_layer_weight

        stu_layer_num = len(student_reps)
        tea_layer_num = len(teacher_reps)

        rep_loss, rep_trans_matrix, rep_distance_matrix = \
            emd_rep_loss(student_reps, teacher_reps, self.rep_student_weight, self.rep_teacher_weight,
                            stu_layer_num, tea_layer_num, self.mse_criterion, self.device)

        if self.update_weight:
            self.rep_student_weight, self.rep_teacher_weight = get_new_layer_weight(self.rep_student_weight, self.rep_teacher_weight, rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num,T)

        if self.use_attn:
            attn_loss, attn_trans_matrix, attn_distance_matrix = \
            emd_attn_loss(student_attns, teacher_attns, self.attn_student_weight, self.attn_teacher_weight,
                            stu_layer_num, tea_layer_num, self.mse_criterion, self.device)

            if self.update_weight:
                self.attn_student_weight, self.attn_teacher_weight = get_new_layer_weight(self.attn_student_weight, self.attn_teacher_weight, attn_trans_matrix, attn_distance_matrix, stu_layer_num, tea_layer_num,T)
    
        else:
            attn_loss = torch.tensor([0])
            
        return rep_loss + attn_loss
        

    def bertemd_loss(self, student_logits, teacher_logits, labels, student_hiddens, teacher_hiddens):
        kd_loss = self.vanilla_kd_loss(student_logits, teacher_logits, labels, student_hiddens, teacher_hiddens)
        loss = self.mse_criterion(student_hiddens[0], teacher_hiddens[0])
        
        new_student_reps = []
        new_student_attns = []
        new_teacher_reps = []
        new_teacher_attns = []

        for i in range(1, len(student_hiddens) // 2 + 1):
            new_student_reps.append(student_hiddens[2*i])
            new_student_attns.append(student_hiddens[2*i-1])

        for i in range(1, len(teacher_hiddens) // 2 + 1):
            new_teacher_reps.append(teacher_hiddens[2*i])
            new_teacher_attns.append(teacher_hiddens[2*i-1])

        loss += self.emd_loss(new_student_reps, new_teacher_reps, new_student_attns, new_teacher_attns, T=30)

        return 1e-2 * loss + kd_loss