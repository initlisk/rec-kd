import torch
from torch import nn

class SRS_KD(nn.Module):
    
    def __init__(self, config):
        super(SRS_KD, self).__init__()
    
        if config['model_type'].lower() == 'nextitnet':
            import NextItNet.NextItNet as SRS_Model

        elif config['model_type'].lower() == 'sasrec':
            import SASRec.SASRec as SRS_Model

        else:
            return

        self.teacher_model = SRS_Model(config['teacher_config'])
        self.student_model = SRS_Model(config['student_config'])
        T_checkpoint = torch.load(config['teacher_model_path'], map_location=config['device'])
        self.teacher_model.load_state_dict(T_checkpoint['net'], strict=False)  
        self.teacher_model.eval()

        self.transform_block = nn.Linear(config['student_config']['hidden_size'], config['teacher_config']['hidden_size'])

        self.loss_func = nn.MSELoss()
    
    def forward(self, x, label): # inputs: [batch_size, seq_len]
        with torch.no_grad():
            teacher_output = self.teacher_model(x)
        sttudent_output = self.student_model(x)
        sttudent_output[]

        embed_loss = mse_criterion(hidden_outputs[0], teacher_hidden_outputs[0])
        embed_loss *= args.lambda_embed

        kd_loss = kd_criterion(F.log_softmax(logits / args.T, dim=1), \
                        F.softmax(teacher_logits / args.T, dim=1)) * args.T * args.T
                    ce_loss = ce_criterion(logits, targets)
                    cls_loss = args.alpha * kd_loss + (1.0 - args.alpha) * ce_loss


        emd_loss = 



        return loss

    def emd_loss(student_reps, teacher_reps, student_attns, teacher_attns, loss_mse, T=1):
        def emd_attn_loss(student_attns, teacher_attns, student_layer_weight, teacher_layer_weight,
                            stu_layer_num, tea_layer_num, device):
            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
            total_num = stu_layer_num + tea_layer_num
            distance_matrix = torch.zeros([total_num, total_num]).to(device)
            
            for i in range(stu_layer_num):
                student_attn = student_attns[i]
                for j in range(tea_layer_num):
                    teacher_attn = teacher_attns[j]
                    tmp_loss = self.loss_func(student_attn, teacher_attn) 
                    distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

            _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                            distance_matrix.detach().cpu().numpy().astype('float64'))
            
            attn_loss = torch.sum(torch.tensor(trans_matrix).to(device) * distance_matrix)

            # tmp = distance_matrix.detach().cpu().numpy()
            
            return attn_loss, trans_matrix, distance_matrix

        def emd_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                            stu_layer_num, tea_layer_num, device):
            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
            total_num = stu_layer_num + tea_layer_num
            distance_matrix = torch.zeros([total_num, total_num]).to(device)
            
            for i in range(stu_layer_num):
                student_rep = student_reps[i]
                for j in range(tea_layer_num):
                    teacher_rep = teacher_reps[j]
                    tmp_loss = self.loss_func(student_rep, teacher_rep) 
                    distance_matrix[i][j + stu_layer_num] =  distance_matrix[j + stu_layer_num][i] = tmp_loss 
                
            _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                            distance_matrix.detach().cpu().numpy().astype('float64'))

            rep_loss = torch.sum(torch.tensor(trans_matrix).to(device) * distance_matrix)

            # tmp = distance_matrix.detach().cpu().numpy()

            return rep_loss, trans_matrix, distance_matrix

  
        stu_layer_num = len(student_reps)
        tea_layer_num = len(teacher_reps)

        rep_loss, rep_trans_matrix, rep_distance_matrix = \
            emd_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                            stu_layer_num, tea_layer_num, device)

        if args.update_weight:
            rep_student_weight, rep_teacher_weight = get_new_layer_weight(rep_student_weight, rep_teacher_weight, rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, args,T=T)

        if args.use_attn:
            attn_loss, attn_trans_matrix, attn_distance_matrix = \
            emd_attn_loss(student_attns, teacher_attns, attn_student_weight, attn_teacher_weight,
                            stu_layer_num, tea_layer_num, loss_mse, device)

            if args.update_weight:
                attn_student_weight, attn_teacher_weight = get_new_layer_weight(attn_student_weight, attn_teacher_weight, attn_trans_matrix, attn_distance_matrix, stu_layer_num, tea_layer_num, args,T=T)
    
        else:
            attn_loss = torch.tensor([0])

        loss = rep_loss + attn_loss
            
        return loss

                