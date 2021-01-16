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
    
    def forward(self, x, label): # inputs: [batch_size, seq_len]
        with torch.no_grad():
            teacher_output = self.teacher_model(x)
        sttudent_output = self.student_model(x)
        sttudent_output[]


        emd_loss = 



        return loss

    def emd_loss(student_reps, teacher_reps, student_attns, teacher_attns, T=1):
        global rep_student_weight, rep_teacher_weight
        global attn_student_weight, attn_teacher_weight
        def emd_attn_loss(student_attns, teacher_attns, student_layer_weight, teacher_layer_weight,
                            stu_layer_num, tea_layer_num, ):
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
                    tmp_loss = loss_mse(student_attn, teacher_attn) 
                    if args.emd_type == "v3":
                        t_dis_matrix[i][j + stu_layer_num] = t_dis_matrix[j + stu_layer_num][i] = tmp_loss
                        tmp_loss *= (1+abs(i/stu_layer_num - j/tea_layer_num) / 5) 
                    distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

            _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                            distance_matrix.detach().cpu().numpy().astype('float64'))
            
            if args.emd_type == "v3":
                attn_loss = torch.sum(torch.tensor(trans_matrix).to(args.device) * t_dis_matrix)
            else:
                attn_loss = torch.sum(torch.tensor(trans_matrix).to(args.device) * distance_matrix)

            tmp = distance_matrix.detach().cpu().numpy()
            
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
                    tmp_loss = loss_mse(student_rep, teacher_rep) 
                    distance_matrix[i][j + stu_layer_num] =  distance_matrix[j + stu_layer_num][i] = tmp_loss 

            if args.emd_type == "v10":
                tmp = distance_matrix.detach().cpu().numpy()
                student_layer_weight = tmp.mean(axis = 1)[:stu_layer_num]
                teacher_layer_weight = tmp.mean(axis = 0)[stu_layer_num:]
                student_layer_weight = sum(student_layer_weight) / student_layer_weight
                teacher_layer_weight = sum(teacher_layer_weight) / teacher_layer_weight
                student_layer_weight = utils.softmax(student_layer_weight / 20)
                teacher_layer_weight = utils.softmax(teacher_layer_weight / 30)
                student_layer_weight = utils.softmax(student_layer_weight)
                teacher_layer_weight = utils.softmax(teacher_layer_weight)
                #print (student_layer_weight)
                #print (teacher_layer_weight)
                student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
                teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
            elif args.emd_type == "v12":
                tmp = distance_matrix.detach().cpu().numpy()
                student_layer_weight = tmp.mean(axis = 1)[:stu_layer_num]
                teacher_layer_weight = tmp.mean(axis = 0)[stu_layer_num:]
                student_layer_weight = utils.softmax(student_layer_weight / 5)
                teacher_layer_weight = utils.softmax(teacher_layer_weight / 5)
                #student_layer_weight = utils.softmax(student_layer_weight)
                #teacher_layer_weight = utils.softmax(teacher_layer_weight)
                #print (student_layer_weight)
                #print (teacher_layer_weight)
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

                