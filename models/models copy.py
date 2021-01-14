from torch import nn
import torch
import torch.functional as F
import torch.nn.functional as F2
import math
import numpy as np
from modules import *

class NextItNet_Decoder(nn.Module):

    def __init__(self, model_para):
        super(NextItNet_Decoder, self).__init__()
        self.item_size = model_para['item_size']
        self.embed_size = model_para['hidden_size']
        self.embeding = nn.Embedding(self.item_size, self.embed_size)
        stdv = np.sqrt(1. / self.item_size)
        self.embeding.weight.data.uniform_(-stdv, stdv) # important initializer
        # nn.init.uniform_(self.in_embed.weight, -1.0, 1.0)
        self.rezero = model_para['rezero']

        self.dilations = model_para['dilations']
        self.hidden_size = model_para['hidden_size']
        self.kernel_size = model_para['kernel_size']
        rb = [ResidualBlock(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size,
                            dilation=dilation, rezero=self.rezero) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rb) 
        self.final_layer = nn.Linear(self.hidden_size, self.item_size)
        self.final_layer.weight.data.normal_(0.0, 0.01)  # initializer
        self.final_layer.bias.data.fill_(0.1)
        if model_para["is_student"] and model_para['fit_size'] > 0:
            self.need_fit = True
            self.fit_dense = nn.Linear(self.hidden_size, model_para['fit_size'])
        else:
            self.need_fit = False

    def forward(self, x, onecall=False): # inputs: [batch_size, seq_len]
        
        hidden_outputs = []

        inputs = self.embeding(x) # [batch_size, seq_len, embed_size]       
        hidden_outputs.append(inputs)

        for rb in self.residual_blocks:
            t, inputs = rb(inputs)
            hidden_outputs.append(t)
            hidden_outputs.append(inputs)
            
        if onecall:
            hidden = inputs[:, -1, :].view(-1, self.hidden_size) # [batch_size, embed_size]
        else:
            hidden = inputs.view(-1, self.hidden_size) # [batch_size*seq_len, embed_size]
        
        logits = self.final_layer(hidden)

        tmp = []
        if self.need_fit:
            for _, sequence_layer in enumerate(hidden_outputs):
                tmp.append(self.fit_dense(sequence_layer))
            hidden_outputs = tmp

        return logits, hidden_outputs
 
class SASRec(nn.Module):
    def __init__(self, model_para):
        super(SASRec, self).__init__()
        self.item_size = model_para['item_size']
        self.embed_size = model_para['hidden_size']
        self.seq_len = model_para['seq_len']
        self.embeding = nn.Embedding(self.item_size, self.embed_size)
        self.pos_embedding = nn.Embedding(self.seq_len, self.embed_size)
        stdv = np.sqrt(1. / self.item_size)
        self.embeding.weight.data.uniform_(-stdv, stdv) # important initializer
        stdv = np.sqrt(1. / self.seq_len)
        self.pos_embedding.weight.data.uniform_(-stdv, stdv)
        # nn.init.uniform_(self.in_embed.weight, -1.0, 1.0)
        self.rezero = model_para['rezero']
        self.hidden_size = model_para['hidden_size']
        self.n_head = model_para['num_head']
        self.dropout = model_para['dropout']
        self.block_num = model_para['block_num']
        self.device = model_para['device']
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        
        tb = [Transformer_Block(self.n_head, self.hidden_size, self.hidden_size // self.n_head,\
              self.hidden_size // self.n_head, self.dropout) for i in range(self.block_num)]
                        
        self.transformer_blocks = nn.Sequential(*tb) 
        
        if model_para["is_student"] and model_para['fit_size'] > 0:
            self.need_fit = True
            self.fit_dense = nn.Linear(self.hidden_size, model_para['fit_size'])
        else:
            self.need_fit = False

    def forward(self, x, onecall=False): # inputs: [batch_size, seq_len]
        
        hidden_outputs = []

        inputs = self.embeding(x) # [batch_size, seq_len, embed_size]   
        pos = self.pos_embedding(torch.arange(self.seq_len-1).to(self.device))  
        inputs += pos  
        hidden_outputs.append(inputs)

        for tb in self.transformer_blocks:
            inputs, attns = tb(inputs)
            hidden_outputs.append(attns)
            hidden_outputs.append(inputs)
            
        inputs = self.layer_norm(inputs) 
        if onecall:
            hidden = inputs[:, -1, :].view(-1, self.hidden_size) # [batch_size, embed_size]
        else:
            hidden = inputs.view(-1, self.hidden_size) # [batch_size*seq_len, embed_size]
        
        
        logits = torch.matmul(hidden, self.embeding.weight.t())

        tmp = []
        if self.need_fit:
            for i in range(len(hidden_outputs)):
                if i % 2 != 0:
                    tmp.append(hidden_outputs[i])
                else:
                    tmp.append(self.fit_dense(hidden_outputs[i]))
            hidden_outputs = tmp

        return logits, hidden_outputs


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F2.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

