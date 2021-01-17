from torch import nn
import numpy as np
import torch
import math
import torch.nn.functional as F

class SASRec(nn.Module):
    
    def __init__(self, config):
        super(SASRec, self).__init__()
        
        assert config['model_type'].lower() == "sasrec", "Wrong config file of the model, expected SASRec, but get {}.\n".format(config["model_type"])

        self.item_num = config['item_num']
        self.embed_size = config['embed_size']
        self.seq_len = config['seq_len']
        self.item_embedding = nn.Embedding(self.item_num, self.embed_size)
        self.pos_embedding = nn.Embedding(self.seq_len, self.embed_size)
        stdv = np.sqrt(1. / self.item_num)
        self.item_embeding.weight.data.uniform_(-stdv, stdv) # important initializer
        stdv = np.sqrt(1. / self.seq_len)
        self.pos_embedding.weight.data.uniform_(-stdv, stdv)
        
        self.hidden_size = config['hidden_size']
        self.device = config['device']

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        
        tb = [Transformer_Block(config['n_head'], self.hidden_size, self.hidden_size // config['n_head'],\
              self.hidden_size // config['n_head'], config['dropout']) for i in range(self.block_num)]
                        
        self.transformer_blocks = nn.Sequential(*tb) 
        
    def forward(self, x, onecall=False): # inputs: [batch_size, seq_len]
        
        hidden_outputs = []

        x = self.item_embedding(x) # [batch_size, seq_len, embed_size]   
        pos = self.pos_embedding(torch.arange(self.seq_len-1).to(self.device))  
        x += pos  
        hidden_outputs.append(x)

        for tb in self.transformer_blocks:
            x, attn = tb(x)
            hidden_outputs.append(attn)
            hidden_outputs.append(x)
            
        x = self.layer_norm(x) 
        
        if onecall:
            final_hidden = x[:, -1, :].view(-1, self.hidden_size) # [batch_size, embed_size]
        else:
            final_hidden = x.view(-1, self.hidden_size) # [batch_size*seq_len, embed_size]
        
        
        logits = torch.matmul(final_hidden, self.item_embedding.weight.t())

        return logits, hidden_outputs

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.5):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc_o = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        masks = torch.tril(torch.ones(scores.size(-2), scores.size(-1)), diagonal=0, out=None).to(q.device)
        scores.masked_fill_(masks==0, -1e9)
        tmp1 = scores[0,0,:,:].detach().cpu().numpy()

        attention = torch.softmax(scores, dim = -1)

        x = torch.matmul(attention, v)
        
        #x = [batch size, n heads, seq len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, seq len, n heads, head dim]
        
        x = x.view(sz_b, -1, self.d_model)
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(self.fc_o(x))

        x = residual + x

        return x, scores

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.5):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(self.layer_norm(x))))

        x = self.dropout(x) + residual

        return x

class Transformer_Block(nn.Module):
    def __init__(self, n_head, hidden_size, d_k, d_v, dropout):
        super(Transformer_Block, self).__init__()
        self.attn_block = MultiHeadAttention(n_head, hidden_size, d_k, d_v, dropout)
        self.ff_block = PositionwiseFeedForward(hidden_size, hidden_size, dropout)

    def forward(self, x):
        x, attns = self.attn_block(x, x, x)

        x = self.ff_block(x)

        return x, attns