from torch import nn
import torch
import math

import torch.nn.functional as F2
import torch.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, rezero=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        # self.conv1.weight = self.truncated_normal_(self.conv1.weight, 0, 0.02)
        # self.conv1.bias.data.zero_()

        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation*2)
        # self.conv1.weight = self.truncated_normal_(self.conv1.weight, 0, 0.02)
        # self.conv1.bias.data.zero_()

        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.rezero = rezero
        if rezero:
            self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x): # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)
        out =  self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        out = F2.relu(self.ln1(out))

        out_pad = self.conv_pad(out, self.dilation*2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F2.relu(self.ln2(out2))
        if self.rezero:
            out2 *= self.resweight
        return out, out2 + x

    def conv_pad(self, x, dilation):
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

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

        x = self.w_2(F2.relu(self.w_1(self.layer_norm(x))))

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

