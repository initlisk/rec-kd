from torch import nn
import numpy as np
import torch.nn.functional as F

class NextItNet(nn.Module):

    def __init__(self, model_config):

        super(NextItNet, self).__init__()

        assert model_config['model_type'] == "NextItNet", "Wrong config file of the model, expected NextItNet, but get {}.\n".format(model_config["model_type"])

        self.item_num = model_config['item_num']
        self.embed_size = model_config['embed_size']
        self.item_embedding = nn.Embedding(self.item_num, self.embed_size)
        stdv = np.sqrt(1. / self.item_size)
        self.item_embedding.weight.data.uniform_(-stdv, stdv) # important initializer

        self.hidden_size = model_config['hidden_size']

        rb = [ResidualBlock(self.hidden_size, self.hidden_size, model_config['kernel_size'],
                            dilation_size) for dilation_size in model_config['dilations']]

        self.residual_blocks = nn.Sequential(*rb) 

        self.final_layer = nn.Linear(self.hidden_size, self.item_num)
        self.final_layer.weight.data.normal_(0.0, 0.01)  # initializer
        self.final_layer.bias.data.fill_(0.1)

    def forward(self, x, onecall=False): # inputs: [batch_size, seq_len]
        
        hidden_output = []

        x = self.item_embedding(x) # [batch_size, seq_len, embed_size]       
        hidden_output.append(x)

        for rb in self.residual_blocks:
            hid, x = rb(x)
            hidden_output.append(hid)
            hidden_output.append(x)
            
        if onecall:
            final_hidden = x[:, -1, :].view(-1, self.hidden_size) # [batch_size, embed_size]
        else:
            final_hidden = x.view(-1, self.hidden_size) # [batch_size*seq_len, embed_size]
        
        logits = self.final_layer(final_hidden)

        return logits, hidden_output


# nextitnet中的残差块
class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation_size=None):
        
        super(ResidualBlock, self).__init__()

        self.dilation_size = dilation_size
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation_size)
        # self.conv1.weight = self.truncated_normal_(self.conv1.weight, 0, 0.02)
        # self.conv1.bias.data.zero_()
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation_size*2)
        # self.conv1.weight = self.truncated_normal_(self.conv1.weight, 0, 0.02)
        # self.conv1.bias.data.zero_()
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

    def forward(self, x): # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_padding(x, self.dilation_size)
        out1 = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        out1 = F.relu(self.ln1(out1))

        out1_pad = self.conv_padding(out1, self.dilation_size*2)
        out2 = self.conv2(out1_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))

        return out1, out2 + x

    def conv_padding(self, x, dilation_size):
        x = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        x = x.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        
        padding = nn.ZeroPad2d(((self.kernel_size - 1) * dilation_size, 0, 0, 0))
        x_pad = padding(x)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]

        return x_pad

    def truncated_normal_(self, tensor, mean=0, std=0.09):

        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

        return tensor