import torch
from torch import nn
from attention import MultiHeadAttention
import math

class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))
    
class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout, layer_norm_eps):
        super(AddNorm, self).__init__()
        self.ln = nn.LayerNorm(norm_shape, layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        return self.ln(x + self.dropout(y))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_feedforward, dropout, use_bias=False, layer_norm_eps=1e-5):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout, use_bias)
        self.addNorm_1 = AddNorm(d_model, dropout, layer_norm_eps)
        self.ffn = PositionWiseFFN(d_model, d_feedforward, d_model)
        self.addNorm_2 = AddNorm(d_model, dropout, layer_norm_eps)

    def forward(self, x, valid_lens):
        y = self.attention(x, x, x, valid_lens)
        y = self.addNorm_1(x, y)
        return self.addNorm_2(y, self.ffn(y))
    
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_feedforward, dropout, index, use_bias=False, layer_norm_eps=1e-5):
        super(DecoderBlock, self).__init__()
        self.index = index
        self.attention_1 = MultiHeadAttention(d_model, n_head, dropout, use_bias)
        self.addNorm_1 = AddNorm(d_model, dropout, layer_norm_eps)
        self.attention_2 = MultiHeadAttention(d_model, n_head, dropout, use_bias)
        self.addNorm_2 = AddNorm(d_model, dropout, layer_norm_eps)
        self.ffn = PositionWiseFFN(d_model, d_feedforward, d_model)
        self.addNorm_3 = AddNorm(d_model, dropout, layer_norm_eps)
        
    def forward(self, x, state):
        enc_output, enc_valid = state[0], state[1]
        if state[2][self.index] is None:
            k = x
        else:
            k = torch.cat((state[2][self.index], x), dim=1)
        state[2][self.index] = k
        if self.training:
            bs, n_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, n_steps + 1, device=x.device).repeat(bs, 1)
        else:
            dec_valid_lens = None
        
        y = self.attention_1(x, k, k, dec_valid_lens)
        y = self.addNorm_1(x, y)
        z = self.attention_2(y, enc_output, enc_output, dec_valid_lens)
        z = self.addNorm_2(y, z)
        return self.addNorm_3(z, self.ffn(z)), state



    
        