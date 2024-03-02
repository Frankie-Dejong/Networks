import math
import torch
from torch import nn
from blocks import EncoderBlock, DecoderBlock


class PositionalEncoding(nn.Module):
    def __init__(self, n_hidden, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.p = torch.zeros((1, max_len, n_hidden))
        j = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, n_hidden, 2, dtype=torch.float32) / n_hidden)
        self.p[:, :, 0::2] = torch.sin(j)
        self.p[:, :, 1::2] = torch.cos(j)
        
    def forward(self, x):
        x = x + self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_layers, d_model, n_head, d_feedforward, dropout, use_bias=False, layer_norm_eps=1e-5):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.blocks = nn.Sequential()
        for _ in range(n_layers):
            self.blocks.append(EncoderBlock(d_model, n_head, d_feedforward, dropout, use_bias, layer_norm_eps))
            
    def forward(self, x, valid_lens):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        self.attention_weights = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            x = block(x, valid_lens)
            self.attention_weights[i] = block.attention.attention.attn_weights
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_layers, d_model, n_head, d_feedforward, dropout, use_bias=False, layer_norm_eps=1e-5):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.blocks = nn.Sequential()
        for i in range(n_layers):
            self.blocks.append(DecoderBlock(d_model, n_head, d_feedforward, dropout, i, use_bias, layer_norm_eps))
        self.dense = nn.Linear(d_model, vocab_size)
        
    def init_state(self, enc_output, enc_valid):
        return [enc_output, enc_valid, [None] * self.n_layers]
    
    def forward(self, x, state):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        self._attention_weights = [[None] * len(self.blocks) for _ in range (2)]
        for i, block in enumerate(self.blocks):
            x, state = block(x, state)
        self._attention_weights[0][i] = block.attention_1.attention.attn_weights 
        self._attention_weights[1][i] = block.attention_2.attention.attn_weights
        return self.dense(x), state