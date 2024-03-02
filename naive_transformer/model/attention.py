import torch
from torch import nn
import torch.nn.functional as F
import math

def mask_softmax(x, valid_lens):
    if valid_lens is None:
        return F.softmax(x, dim=-1)
    else:
        shape = x.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    x = x.reshape(-1, shape[-1])
    mask = torch.arange(shape[-1]).unsqueeze(0) < valid_lens.unsqueeze(1)
    x[mask==False] = -1e6
    return F.softmax(x.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, valid_lens=None):
        bs, n_q, dim = q.shape
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(dim)
        self.attn_weights = mask_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attn_weights), v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, bias=False, k_size=None, v_size=None):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        k_size = k_size if k_size is not None else d_model
        v_size = v_size if v_size is not None else d_model
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(k_size, d_model, bias=bias)
        self.W_v = nn.Linear(v_size, d_model, bias=bias)
        self.embedder = nn.Linear(d_model, d_model, bias=bias)
    
    def forward(self, q, k, v, valid_lens):
        q = self.transpose_qkv(self.W_q(q))
        k = self.transpose_qkv(self.W_k(k))
        v = self.transpose_qkv(self.W_v(v))
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.n_head, dim=0)
        output = self.attention(q, k, v, valid_lens)
        return self.embedder(self.transpose_output(output))
        
        
    def transpose_qkv(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.n_head, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])
    
    def transpose_output(self, x):
        x = x.reshape(-1, self.n_head, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)
        
    
