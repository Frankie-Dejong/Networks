import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from diffusion import get_timestep_embeddings

def nonlinearity(x):
    return torch.sigmoid(x) * x

class UpsampleBlock(Module):
    def __init__(self, with_conv, channels=None, mode="bilinear"):
        super().__init__()
        self.with_conv = with_conv
        self.mode = mode
        if self.with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        
    def forward(self, x, time_embeddings):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=[H * 2, W * 2], mode=self.mode, align_corners=True)
        if self.with_conv:
            x = self.conv(x)
        assert x.shape == (B, C, 2 * H, 2 * W)
        return x, time_embeddings
    
class DownsampleBlock(Module):
    def __init__(self, with_conv, channels=None):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, 2, 1)
    
    def forward(self, x, time_embeddings):
        B, C, H, W = x.shape
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, 2, 2)
        return x, time_embeddings

class Activate(Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.dense = nn.Linear(in_dims, out_dims)
    
    def forward(self, x):
        x = nonlinearity(x)
        return self.dense(x)

class ResNetBlock(Module):
    def __init__(self, in_channels, out_channels, embedding_dims, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.activate = Activate(embedding_dims, out_channels)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv_residual = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        
    def forward(self, x, timeEmbedding):
        B, C, H, W = x.shape
        h = x
        x = nonlinearity(self.group_norm1(x))
        x = self.conv1(x)
        x += self.activate(timeEmbedding)[:,:,None,None]
        x = nonlinearity(self.group_norm2(x))
        x = self.dropout(x)
        if C != self.out_channels:
            h = self.conv_residual(h)
            
        assert x.shape == h.shape
        
        return x + h

class AttnBlock(Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.norm = nn.GroupNorm(num_groups=32, num_channels=hidden_dims)
        self.proj_q = nn.Linear(hidden_dims, hidden_dims)
        self.proj_k = nn.Linear(hidden_dims, hidden_dims)
        self.proj_v = nn.Linear(hidden_dims, hidden_dims)
        self.proj_out = nn.Linear(hidden_dims, hidden_dims)
        
    def forward(self, x: torch.tensor):
        B, C, H, W = x.shape
        h = x
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        
        w = torch.einsum('bhwc,bHWc->bhwHW', q, k) / (C ** 0.5)
        w = w.reshape((B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = w.reshape((B, H, W, H, W))
        x = torch.einsum('bhwHW,bHWc->bhwc', w, v)
        x = self.proj_out(x)
        
        x = x.permute(0, 3, 1, 2)
        
        assert x.shape == h.shape
        return x + h
    
class MLP(Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super().__init__()
        self.dense_in = nn.Linear(in_dims, hidden_dims)
        self.dense_out = nn.Linear(hidden_dims, out_dims)
        
    def forward(self, x):
        x = self.dense_in(x)
        x = nonlinearity(x)
        x = self.dense_out(x)
        return x
    
class BasicBlock(Module):
    def __init__(self, in_channels, out_channels, embedding_dims, dropout, need_attn=False):
        super().__init__()
        self.need_attn = need_attn
        self.resnet_block = ResNetBlock(in_channels, out_channels, embedding_dims, dropout)
        if self.need_attn:
            self.attn_block = AttnBlock(out_channels)
        
    def forward(self, x, time_embedding):
        x = self.resnet_block(x, time_embedding)
        if self.need_attn:
            x = self.attn_block(x)
        return x, time_embedding

class Unet(Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_res_blocks, 
                 attn_resolutions, 
                 initial_resolutions, 
                 dropout=0., 
                 ch_mult=(1, 2, 4, 8), 
                 resample_withconv=True):
        super().__init__()
        self.in_chanels = in_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.timeEmbedder = MLP(in_channels, in_channels * 4, in_channels * 4)
        self.embedding_dims = in_channels * 4
        self.conv_in = nn.Conv2d(self.out_channels, in_channels, 3, 1, 1)
        down_blocks = []
        mid_blocks = []
        up_blocks = []
        channels = [in_channels * x for x in ch_mult]
        resolution = initial_resolutions
        cur_channel = in_channels
        down_channels = [in_channels]
        # down
        for i_level in range(self.num_resolutions):
            for i_block in range(num_res_blocks):
                if resolution in attn_resolutions:
                    down_blocks.append(BasicBlock(cur_channel, channels[i_level], self.embedding_dims, dropout, True))
                else:
                    down_blocks.append(BasicBlock(cur_channel, channels[i_level], self.embedding_dims, dropout, False))
                cur_channel = channels[i_level]
                down_channels.append(cur_channel)
            if i_level != self.num_resolutions - 1:
                down_blocks.append(DownsampleBlock(with_conv=resample_withconv, channels=channels[i_level]))
                resolution = resolution // 2
                down_channels.append(channels[i_level])
        # middle
        mid_channels = channels[-1]
        mid_blocks.append(BasicBlock(mid_channels, mid_channels, self.embedding_dims, dropout, True))
        mid_blocks.append(BasicBlock(mid_channels, mid_channels, self.embedding_dims, dropout, False))
        # up
        cur_channel = mid_channels
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                if resolution in attn_resolutions:
                    up_blocks.append(BasicBlock(cur_channel + down_channels.pop(), channels[i_level], self.embedding_dims, dropout, True))
                else:
                    up_blocks.append(BasicBlock(cur_channel + down_channels.pop(), channels[i_level], self.embedding_dims, dropout, False))
                cur_channel = channels[i_level]
            if i_level != 0:
                up_blocks.append(UpsampleBlock(with_conv=resample_withconv, channels=channels[i_level]))
                resolution = resolution * 2
        assert not down_channels
        self.down_blocks = down_blocks
        self.mid_blocks = mid_blocks
        self.up_blocks = up_blocks
        self.norm_out = nn.GroupNorm(32, in_channels)
        self.conv_out = nn.Conv2d(in_channels, out_channels, 3, 1, 1)        
        self.down_seq = nn.Sequential(*down_blocks)
        self.mid_seq = nn.Sequential(*mid_blocks)
        self.up_seq = nn.Sequential(*up_blocks)
        
    def forward(self, x, t):
        time_embeddings = get_timestep_embeddings(t, self.in_chanels)
        time_embeddings = self.timeEmbedder(time_embeddings)
        hs = []
        hs.append((self.conv_in(x), time_embeddings))
        for block in self.down_blocks:
            hs.append(block(*hs[-1]))
        h = hs[-1]
        for block in self.mid_blocks:
            h = block(*h)
        for block in self.up_blocks:
            if isinstance(block, BasicBlock):
                map_from_down, temb = hs.pop()
                map_this, temb = h
                new_map = torch.concat((map_this, map_from_down), dim=1)
                h = (new_map, temb)
                h = block(*h)
            else:
                h = block(*h)
        assert not hs
        
        x_pred, temb = h
        x_pred = nonlinearity(self.norm_out(x_pred))
        x_pred = self.conv_out(x_pred)
        assert x_pred.shape == (x.shape[0], self.out_channels, x.shape[2], x.shape[3])
        return x_pred