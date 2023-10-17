"""
Problem is the input shape changes during upsampling. It is not used.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List
from labml_helpers.module import Module

class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, num_ch: int) -> None:
        super().__init__()
        self.num_ch = num_ch

        self.lin_1 = nn.Linear(self.num_ch // 4, self.num_ch)
        self.act = Swish()
        self.lin_2 = nn.Linear(self.num_ch, self.num_ch)
    
    def forward(self, t: th.Tensor):
        half_dim = self.num_ch // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = th.cat([emb.sin(), emb.cos()], dim=-1)

        return self.lin_2(self.act(self.lin_1(emb)))

class ResidualBlock(Module):
    '''
    A residual block has two convolution layers with group normalization. Each resolution is processed with two residual blocks.
    '''
    def __init__(self, in_ch: int, out_ch: int, time_ch: int,
                 num_groups: int = 16, dropout: float = 0.1) -> None:
        '''
        in_ch: the number of input channels for the model. 
        out_ch: the number of output channels for the model. 
        time_ch: the number of time channels for the model.
        num_groups: the number of groups to use in the group normalization layers.
        dropout: dropout probability to use in the model.
        '''
        super().__init__()

        '''
        Group normalization and the first convolution layer.
        '''
        self.norm_1 = nn.GroupNorm(num_groups, in_ch)
        self.act_1 = Swish()
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), padding=(1,1))

        '''
        Group normalization and the second convolution layer.
        '''
        self.norm_2 = nn.GroupNorm(num_groups, out_ch)
        self.act_2 = Swish()
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), padding=(1,1))

        '''
        Check if in_ch == out_ch. If so, then we can add the input to the output of the second convolution layer.
        '''
        if in_ch == out_ch:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=(1,1))
        
        '''
        Linear layer to embed the time dimension.
        '''
        self.time_embed = nn.Linear(time_ch, out_ch)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: th.Tensor, t: th.Tensor):
        '''
        x.shape = (batch_size, in_ch, height, width)
        t.shape = (batch_size, time_ch)
        '''
        h = self.conv_1(self.act_1(self.norm_1(x)))
        h += self.time_embed(self.time_act(t))[:, :, None, None]
        h = self.conv_2(self.dropout(self.act_2(self.norm_2(h))))

        return h + self.proj(x)

class AttentionBlock(Module):
    '''
    Similar to Transformer MHA block.
    '''
    def __init__(self, num_ch: int, num_head: int = 1, d_k: int = None, num_group: int = 32) -> None:
        """
        Args:
            num_ch (int): Number of input channels.
            num_head (int): Number of attention heads.
            d_k (int): Dimensionality of the key and query vectors.
            num_group (int, optional): Number of groups for group normalization. Defaults to 32.
        """
        super().__init__()
        
        if d_k is None:
            d_k = num_ch
        
        self.norm = nn.GroupNorm(num_group, num_ch)                 # Group normalization
        self.projection = nn.Linear(num_ch, num_head * d_k * 3)     # Projections for query, key and values
        self.output = nn.Linear(num_head * d_k, num_ch)             # Output projection
        self.scale = d_k ** -0.5                                    # Scale for the dot product attention

        self.num_head = num_head
        self.d_k = d_k

    def forward(self, x: th.Tensor, t: Optional[th.Tensor] = None):
        _ = t

        batch_size, num_ch, height, width = x.shape
        x = x.view(batch_size, num_ch, -1).permute(0,2,1)           # Change x to shape [batch_size, seq, num_channels]

        qkv = self.projection(x).view(batch_size, -1, 
                                      self.num_head, 3 * self.d_k)  # Concatenated query, key and value projections
        
        q, k, v = th.chunk(qkv, 3, dim=-1)                          # Split into query, key and value

        attn = th.einsum('bihd, bjhd -> bijh', q, k) * self.scale   # Compute scaled dot product attention
        attn = attn.softmax(dim=2)                                  # Apply softmax to get attention weights

        res = th.einsum('bijh, bjhd -> bihd', attn, v)              # Compute attention-weighted sum of values
        res = res.view(batch_size, -1, self.num_head * self.d_k)    # Concatenate attention heads
        res = self.output(res)                                      # Apply output projection

        res += x                                                    # Add skip connection
        res = res.permute(0,2,1).view(batch_size, num_ch, 
                                      height, width)                # Reshape to [batch_size, num_channels, height, width]

        return res

class DownBlock(Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, has_attn: bool) -> None:
        super().__init__()

        self.res = ResidualBlock(in_ch, out_ch, time_ch)
        if has_attn:
            self.attn = AttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: th.Tensor, t: th.Tensor):
        x = self.res(x, t)
        x = self.attn(x)

        return x

class UpBlock(Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, has_attn: bool) -> None:
        super().__init__()

        self.res = ResidualBlock(in_ch + out_ch, out_ch, time_ch)
        if has_attn:
            self.attn = AttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: th.Tensor, t: th.Tensor):
        x = self.res(x, t)
        x = self.attn(x)

        return x

class MiddleBlock(Module):
    def __init__(self, num_ch: int, time_ch: int) -> None:
        super().__init__()

        self.res_1 = ResidualBlock(num_ch, num_ch, time_ch)
        self.attn = AttentionBlock(num_ch)
        self.res_2 = ResidualBlock(num_ch, num_ch, time_ch)

    def forward(self, x: th.Tensor, t: th.Tensor):
        x = self.res_1(x, t)
        x = self.attn(x)
        x = self.res_2(x, t)

        return x

class Upsample(nn.Module):
    def __init__(self, num_ch: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(num_ch, num_ch, (4,4), (2,2), (1,1))
    
    def forward(self, x: th.Tensor, t: th.Tensor):
        _ = t
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, num_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(num_ch, num_ch, (3,3), (2,2), (1,1))
    
    def forward(self, x: th.Tensor, t: th.Tensor):
        _ = t
        
        return self.conv(x)

class UNetDDPM(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()

        im_ch = params.get("Image Channels")                    # Number of image channels.
        num_ch = params.get("Feature Channels")                 # Number of feature channels.
        ch_multipliers = params.get("Channel Multipliers")      # Channel multipliers.
        is_attn = [x == 1 for x in params.get("Attention")]     # Attention blocks.
        n_blocks = params.get("NumBlocks")                      # Number of blocks per resolution.

        n_res_blocks = len(ch_multipliers)                      # Number of resolutions.
        self.img_proj = nn.Conv2d(im_ch, num_ch, 
                                    kernel_size=(3,3), 
                                    padding=(1,1))              # Projection for the image.
        self.time_embed = TimeEmbedding(num_ch * 4)             # Embedding for the time dimension.

        '''
        First half of the model.
        '''
        self.down_blocks = nn.ModuleList()                      # List of down blocks.

        out_ch = in_ch = num_ch                                 # Number of input and output channels.

        for i in range(n_res_blocks):                           # Loop over resolutions.
            out_ch = num_ch * ch_multipliers[i]
            for _ in range(n_blocks):
                self.down_blocks.append(DownBlock(              # Append a down block.
                                                    in_ch, 
                                                    out_ch, 
                                                    num_ch * 4, 
                                                    is_attn[i]
                                                    ))
                in_ch = out_ch                                  # Update the number of input channels.
            if i < n_res_blocks - 1:                            # If not the last resolution.
                self.down_blocks.append(Downsample(in_ch))
        
        self.middle_block = MiddleBlock(out_ch, num_ch * 4,)    # Middle block.

        self.up_blocks = nn.ModuleList()                        # List of up blocks.
        in_ch = out_ch                                          # Number of input channels.

        for i in reversed(range(n_res_blocks)):
            out_ch = in_ch
            for _ in range(n_blocks):
                self.up_blocks.append(UpBlock(                  # Append an up block.
                                        in_ch, 
                                        out_ch, 
                                        num_ch * 4, 
                                        is_attn[i]
                                        ))
            out_ch = in_ch // ch_multipliers[i]                 # Update the number of output channels.
            self.up_blocks.append(UpBlock(                      # Append an up block.
                                    in_ch, 
                                    out_ch, 
                                    num_ch * 4, 
                                    is_attn[i]
                                    ))
            in_ch = out_ch                                      # Update the number of input channels.
            if i > 0:                                           # If not the first resolution.
                self.up_blocks.append(Upsample(in_ch))
        
        self.norm = nn.GroupNorm(8, num_ch)                     # Group normalization.
        self.act = Swish()                                      # Activation function.
        self.final = nn.Conv2d(num_ch, im_ch, (3,3), (1,1))     # Final convolution layer.
    
    def forward(self, x: th.Tensor, t: th.Tensor):
        '''
        x has shape (batch_size, num_channels, height, width)
        t has shape (batch_size)
        '''
        t = self.time_embed(t)                                  # Embed the time dimension.
        x = self.img_proj(x)                                    # Project the image.
        h = [x]                                                 # List of hidden states.

        for m in self.down_blocks:
            x = m(x, t)                                         # Apply the down blocks.
            h.append(x)                                         # Append the hidden state.
        
        x = self.middle_block(x, t)                             # Apply the middle block.

        for m in self.up_blocks:
            if isinstance(m, Upsample):
                x = m(x, t)                                     # Apply the upsample block.
            else:
                s = h.pop()                                     # Get the hidden state.
                x = th.cat((x, s), dim=1)                       # Apply the up block.
                x = m(x, t)
        
        return self.final(self.act(self.norm(x)))               # Apply the final convolution layer.