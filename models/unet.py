import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import GetDevice

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=(1, 1), bias=False),
            nn.GroupNorm(1, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=(1, 1), bias=False),
            nn.GroupNorm(1, out_ch),
        )
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim = 256) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )
    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim = 256) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
                DoubleConv(in_ch, in_ch, residual=True),
                DoubleConv(in_ch, out_ch, in_ch // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + emb
    
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1,2)
        x_ln = self.ln(x)

        attn_val,_ = self.mha(x_ln, x_ln, x_ln)
        attn_val = attn_val + x
        attn_val = self.ff_self(attn_val) + attn_val

        return attn_val.swapaxes(2,1).view(-1, self.channels, self.size, self.size)

class UNet(nn.Module):
    def __init__(self, in_ch = 1, out_ch = 1, time_dim = 256) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_dim = time_dim
        self.device = GetDevice()

        self.inc1 = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128,32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256,16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256,8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512,128)
        self.sa4 = SelfAttention(128,16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64,32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64,64)

        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)
    
    def PositionalEmbedding(self, t: torch.Tensor, channels: int):
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))

        pos_enc_a = torch.sin(t.repeat(1, channels //2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels //2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=1)
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.double)
        t = self.PositionalEmbedding(t, self.time_dim)

        x1 = self.inc1(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        return self.outc(x)