# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class PatchMerging(nn.Module):
    """简易下采样，把 H,W 减半，通道翻倍"""
    def __init__(self, c_in):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_in*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(c_in*2)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn(self.proj(x)))
        return x

class PositionalEncoding(nn.Module):
    """标准正弦位置编码（随序列长度自适应）"""
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [L,D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe)  # [L,D]

    def forward(self, x):
        # x: [T,N,D]
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)  # [T,1,D] + [T,N,D]

class SVTRTiny(nn.Module):
    """
    极简 SVTR：CNN 把 H 压到 1 -> TransformerEncoder -> 全连接
    - in_channels: 默认 3（彩色）
    - 输出 [T, N, C]
    """
    def __init__(self, num_classes: int, in_channels: int = 3, embed_dim: int = 192, nhead: int = 6, nlayers: int = 4, dim_ff: int = 512):
        super().__init__()
        # CNN Stem：逐步下采样到较低高度，再自适应到 H=1
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, 64, k=3, s=1, p=1),
            PatchMerging(64),   # 128, H/2
            PatchMerging(128),  # 256, H/4
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.to_line = nn.AdaptiveAvgPool2d((1, None))  # [N,embed_dim,1,W']

        # Transformer Encoder（batch_first=False，对齐 train.py 的 [T,N,C] 约定）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_ff, batch_first=False, dropout=0.1, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pos = PositionalEncoding(embed_dim)
        self.fc  = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [N,C,H,W]
        x = self.stem(x)       # [N,D,H',W']
        x = self.to_line(x)    # [N,D,1,W']
        x = x.squeeze(2)       # [N,D,W']
        x = x.permute(2,0,1).contiguous()  # [T=W', N, D]
        x = self.pos(x)        # [T,N,D]
        y = self.encoder(x)    # [T,N,D]
        logits = self.fc(y)    # [T,N,C]
        return logits
