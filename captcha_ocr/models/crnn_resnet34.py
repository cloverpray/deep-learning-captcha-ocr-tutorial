# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            idt = self.down(idt)
        out = self.relu(out + idt)
        return out

def make_layer(in_ch, out_ch, nblock, stride=1):
    layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
    for _ in range(nblock-1):
        layers.append(ResidualBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)

class CRNNResNet34(nn.Module):
    """
    简化版 ResNet34 编码器 + 双向LSTM + 全连接，输出 [T, N, C]
    - in_channels: 3（彩色），与数据管道一致
    - 高度通过 AdaptiveAvgPool2d((1, None)) 压缩为 1，确保时序来自宽度
    """
    def __init__(self, num_classes: int, in_channels: int = 3, lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, 64, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2
        )
        # 近似 ResNet34 的层配置：3,4,6,3
        self.layer1 = make_layer(64,   64, nblock=3, stride=1)  # H/2
        self.layer2 = make_layer(64,  128, nblock=4, stride=2)  # H/4
        self.layer3 = make_layer(128, 256, nblock=6, stride=2)  # H/8
        self.layer4 = make_layer(256, 512, nblock=3, stride=2)  # H/16

        # 把高度压到 1，宽度保持自适应
        self.to_line = nn.AdaptiveAvgPool2d((1, None))  # [N,512,1,W']

        # 双向 LSTM 做时序建模
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=lstm_hidden, num_layers=lstm_layers,
            bidirectional=True, batch_first=False, dropout=0.1
        )
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x: [N,C,H,W]
        x = self.stem(x)
        x = self.layer1(x)  # [N,64,H/2,W']
        x = self.layer2(x)  # [N,128,H/4,W']
        x = self.layer3(x)  # [N,256,H/8,W']
        x = self.layer4(x)  # [N,512,H/16,W']
        x = self.to_line(x) # [N,512,1,W']
        x = x.squeeze(2)    # [N,512,W']
        x = x.permute(2,0,1).contiguous()  # [T=W', N, 512]
        y, _ = self.rnn(x)  # [T,N,2*hidden]
        logits = self.fc(y) # [T,N,C]
        return logits
