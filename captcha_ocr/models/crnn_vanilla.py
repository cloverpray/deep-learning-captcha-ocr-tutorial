# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class CRNNVanilla(nn.Module):
    """
    经典 CRNN（CNN + BiLSTM + CTC），输入 [N, C, H, W]
    - 自适应把高度汇聚到 1（AdaptiveAvgPool2d((1, None)))，避免 squeeze/permute 维度不匹配。
    - 默认 in_ch=1（灰度），与 train.py 中 vanilla_crnn 的数据通道设置一致。
    - 兼容参数名：既支持 in_ch，也支持 in_channels。
    """

    def __init__(self, num_classes: int, in_ch: int = 1, hidden: int = 256, **kwargs):
        # 兼容旧项目/其它模块传参名 in_channels
        if "in_channels" in kwargs and kwargs["in_channels"] is not None:
            in_ch = int(kwargs.pop("in_channels"))
        # 忽略可能的多余 kwargs（也可在此检查/报错）
        super().__init__()
        self.in_ch = in_ch

        # ---- CNN 主干 ----
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),  # H, W
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),             # H/2, W/2

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),             # H/4, W/4

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),   # H/8, W/4

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),   # H/16, W/4

            nn.Conv2d(512, 512, 2, 1, 0),   # 进一步拉直高度
            nn.ReLU(True),
        )

        # 关键：将高度自适应汇聚为 1，宽度保持不变
        self.hpool = nn.AdaptiveAvgPool2d((1, None))

        # ---- BiLSTM + 全连接 ----
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden,
            num_layers=2, bidirectional=True, batch_first=False
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, H, W]  ->  logits: [T, N, num_classes]
        """
        f = self.cnn(x)                      # [N, 512, H', W']
        if f.dim() != 4:
            raise RuntimeError(f"Expect CNN output 4D, got {f.dim()}D")
        f = self.hpool(f)                    # [N, 512, 1, W']
        f = f.squeeze(2)                     # [N, 512, W']
        f = f.permute(2, 0, 1).contiguous()  # [T=W', N, 512]
        y, _ = self.rnn(f)                   # [T, N, 2*hidden]
        logits = self.fc(y)                  # [T, N, num_classes]
        return logits


# === 兼容旧项目的类名 ===
CRNN_Vanilla_CTC = CRNNVanilla

__all__ = ["CRNNVanilla", "CRNN_Vanilla_CTC"]
