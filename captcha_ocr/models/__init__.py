# -*- coding: utf-8 -*-
from .crnn_vanilla import CRNNVanilla
from .crnn_resnet34 import CRNNResNet34
from .svtr_tiny import SVTRTiny

def get_model(arch: str, num_classes: int):
    """
    返回指定架构的模型，输出为 [T, N, C]，其中 C=num_classes（包含CTC中的所有字符类，blank请放在外层CTCLoss里设为0）
    - vanilla_crnn  : 单通道（灰度）CRNN
    - crnn_resnet34 : 三通道（彩色）CRNN-ResNet变体
    - svtr_tiny     : 三通道（彩色）SVTR小模型
    """
    arch = arch.lower()
    if arch == "vanilla_crnn":
        # vanilla 用灰度更稳（与早前数据习惯一致）
        return CRNNVanilla(num_classes=num_classes, in_channels=1)
    elif arch == "crnn_resnet34":
        # resnet34 默认 3通道
        return CRNNResNet34(num_classes=num_classes, in_channels=3)
    elif arch == "svtr_tiny":
        # svtr 默认 3通道
        return SVTRTiny(num_classes=num_classes, in_channels=3)
    else:
        raise ValueError(f"Unknown arch: {arch}")
