from __future__ import annotations

import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=s, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BionnicaLite(nn.Module):
    def __init__(self, num_classes=2, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, k=3, s=2, p=1),
            DepthwiseSeparableConv(32, 64, s=1),
            DepthwiseSeparableConv(64, 128, s=2),
            DepthwiseSeparableConv(128, 128, s=1),
            DepthwiseSeparableConv(128, 256, s=2),
            DepthwiseSeparableConv(256, 256, s=1),
            DepthwiseSeparableConv(256, 512, s=2),
            DepthwiseSeparableConv(512, 512, s=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.features(x)
        pooled = self.pool(feats)
        return self.classifier(pooled)
