import torch
import torch.nn as nn
import torch.nn.functional as F

from Training_ResidualBlock import ResidualBlock
from Training_Integral_Regression_Heatmap import IntegralRegression


class HeatmapPoseNet(nn.Module):
    def __init__(self, num_joints=16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.layer4 = ResidualBlock(256, 512)

        self.lateral3 = nn.Conv2d(256, 256, 1)
        self.lateral4 = nn.Conv2d(512, 256, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_joints, 1)
        )

        self.integral = IntegralRegression()

    def forward(self, x):
        x = self.stem(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode="bilinear", align_corners=False
        )

        feats = self.fuse(p3)

        heatmaps = self.head(feats)
        heatmaps = F.interpolate(
            heatmaps, size=(128, 128), mode="bilinear", align_corners=False
        )

        coords = self.integral(heatmaps)
        return heatmaps, coords
