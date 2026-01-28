import os
import time
import random
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt


class IntegralRegression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, heatmaps):
        B, J, H, W = heatmaps.shape

        heatmaps = heatmaps.view(B, J, -1)
        probs = torch.softmax(heatmaps, dim=-1)
        probs = probs.view(B, J, H, W)

        xs = torch.linspace(0, 1, W, device=heatmaps.device)
        ys = torch.linspace(0, 1, H, device=heatmaps.device)

        prob_x = probs.sum(dim=2)
        prob_y = probs.sum(dim=3)

        x = (prob_x * xs).sum(dim=-1)
        y = (prob_y * ys).sum(dim=-1)

        return torch.stack([x, y], dim=-1)
