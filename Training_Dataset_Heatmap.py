import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import Training_Config as cfg


class PoseDatasetHeatmap(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        img_size=256,
        heatmap_size=128,
        sigma=cfg.HEATMAP_SIGMA,
        limit=None
    ):
        self.image_dir = image_dir
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.rows = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                self.rows.append(row)

        try:
            float(self.rows[0][1])
        except ValueError:
            self.rows = self.rows[1:]

        if limit:
            self.rows = self.rows[:limit]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rows)

    def _generate_heatmap(self, x, y):
        hm = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)

        scale = self.heatmap_size / self.img_size
        x = x * scale
        y = y * scale

        xx, yy = np.meshgrid(
            np.arange(self.heatmap_size),
            np.arange(self.heatmap_size)
        )

        hm = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
        return hm

    def __getitem__(self, idx):
        row = self.rows[idx]

        img_name = row[0]

        coords_px = np.array(row[1:], dtype=np.float32).reshape(-1, 2)
        coords_norm = coords_px / self.img_size

        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        heatmaps = np.zeros(
            (cfg.NUM_KEYPOINTS, self.heatmap_size, self.heatmap_size),
            dtype=np.float32
        )

        for j, (x, y) in enumerate(coords_px):
            heatmaps[j] = self._generate_heatmap(x, y)

        heatmaps = torch.from_numpy(heatmaps)
        coords_norm = torch.from_numpy(coords_norm)

        return img, heatmaps, coords_norm
