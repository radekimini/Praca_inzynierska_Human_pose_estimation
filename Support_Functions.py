import os
import time
import random
import math
import csv
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import Training_Config as cfg
import random

JOINT_NAMES = {
    0: "R_FOOT",
    1: "R_KNEE",
    2: "R_HIP",
    3: "L_HIP",
    4: "L_KNEE",
    5: "L_FOOT",
    6: "C_HIP",
    7: "C_SHOULDER",
    8: "NECK",
    9: "HEAD",
    10: "R_HAND",
    11: "R_ELBOW",
    12: "R_SHOULDER",
    13: "L_SHOULDER",
    14: "L_ELBOW",
    15: "L_HAND",
}

def compute_pixel_errors(pred_coords, gt_coords, img_size, thresh=10.0):
    pred_px = pred_coords * img_size
    gt_px = gt_coords * img_size

    dists = torch.norm(pred_px - gt_px, dim=-1)

    mean_px = dists.mean().item()
    rmse_px = torch.sqrt((dists ** 2).mean()).item()

    joint_err = dists.mean(dim=0)
    joint_acc = (dists < thresh).float().mean(dim=0)

    return mean_px, rmse_px, joint_err, joint_acc

def draw_keypoints_on_image(img_path, pred_kp, gt_kp, out_path, img_size=cfg.IMG_SIZE):
    with Image.open(img_path).convert("RGB") as im:
        im = im.resize((img_size, img_size))
        draw = ImageDraw.Draw(im)
        J = min(len(gt_kp), len(pred_kp))
        for j in range(J):
            gx = float(gt_kp[j][0]) * img_size
            gy = float(gt_kp[j][1]) * img_size
            px = float(pred_kp[j][0]) * img_size
            py = float(pred_kp[j][1]) * img_size
            r = 3
            draw.ellipse((gx - r, gy - r, gx + r, gy + r), outline="green", width=2)
            draw.ellipse((px - r, py - r, px + r, py + r), outline="red", width=2)
            draw.line((gx, gy, px, py), fill="yellow", width=1)
        im.save(out_path)

def get_dataset_row(dataset, global_idx):
    if isinstance(dataset, ConcatDataset):
        cum = 0
        for ds in dataset.datasets:
            if global_idx < cum + len(ds):
                local_idx = global_idx - cum
                return ds.get_row(local_idx)
            cum += len(ds)
        raise IndexError("Index out of range for ConcatDataset")
    else:
        return dataset.get_row(global_idx)

def save_final_predictions_ext(model, dataset, out_dir, device=cfg.DEVICE, max_samples=None):
    ensure_dir(out_dir)
    model.eval()

    with torch.no_grad():
        for idx in range(len(dataset)):
            if max_samples is not None and idx >= max_samples:
                break

            img_path, gt_kp = dataset.get_row(idx)

            img = Image.open(img_path).convert("RGB").resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

            preds, _ = model(img_tensor)
            pred_kp = preds[0].cpu().numpy()
            gt_kp_np = gt_kp.cpu().numpy()

            out_path = os.path.join(out_dir, f"{idx:03d}.png")
            draw_keypoints_on_image(
                img_path,
                pred_kp,
                gt_kp_np,
                out_path,
                img_size=cfg.IMG_SIZE
            )

    model.train()

def save_final_predictions(model, dataset, out_dir, device, max_samples=None):
    ensure_dir(out_dir)
    model.eval()

    with torch.no_grad():
        for idx in range(len(dataset)):
            if max_samples is not None and idx >= max_samples:
                break

            img_path, gt_kp = dataset.get_row(idx)

            img = Image.open(img_path).convert("RGB").resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

            preds = model(img_tensor)
            pred_kp = preds[0].cpu().numpy()
            gt_kp_np = gt_kp.cpu().numpy()

            out_path = os.path.join(out_dir, f"{idx:03d}.png")
            draw_keypoints_on_image(
                img_path,
                pred_kp,
                gt_kp_np,
                out_path,
                img_size=cfg.IMG_SIZE
            )

    model.train()

def save_debug_heatmaps(out_dir, epoch, imgs, pred_hm, gt_hm, joint_id=0):
    os.makedirs(out_dir, exist_ok=True)

    img = imgs[0].permute(1, 2, 0).numpy()
    phm = pred_hm[0, joint_id].numpy()
    ghm = gt_hm[0, joint_id].numpy()

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT heatmap")
    plt.imshow(ghm, cmap="hot", vmin=0, vmax=1)
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Pred heatmap")
    plt.imshow(phm, cmap="hot")
    plt.colorbar()

    plt.tight_layout()

    fname = os.path.join(
        out_dir,
        f"epoch_{epoch:04d}_joint_{joint_id}.png"
    )

    plt.savefig(fname)
    plt.close(fig)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def crop_person(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2 - x1, y2 - y1)

def preprocess(crop, device):
    img = cv2.resize(crop, (256, 256))
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

def backproject_keypoints(preds, bbox):
    x, y, w, h = bbox
    pts = []

    for px, py in preds:
        pts.append((
            int(x + px * w),
            int(y + py * h)
        ))

    return pts

SKELETON = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (7, 10), (10, 11), (11, 12),
    (7, 13), (13, 14), (14, 15)
]

def draw_skeleton(img, keypoints, color=(0, 255, 0)):
    h, w, _ = img.shape

    pts = []
    for (x, y) in keypoints:
        pts.append((int(x * w), int(y * h)))

    for (i, j) in SKELETON:
        cv2.line(img, pts[i], pts[j], color, 2)

    for p in pts:
        cv2.circle(img, p, 4, (0, 0, 255), -1)

    return img

def debug_show_heatmaps(imgs, pred_hm, gt_hm, joint_id=0, out_dir="debug_heatmaps", fname_prefix="sample", model="undefined"):
    final_out_dir = f"{out_dir}_{model}"
    os.makedirs(final_out_dir, exist_ok=True)

    joint_name = JOINT_NAMES.get(joint_id, f"joint{joint_id}")

    img = imgs[0].permute(1, 2, 0).cpu().numpy()

    phm = torch.sigmoid(pred_hm[0, joint_id])
    phm = phm / (phm.max() + 1e-6)
    phm = phm.cpu().numpy()

    ghm = gt_hm[0, joint_id].cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT heatmap")
    plt.imshow(ghm, cmap="hot", vmin=0, vmax=1)
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Pred heatmap")
    plt.imshow(phm, cmap="hot", vmin=0, vmax=1)
    plt.colorbar()

    plt.tight_layout()

    out_path = os.path.join(
        final_out_dir,
        f"{fname_prefix}_{joint_name}_j{joint_id}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def build_debug_indices(dataset, joints, images_per_joint=5, seed=123):
    random.seed(seed)
    debug_indices = {j: [] for j in joints}

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        for j in joints:
            if target["visible"][j] and len(debug_indices[j]) < images_per_joint:
                debug_indices[j].append(idx)

        if all(len(v) == images_per_joint for v in debug_indices.values()):
            break

    return debug_indices

def get_coord_weight(epoch, cfg):
    if epoch < cfg.HEATMAP_WARMUP_EPOCHS:
        return 0.0
    elif epoch < cfg.HEATMAP_WARMUP_EPOCHS * 3:
        return cfg.COORD_LOSS_WEIGHT * (
            (epoch - cfg.HEATMAP_WARMUP_EPOCHS) /
            (cfg.HEATMAP_WARMUP_EPOCHS * 2)
        )
    else:
        return cfg.COORD_LOSS_WEIGHT
