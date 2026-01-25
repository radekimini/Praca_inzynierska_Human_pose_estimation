import os
import time
import random
import math
import csv
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

import Training_Config as cfg
from Training_Dataset_Heatmap import PoseDatasetHeatmap
from Training_HeatmapPoseNet import HeatmapPoseNet
from Support_Functions import (
    compute_pixel_errors,
    draw_keypoints_on_image,
    ensure_dir,
    get_dataset_row,
    save_final_predictions_ext,
    debug_show_heatmaps,
    get_coord_weight
)
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

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

def train_heatmap(
    load_initial=False,
    load_final=False,
    resume_checkpoint=False,
    trial=None
):
    train_ds = PoseDatasetHeatmap(
        cfg.CSV_PATH,
        cfg.IMAGE_FOLDER,
        limit=cfg.DEBUG_LIMIT,
        sigma=cfg.HEATMAP_SIGMA
    )

    if len(train_ds) < 2:
        train_subset = train_ds
        val_subset = None
        train_size = len(train_ds)
        val_size = 0
    else:
        val_size = max(1, int(len(train_ds) * cfg.VAL_SPLIT))
        train_size = len(train_ds) - val_size

        train_subset, val_subset = random_split(
            train_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.RANDOM_STATE)
        )

    DEBUG_SAMPLE_IDXS = [0, 10, 20, 30, 40]
    DEBUG_JOINT_ID = 10

    fixed_debug_samples = []

    for idx in DEBUG_SAMPLE_IDXS:
        img, gt_heatmaps, gt_coords = train_subset[idx]
        fixed_debug_samples.append({
            "img": img.unsqueeze(0),
            "gt_heatmaps": gt_heatmaps.unsqueeze(0),
            "idx": idx
        })

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=2,
            shuffle=True
        )

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    model = HeatmapPoseNet(
        num_joints=cfg.NUM_KEYPOINTS
    ).to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    criterion = nn.MSELoss()
    coord_criterion = nn.SmoothL1Loss()
    scaler = GradScaler()

    start_epoch = 0

    if resume_checkpoint and os.path.exists(cfg.CHECKPOINT_PATH):
        ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from checkpoint @ epoch {start_epoch}")

    elif load_final and os.path.exists(cfg.FINAL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(cfg.FINAL_WEIGHTS_PATH))
        print("Loaded FINAL weights")

    elif load_initial and os.path.exists(cfg.INITIAL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(cfg.INITIAL_WEIGHTS_PATH))
        print("Loaded INITIAL weights")

    else:
        torch.save(model.state_dict(), cfg.INITIAL_WEIGHTS_PATH)
        print("Starting from random weights")

    ensure_dir(cfg.STATS_DIR)
    stats_csv = os.path.join(
        cfg.STATS_DIR,
        f"training_stats_raw_{cfg.MODEL_SUFFIX}.csv"
    )

    write_header = not os.path.exists(stats_csv)
    with open(stats_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch",
                "train_hm_loss",
                "train_px",
                "train_rmse",
                "val_px",
                "val_rmse",
                "epoch_time_s",
                "lr"
            ])

    history = []
    best_val_px = float("inf")
    best_epoch = None
    final_train_px = None
    final_val_px = None
    epochs_no_improve = 0

    total_start = time.time()

    for epoch in range(start_epoch, cfg.EPOCHS):
        avg_val_px = None
        epoch_start = time.time()
        model.train()

        train_loss = 0.0
        train_px_sum = 0.0
        train_rmse_sum = 0.0
        train_px_count = 0

        joint_err_sum = torch.zeros(cfg.NUM_KEYPOINTS, device=cfg.DEVICE)
        joint_acc_sum = torch.zeros(cfg.NUM_KEYPOINTS, device=cfg.DEVICE)
        joint_batches = 0

        num_batches = len(train_loader)
        lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS} | LR {lr:.2e}")

        for batch_idx, (imgs, gt_heatmaps, gt_coords) in enumerate(train_loader):
            imgs = imgs.to(cfg.DEVICE)
            gt_coords = gt_coords.to(cfg.DEVICE)
            gt_heatmaps = gt_heatmaps.to(cfg.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                pred_heatmaps, pred_coords = model(imgs)

                hm_loss = F.binary_cross_entropy_with_logits(
                    pred_heatmaps,
                    gt_heatmaps
                )

                coord_weight = get_coord_weight(epoch, cfg)

                if coord_weight > 0.0:
                    coord_loss = coord_criterion(pred_coords, gt_coords)
                    loss = hm_loss + coord_weight * coord_loss
                else:
                    coord_loss = torch.zeros((), device=cfg.DEVICE)
                    loss = hm_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            mean_px, rmse_px, joint_err, joint_acc = compute_pixel_errors(
                pred_coords.detach(),
                gt_coords,
                cfg.IMG_SIZE
            )

            joint_err_sum += joint_err
            joint_acc_sum += joint_acc
            joint_batches += 1

            train_px_sum += mean_px * imgs.size(0)
            train_rmse_sum += rmse_px * imgs.size(0)
            train_px_count += imgs.size(0)

            progress = (batch_idx + 1) / num_batches * 100
            elapsed = time.time() - epoch_start
            eta_epoch = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)

            print(
                f"\rProgress {progress:6.2f}% | "
                f"Batch {batch_idx+1}/{num_batches} | "
                f"L1 {loss.item():.4f} | "
                f"ETA {eta_epoch:5.1f}s",
                end="",
                flush=True
            )

        joint_err = joint_err_sum / joint_batches
        joint_acc = joint_acc_sum / joint_batches

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for i, sample in enumerate(fixed_debug_samples):
                    img = sample["img"].to(cfg.DEVICE)
                    gt_hm = sample["gt_heatmaps"]
                    pred_hm, _ = model(img)

                    debug_show_heatmaps(
                        img.detach().cpu(),
                        pred_hm.detach().cpu(),
                        gt_hm.detach().cpu(),
                        joint_id=DEBUG_JOINT_ID,
                        fname_prefix=f"epoch{epoch}_{cfg.MODEL_SUFFIX}_dbg{i}_idx{sample['idx']}",
                        model=cfg.MODEL_SUFFIX
                    )
            model.train()

        DEBUG_JOINT_ID = 10

        with torch.no_grad():
            phm = torch.sigmoid(pred_heatmaps[0, DEBUG_JOINT_ID])
            print(
                f"[DBG HM j{DEBUG_JOINT_ID}] "
                f"max={phm.max().item():.4f} "
                f"mean={phm.mean().item():.6f}"
            )

        avg_train_l1 = train_loss / num_batches
        avg_train_px = train_px_sum / max(1, train_px_count)
        avg_train_rmse = train_rmse_sum / max(1, train_px_count)

        final_train_px = avg_train_px
        if avg_val_px is not None:
            final_val_px = avg_val_px

        avg_val_px = None

        if val_loader is not None:
            model.eval()
            val_px_sum = 0.0
            val_px_count = 0
            val_rmse_sum = 0.0

            with torch.no_grad():
                for imgs, gt_heatmaps, gt_coords in val_loader:
                    imgs = imgs.to(cfg.DEVICE)
                    gt_coords = gt_coords.to(cfg.DEVICE)

                    _, pred_coords = model(imgs)
                    mean_px, rmse_px, _, _ = compute_pixel_errors(
                        pred_coords,
                        gt_coords,
                        cfg.IMG_SIZE
                    )

                    val_px_sum += mean_px * imgs.size(0)
                    val_rmse_sum += rmse_px * imgs.size(0)
                    val_px_count += imgs.size(0)

            avg_val_px = val_px_sum / max(1, val_px_count)
            avg_val_rmse = val_rmse_sum / max(1, val_px_count)

        if avg_val_px is not None and epoch >= 150:
            scheduler.step(avg_val_px)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1} | "
            f"Heatmap loss: {avg_train_l1:.4f} | "
            f"Mean pixel error: {avg_train_px:.2f}px"
        )

        for j, name in JOINT_NAMES.items():
            print(
                f"{name:12s} | "
                f"{joint_err[j]:6.2f}px | "
                f"acc≤10px {joint_acc[j] * 100:5.1f}%"
            )

        with torch.no_grad():
            print(
                "Pred coord mean:",
                pred_coords.mean(dim=(0, 1)).cpu().numpy()
            )

        if trial is not None:
            trial.report(avg_val_px, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        with open(stats_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_train_l1,
                avg_train_px,
                avg_train_rmse,
                avg_val_px,
                avg_val_rmse,
                epoch_time,
                lr
            ])

        history.append({
            "epoch": epoch + 1,
            "train_px": avg_train_px,
            "val_px": avg_val_px
        })

        if avg_val_px is not None and avg_val_px < best_val_px - cfg.EARLY_CHECKPOINT_DELTA:
            best_val_px = avg_val_px
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg.FINAL_WEIGHTS_PATH)
            print(f"✔ New BEST model ({best_val_px:.2f}px)")
        else:
            epochs_no_improve += 1
            print(f"No improvement {epochs_no_improve}/{cfg.EARLY_CHECKPOINT_PATIENCE}")

        if epochs_no_improve >= cfg.EARLY_CHECKPOINT_PATIENCE:
            print("EARLY STOPPING")
            break

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, cfg.CHECKPOINT_PATH)

        print("coord loss:", coord_loss.item())

    print("\nEvaluating on TEST set...")

    test_ds = PoseDatasetHeatmap(
        cfg.TEST_CSV_PATH,
        cfg.TEST_IMAGE_FOLDER
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False
    )

    model.eval()
    test_px_sum = 0.0

    with torch.no_grad():
        for imgs, gt_heatmaps, gt_coords in test_loader:
            imgs = imgs.to(cfg.DEVICE)
            gt_heatmaps = gt_heatmaps.to(cfg.DEVICE)
            gt_coords = gt_coords.to(cfg.DEVICE)

            pred_heatmaps, pred_coords = model(imgs)
            mean_px, rmse_px, joint_err, joint_acc = compute_pixel_errors(
                pred_coords.detach(),
                gt_coords,
                cfg.IMG_SIZE
            )

            test_px_sum += mean_px

    avg_test_px = test_px_sum / len(test_loader)
    print(f"FINAL TEST ERROR: {avg_test_px:.2f}px")

    if cfg.SAVE_FINAL_PREDS and trial is None:
        final_root = f"final_preds_raw_{cfg.MODEL_SUFFIX}"
        print("\nSaving final predictions...")
        try:
            save_final_predictions_ext(
                model,
                train_ds,
                os.path.join(final_root, "train"),
                device=cfg.DEVICE
            )

            save_final_predictions_ext(
                model,
                test_ds,
                os.path.join(final_root, "test"),
                device=cfg.DEVICE
            )
        except Exception as e:
            print(f"[WARN] Saving predictions failed: {e}")

        print(f"Predictions saved to {final_root}")

    epochs = [h["epoch"] for h in history]
    vals = [h["val_px"] for h in history]

    plt.figure()
    plt.plot(epochs, vals, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val Pixel Error")
    plt.grid(True)

    plot_path = os.path.join(
        cfg.STATS_DIR,
        f"loss_plot_raw_{cfg.MODEL_SUFFIX}.png"
    )
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot to {plot_path}")

    total_time = time.time() - total_start
    print(f"Total training time: {total_time/60:.1f} minutes")

    return {
        "best_val_px": best_val_px,
        "best_epoch": best_epoch,
        "epochs_ran": epoch + 1,
        "final_train_px": final_train_px,
        "final_val_px": final_val_px if final_val_px is not None else best_val_px,
    }
