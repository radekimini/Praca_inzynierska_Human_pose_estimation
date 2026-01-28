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

MODEL_SUFFIX = f"OPTUNA_GHM"

CSV_PATH = "data/merged_1000/keypoints_merged.csv"
IMAGE_FOLDER = "data/merged_1000/images_merged"
TEST_CSV_PATH = "data/test_data/test_keypoints.csv"
TEST_IMAGE_FOLDER = "data/test_data/test_images"

INITIAL_WEIGHTS_PATH = f"checkpoints/initial_weights_{MODEL_SUFFIX}.pth"
FINAL_WEIGHTS_PATH = f"checkpoints/final_weights_{MODEL_SUFFIX}.pth"
CHECKPOINT_PATH = f"checkpoints/pose_model_checkpoint_{MODEL_SUFFIX}.pth"

EARLY_CHECKPOINT_DELTA = 0.005
EARLY_CHECKPOINT_PATIENCE = 50
EARLY_CHECKPOINT_PATH_TEMPLATE = "checkpoints/early_checkpoint_{suffix}_epoch{epoch}.pth"

HEATMAP_WARMUP_EPOCHS = 100
COORD_LOSS_WEIGHT = 1.0

DROPOUT = 0.15
HEATMAP_SIGMA = 5.0

PREDICTIONS_DIR = "predictions"
STATS_DIR = "training_stats"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_KEYPOINTS = 16
HEATMAP_WARMUP_EPOCHS = 20
COORD_LOSS_WEIGHT = 0.5
SAVE_FINAL_PREDS = True
nn_depth = 60
IMG_SIZE = 256
NUM_KEYPOINTS = 16
VAL_SPLIT = 0.15
EPOCHS = 500
BATCH_SIZE = 16
LR = 3e-4
DEBUG_LIMIT = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRED_SAMPLES = 50
SAVE_PRED_INTERVAL = 100

RANDOM_STATE = 45
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
