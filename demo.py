from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from . import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform
from types import SimpleNamespace
from pathlib import Path

POINT_CONSUMER = None

INPUT_TYPE = "webcam"
CFG_PATH = "experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml"
VIDEO_PATH = "demo/Take8_Calibratedc.mp4"
IMAGE_PATH = None
WEBCAM_INDEX = 0

SHOW_WINDOW = False
SHOW_FPS = False
WRITE_VIDEO = False

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

PIXEL_THRESHOLD = 10.0

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [0,1], [1,2], [2,6],
    [6,3], [3,4], [4,5],
    [6,7], [7,8], [8,9],
    [7,12], [12,11], [11,10],
    [7,13], [13,14], [14,15]
]

CocoColors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85]
]

NUM_KPTS = 16
SAMPLE_PERIOD = 0.5

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_joint_metrics(pred_points, gt_points):
    errors = []
    for p, g in zip(pred_points, gt_points):
        err = np.linalg.norm(np.array(p) - np.array(g))
        errors.append(err)
    return errors


def draw_pose(keypoints, image):
    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    if keypoints.shape[1] == 3:
        keypoints = keypoints[:, :2]

    assert keypoints.shape == (NUM_KPTS, 2)

    for a, b in SKELETON:
        x_a, y_a = keypoints[a]
        x_b, y_b = keypoints[b]

        cv2.circle(image, (int(x_a), int(y_a)), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(x_b), int(y_b)), 5, (0, 255, 0), -1)
        cv2.line(
            image,
            (int(x_a), int(y_a)),
            (int(x_b), int(y_b)),
            (255, 0, 0),
            2
        )


def draw_bbox(box, img):
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [
        COCO_INSTANCE_CATEGORY_NAMES[i]
        for i in list(pred[0]['labels'].cpu().numpy())
    ]
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]['boxes'].detach().cpu().numpy())
    ]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    if not pred_score or max(pred_score) < threshold:
        return []

    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    model_input = transform(model_input).unsqueeze(0)
    pose_model.eval()

    with torch.no_grad():
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale])
        )

    return preds
