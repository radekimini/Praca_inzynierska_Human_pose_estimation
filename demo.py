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
POINT_CONSUMER = None
from pathlib import Path


# ============================
# HARD-CODED INPUT CONFIG
# ============================



INPUT_TYPE = "webcam"   # "video" | "webcam" | "image" | "folder

CFG_PATH = "experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml"
VIDEO_PATH = "demo/Take8_Calibratedc.mp4"
IMAGE_PATH = None      # only if INPUT_TYPE == "image"
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
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
CALIB_CFG = {
    'camera_matrix': np.array([
        [659.1335331, 0., 309.26599601],
        [0., 658.13176293, 225.12565591],
        [0., 0., 1.]
    ], dtype=np.float32),

    'dist_matrix': np.array([
        [1.20261687e-01, -7.89795463e-01, 2.13838781e-03, -2.47974967e-05, 1.03624635e+00]
    ], dtype=np.float32),

    'new_camera_matrix': np.array([
        [651.0860892, 0., 309.02327403],
        [0., 650.56864018, 225.98281995],
        [0., 0., 1.]
    ], dtype=np.float32),

    'roi': (3, 4, 632, 471)  
}

SKELETON = [
    [0,1], [1,2], [2,6],        # prawa noga
    [6,3], [3,4], [4,5],        # lewa noga
    [6,7], [7,8], [8,9],        # kręgosłup
    [7,12], [12,11], [11,10],   # prawa ręka
    [7,13], [13,14], [14,15]    # lewa ręka
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 16

SAMPLE_PERIOD = 0.5


CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_joint_metrics(pred_points, gt_points):

    errors = []
    for p, g in zip(pred_points, gt_points):
        err = np.linalg.norm(np.array(p) - np.array(g))
        errors.append(err)
    return errors

def calibrate_frame(frame_bgr):

    frame = cv2.undistort(
        frame_bgr,
        CALIB_CFG['camera_matrix'],
        CALIB_CFG['dist_matrix'],
        None,
        CALIB_CFG['new_camera_matrix']
    )

    x, y, w, h = CALIB_CFG['roi']
    frame = frame[y:y + h, x:x + w]

    return frame

def draw_pose(keypoints, image):

    # obsługa (N,16,3) / (16,3) / (16,2)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    if keypoints.shape[1] == 3:
        keypoints = keypoints[:, :2]

    assert keypoints.shape == (NUM_KPTS, 2)

    for i, (a, b) in enumerate(SKELETON):
        x_a, y_a = keypoints[a]
        x_b, y_b = keypoints[b]

        cv2.circle(image, (int(x_a), int(y_a)), 5, (0,255,0), -1)
        cv2.circle(image, (int(x_b), int(y_b)), 5, (0,255,0), -1)
        cv2.line(image,
                 (int(x_a), int(y_a)),
                 (int(x_b), int(y_b)),
                 (255,0,0), 2)


def draw_bbox(box,img):

    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

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
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model_input = transform(model_input).unsqueeze(0)
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def run_folder_mode(
    image_dir,
    csv_path,
    pose_model,
    box_model,
    draw=True,
    save_vis_dir=None
):
    df = pd.read_csv(csv_path)

    joint_errors = defaultdict(list)

    if save_vis_dir:
        os.makedirs(save_vis_dir, exist_ok=True)

    for _, row in df.iterrows():
        img_name = row["image"]
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            continue

        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        gt_points = []
        for j in range(16):
            gt_points.append((row[f"x{j}"], row[f"y{j}"]))

        img_tensor = torch.from_numpy(image_rgb / 255.).permute(2, 0, 1).float().to(CTX)
        pred_boxes = get_person_detection_boxes(box_model, [img_tensor], threshold=0.9)

        if len(pred_boxes) == 0:
            continue

        box = pred_boxes[0]
        center, scale = box_to_center_scale(
            box,
            cfg.MODEL.IMAGE_SIZE[0],
            cfg.MODEL.IMAGE_SIZE[1]
        )

        preds = get_pose_estimation_prediction(
            pose_model,
            image_rgb,
            center,
            scale
        )

        if len(preds) == 0:
            continue

        pred_points = extract_mpii_points(preds[0])

        errors = compute_joint_metrics(pred_points, gt_points)
        for j, err in enumerate(errors):
            joint_errors[j].append(err)

        if draw:
            for (px, py), (gx, gy) in zip(pred_points, gt_points):
                cv2.circle(image_bgr, (int(px), int(py)), 4, (0, 255, 0), -1)
                cv2.circle(image_bgr, (int(gx), int(gy)), 4, (0, 0, 255), -1)

            if save_vis_dir:
                cv2.imwrite(
                    os.path.join(save_vis_dir, img_name),
                    image_bgr
                )

    return joint_errors


def box_to_center_scale(box, model_image_width, model_image_height):
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def set_point_consumer(fn):
    global POINT_CONSUMER
    POINT_CONSUMER = fn

def extract_mpii_points(kpt):
    if kpt.ndim == 3:
        kpt = kpt[0]

    if kpt.shape[1] >= 2:
        kpt = kpt[:, :2]

    points = [(float(x), float(y)) for x, y in kpt]

    assert len(points) == 16, "Expected 16 MPII keypoints"

    return points


def main(queue_frames_skeleton=None, queue_frames_visual=None):
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    update_config(
        cfg,
        SimpleNamespace(
            cfg=CFG_PATH,
            opts=[],
            modelDir='',
            logDir='',
            dataDir='',
            prevModelDir=''
        )
    )

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(
        pose_model,
        device_ids=[cfg.GPUS]
    )
    pose_model.to(CTX)
    pose_model.eval()

    if INPUT_TYPE == "webcam":
        vidcap = cv2.VideoCapture(WEBCAM_INDEX)
    elif INPUT_TYPE == "video":
        vidcap = cv2.VideoCapture(VIDEO_PATH)
    elif INPUT_TYPE == "image":
        image_bgr = cv2.imread(IMAGE_PATH)
        vidcap = None
    elif INPUT_TYPE == "folder":
        joint_errors = run_folder_mode(
        image_dir="demo/test_data/test_images",
        csv_path="demo/test_data/test_keypoints.csv",
        pose_model=pose_model,
        box_model=box_model,
        draw=True,
        save_vis_dir="output_vis"
    )
        summarize_and_plot(joint_errors)
    else:
        raise ValueError("Invalid INPUT_TYPE")
    next_sample_time = time.monotonic()

    if INPUT_TYPE in ("webcam", "video"):
        if WRITE_VIDEO:
            save_path = 'output.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path,fourcc, 24.0, (int(vidcap.get(3)),int(vidcap.get(4))))
        while True:
            now = time.monotonic()
            pipeline_start = time.perf_counter()
            if now < next_sample_time:
                time.sleep(next_sample_time - now)

            next_sample_time += SAMPLE_PERIOD

            for _ in range(5):
                vidcap.grab()

            ret, image_bgr = vidcap.read()
            if not ret:
                print("cannot load the video.")
                break
            image_bgr = calibrate_frame(image_bgr)
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            img_tensor = torch.from_numpy(image_rgb / 255.) \
                .permute(2, 0, 1).float().to(CTX)

            input = [img_tensor]

            pred_boxes = get_person_detection_boxes(
                box_model, input, threshold=0.9
            )

            if len(pred_boxes) >= 1:
                box = pred_boxes[0]

                center, scale = box_to_center_scale(
                    box,
                    cfg.MODEL.IMAGE_SIZE[0],
                    cfg.MODEL.IMAGE_SIZE[1]
                )

                image_pose = image_rgb.copy()

                pose_preds = get_pose_estimation_prediction(
                    pose_model, image_pose, center, scale
                )

                if len(pose_preds) >= 1:
                    points = extract_mpii_points(pose_preds[0])

                    packet = {
                        "id": time.time_ns(),
                        "time": time.time(),
                        "frame": image_bgr,
                        "angles": [0],
                        "points": np.array(points)
                    }
                    send_time = time.time()
                    send_time_ns = time.time_ns()
                    pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000
                    for q in (queue_frames_skeleton, queue_frames_visual):
                        if q is None:
                            continue
                        while not q.empty():
                            q.get()
                        q.put(packet)

                        print(
                            f"[SEND] ts={send_time:.6f} "
                            f"latency={pipeline_latency_ms:.2f} ms "
                            f"period={SAMPLE_PERIOD:.2f}s"
                        )

            if SHOW_WINDOW:
                cv2.imshow("demo", image_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        vidcap.release()
        if WRITE_VIDEO:
            print('video has been saved as {}'.format(save_path))
            out.release()

    else:
        last_time = time.time()
        image = image_bgr[:, :, [2, 1, 0]]

        input = []
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
        input.append(img_tensor)

        pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

        if len(pred_boxes) >= 1:
            for box in pred_boxes:
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                if len(pose_preds)>=1:
                    for kpt in pose_preds:
                        draw_pose(kpt,image_bgr) # draw the poses
        
        if SHOW_FPS:
            fps = 1/(time.time()-last_time)
            img = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        if WRITE_VIDEO:
            save_path = 'output.jpg'
            cv2.imwrite(save_path,image_bgr)
            print('the result image has been saved as {}'.format(save_path))

        cv2.imshow('demo',image_bgr)
        if cv2.waitKey(0) & 0XFF==ord('q'):
            cv2.destroyAllWindows()


def summarize_and_plot(joint_errors):
    summary = []

    for j in range(16):
        errs = np.array(joint_errors[j])
        mean_err = errs.mean()
        acc = (errs <= PIXEL_THRESHOLD).mean() * 100

        summary.append((JOINT_NAMES[j], mean_err, acc))

    summary.sort(key=lambda x: x[2], reverse=True)

    print("\n=== JOINT ACCURACY SUMMARY ===\n")
    for name, err, acc in summary:
        print(f"{name:<12} | {err:6.2f}px | acc≤10px  {acc:5.1f}%")

    names = [s[0] for s in summary]
    accs = [s[2] for s in summary]

    plt.figure(figsize=(12, 6))
    plt.bar(names, accs)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy ≤ 10px [%]")
    plt.title("Joint Accuracy")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
