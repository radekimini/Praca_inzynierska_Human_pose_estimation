import os
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import math
import time
import random

image_dir = r'G:\Programowanie\DeepLearning\Plank\mpii_human_pose_v1\images'
csv_path = 'body_positions_complete_single_person.csv'
augmented_csv_path = 'augmented_new_keypoints_128.csv'
augmented_image_dir = 'augmented_new_cropped_images_128'
output_size = 128
buffer = 50
rotation_angles = [-30, 30]
sample_size = None

try:
    resample_method = Image.Resampling.LANCZOS
except AttributeError:
    resample_method = Image.ANTIALIAS

os.makedirs(augmented_image_dir, exist_ok=True)
if os.path.exists(augmented_csv_path):
    os.remove(augmented_csv_path)

def flip_horizontal(image, keypoints):
    w = image.width
    new_kps = keypoints.copy()
    new_kps[:, 0] = w - 1 - new_kps[:, 0]
    return image.transpose(Image.FLIP_LEFT_RIGHT), new_kps

def flip_vertical(image, keypoints):
    h = image.height
    new_kps = keypoints.copy()
    new_kps[:, 1] = h - 1 - new_kps[:, 1]
    return image.transpose(Image.FLIP_TOP_BOTTOM), new_kps

def adjust_brightness(image, keypoints, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor), keypoints

def adjust_sharpness(image, keypoints, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor), keypoints

def blur_image(image, keypoints):
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(arr, (9, 9), 0)
    return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)), keypoints

transformations = [
    ('hflip', flip_horizontal),
    ('vflip', flip_vertical),
    ('darker', lambda img, kp: adjust_brightness(img, kp, 0.6)),
    ('brighter', lambda img, kp: adjust_brightness(img, kp, 1.4)),
    ('sharper', lambda img, kp: adjust_sharpness(img, kp, 2.0)),
    ('blurred', blur_image),
]

def rotate_image_and_keypoints(image, keypoints, angle):
    w, h = image.size
    center = (w / 2, h / 2)
    rotated_image = image.rotate(angle, expand=True, fillcolor=(0, 0, 0))

    rad = math.radians(-angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    new_kps = []
    for x, y in keypoints:
        x0, y0 = x - center[0], y - center[1]
        x_new = x0 * cos_a - y0 * sin_a + rotated_image.size[0] / 2
        y_new = x0 * sin_a + y0 * cos_a + rotated_image.size[1] / 2
        new_kps.append((x_new, y_new))

    return rotated_image, np.array(new_kps)

def crop_and_save(image, keypoints, image_name, output_dir, csv_list):
    min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
    min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()

    size = max(max_x - min_x, max_y - min_y)
    final_size = size + buffer * 2
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    crop_left = max(0, int(center_x - final_size / 2))
    crop_top = max(0, int(center_y - final_size / 2))
    crop_right = min(image.width, crop_left + int(final_size))
    crop_bottom = min(image.height, crop_top + int(final_size))

    crop_box = (crop_left, crop_top, crop_right, crop_bottom)
    cropped_region = image.crop(crop_box)

    canvas = Image.new(
        "RGB",
        (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
        (0, 0, 0)
    )
    canvas.paste(cropped_region)

    resized = canvas.resize((output_size, output_size), resample_method)
    resized.save(os.path.join(output_dir, image_name))

    scale_x = output_size / (crop_box[2] - crop_box[0])
    scale_y = output_size / (crop_box[3] - crop_box[1])

    for joint_id, (x, y) in enumerate(keypoints):
        is_visible = int(
            (crop_box[0] <= x <= crop_box[2] - 1) and
            (crop_box[1] <= y <= crop_box[3] - 1)
        )
        new_x = (x - crop_box[0]) * scale_x
        new_y = (y - crop_box[1]) * scale_y

        csv_list.append({
            'image_name': image_name,
            'joint_id': joint_id,
            'x': new_x,
            'y': new_y,
            'is_visible': is_visible
        })

df = pd.read_csv(csv_path)
grouped = df.groupby('image_name')
processed_count = 0
total_images = sample_size if sample_size is not None else len(grouped)
start_time = time.time()

for image_name, group in grouped:
    if sample_size is not None and processed_count >= sample_size:
        break

    t0 = time.time()

    img_path = os.path.join(image_dir, image_name)
    if not os.path.exists(img_path):
        print(f"Missing image: {image_name}")
        continue

    image = Image.open(img_path).convert("RGB")
    keypoints = group.sort_values('joint_id')[['x', 'y']].to_numpy()

    variants = [(image_name.replace('.jpg', '_orig.jpg'), image, keypoints)]
    
    angle = random.uniform(rotation_angles[0], rotation_angles[1])

    rotated_img, rotated_kp = rotate_image_and_keypoints(image, keypoints, angle)
    variants.append(
        (image_name.replace('.jpg', f'_rot{angle:.2f}.jpg'), rotated_img, rotated_kp)
    )

    aug_variants = []
    for name, img_var, kp_var in variants:
        for suffix, transform in transformations:
            aug_img, aug_kp = transform(img_var, kp_var)
            aug_name = name.replace('.jpg', f'_{suffix}.jpg')
            aug_variants.append((aug_name, aug_img, aug_kp))
    variants.extend(aug_variants)

    temp_data = []
    for new_name, var_img, var_kp in variants:
        crop_and_save(var_img, var_kp, new_name, augmented_image_dir, temp_data)

    pd.DataFrame(temp_data).to_csv(
        augmented_csv_path,
        mode='a',
        index=False,
        header=not os.path.exists(augmented_csv_path) or processed_count == 0
    )

    processed_count += 1
    t1 = time.time()
    elapsed = t1 - start_time
    avg_time = elapsed / processed_count
    remaining = total_images - processed_count
    eta = avg_time * remaining

    print(
        f"{processed_count}/{total_images} {image_name} completed "
        f"(time: {t1 - t0:.2f}s, ETA: {eta:.1f}s)"
    )

total_time = time.time() - start_time
print(f"Saved augmented keypoints to {augmented_csv_path}")
print(f"Total time: {total_time:.1f} seconds")

