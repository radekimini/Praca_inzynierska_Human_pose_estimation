import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image

INPUT_CSV = "test_data/test_data_keypoints.csv"
IMAGES_DIR = "test_data/test_data_images"
OUTPUT_DIR = "test_data/test_images"
OUTPUT_CSV = "test_data/test_keypoints.csv"
OUTPUT_SIZE = 256
BUFFER = 50
RESAMPLE = Image.BICUBIC

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
keypoint_cols = [c for c in df.columns if c != "filename"]
num_keypoints = len(keypoint_cols) // 2

cropped_rows = []

for idx, row in df.iterrows():
    filename = row["filename"]
    img_path = os.path.join(IMAGES_DIR, filename)

    if not os.path.exists(img_path):
        print(f"Missing image: {filename}")
        continue

    keypoints = np.array(
        [[row[f"x{i+1}"], row[f"y{i+1}"]] for i in range(num_keypoints)],
        dtype=np.float32
    )

    if np.isnan(keypoints).any():
        print(f"Skipping {filename} due to NaN coordinates")
        continue

    image = Image.open(img_path).convert("RGB")
    width, height = image.size

    min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
    min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()

    box_size = max(max_x - min_x, max_y - min_y)
    final_size = box_size + BUFFER * 2

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    crop_left = int(center_x - final_size / 2)
    crop_top = int(center_y - final_size / 2)
    crop_right = crop_left + int(final_size)
    crop_bottom = crop_top + int(final_size)

    pad_left = max(0, -crop_left)
    pad_top = max(0, -crop_top)
    pad_right = max(0, crop_right - width)
    pad_bottom = max(0, crop_bottom - height)

    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(width, crop_right)
    crop_bottom = min(height, crop_bottom)

    cropped_region = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    canvas_size = (
        crop_right - crop_left + pad_left + pad_right,
        crop_bottom - crop_top + pad_top + pad_bottom
    )
    canvas = Image.new("RGB", canvas_size, (0, 0, 0))
    canvas.paste(cropped_region, (pad_left, pad_top))

    resized = canvas.resize((OUTPUT_SIZE, OUTPUT_SIZE), RESAMPLE)

    scale_x = OUTPUT_SIZE / canvas_size[0]
    scale_y = OUTPUT_SIZE / canvas_size[1]

    new_row = {"filename": f"crop_{filename}"}
    for i, (x, y) in enumerate(keypoints):
        new_x = (x - crop_left + pad_left) * scale_x
        new_y = (y - crop_top + pad_top) * scale_y
        new_row[f"x{i+1}"] = new_x
        new_row[f"y{i+1}"] = new_y

    cropped_rows.append(new_row)

    out_path = os.path.join(OUTPUT_DIR, f"crop_{filename}")
    resized.save(out_path)

out_df = pd.DataFrame(cropped_rows)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"Done. Cropped images saved to '{OUTPUT_DIR}', CSV saved to '{OUTPUT_CSV}'")
