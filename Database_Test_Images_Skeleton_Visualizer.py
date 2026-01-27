import os
import math
import pandas as pd
import cv2
import numpy as np

image_folder = r'G:\Programowanie\DeepLearning\Plank\DeepLearning\Kody\test_data\test_images'
csv_path = r'G:\Programowanie\DeepLearning\Plank\DeepLearning\Kody\test_data\test_keypoints.csv'
output_folder = 'images_for_processing'
os.makedirs(output_folder, exist_ok=True)

connections = [
    (0, 1), (1, 2), (2, 6), (6, 7),
    (5, 4), (4, 3), (3, 6),
    (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]

df = pd.read_csv(csv_path)
num_points = 16

for idx, row in df.iterrows():
    filename = row['filename']
    img_path = os.path.join(image_folder, filename)

    if not os.path.exists(img_path):
        print(f"Missing image: {filename}")
        continue

    coords = []
    has_valid = False
    for i in range(1, num_points + 1):
        x = row[f'x{i}']
        y = row[f'y{i}']
        if not (math.isnan(x) or math.isnan(y)):
            has_valid = True
        coords.append((x, y))

    if not has_valid:
        print(f"Skipping {filename} (all NaN)")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {filename}")
        continue

    for a, b in connections:
        if a < num_points and b < num_points:
            x1, y1 = coords[a]
            x2, y2 = coords[b]
            if not (math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2)):
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    for i, (x, y) in enumerate(coords):
        if math.isnan(x) or math.isnan(y):
            continue
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(
            img,
            str(i + 1),
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

print("All frames processed successfully")
