import os
import shutil
import pandas as pd
import numpy as np

image_folder = r'G:\Programowanie\DeepLearning\Plank\DeepLearning\Vol_6\Vol_6\Radzio17_KibicG\frames_radzio17'
csv_path = r'G:\Programowanie\DeepLearning\Plank\DeepLearning\Vol_6\Vol_6\Radzio17_KibicG\frame_coordinates_radzio17.csv'
output_root = "processing"
output_image_folder = os.path.join(output_root, "images")
output_csv_path = os.path.join(output_root, "frame_coordinates_subset.csv")

os.makedirs(output_image_folder, exist_ok=True)

df = pd.read_csv(csv_path)

coord_cols = [c for c in df.columns if c.startswith('x') or c.startswith('y')]

df_valid = df.dropna(subset=coord_cols, how='all').reset_index(drop=True)

subset_df = df_valid.iloc[::5].reset_index(drop=True)

copied_count = 0
for _, row in subset_df.iterrows():
    filename = row['filename']
    src = os.path.join(image_folder, filename)
    dst = os.path.join(output_image_folder, filename)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied_count += 1
        print(f"Copied: {filename}")
    else:
        print(f"Missing image: {filename}")

subset_df.to_csv(output_csv_path, index=False)
print(f"Done. Saved {len(subset_df)} entries to {output_csv_path}")
print(f"Copied {copied_count} images into {output_image_folder}")
