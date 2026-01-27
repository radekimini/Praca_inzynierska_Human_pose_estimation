import os
import shutil
import pandas as pd

processing_csv = "processing/frame_coordinates_subset.csv"
processing_images = "processing/images"

test_data_root = "test_data"
test_data_images = os.path.join(test_data_root, "images")
test_data_csv = os.path.join(test_data_root, "frame_coordinates.csv")

os.makedirs(test_data_images, exist_ok=True)

if not os.path.exists(processing_csv):
    raise FileNotFoundError(f"Processing CSV not found: {processing_csv}")

new_df = pd.read_csv(processing_csv)

if os.path.exists(test_data_csv):
    print(f"Found existing CSV, merging: {test_data_csv}")
    existing_df = pd.read_csv(test_data_csv)
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    print("No existing CSV found, creating new one.")
    merged_df = new_df

if 'filename' in merged_df.columns:
    merged_df.drop_duplicates(subset='filename', keep='last', inplace=True)

merged_df.to_csv(test_data_csv, index=False)
print(f"Merged CSV saved to {test_data_csv}")

copied = 0
for filename in new_df['filename']:
    src = os.path.join(processing_images, filename)
    dst = os.path.join(test_data_images, filename)

    if not os.path.exists(src):
        print(f"Missing image: {filename}")
        continue

    shutil.copy2(src, dst)
    copied += 1

print(f"Copied {copied} images into {test_data_images}")
print("Merge completed successfully")
