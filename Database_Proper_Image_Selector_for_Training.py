import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'CSV', 'body_positions.csv')
output_path = os.path.join(script_dir, '..', 'CSV', 'body_positions_filtered.csv')

df = pd.read_csv(csv_path)

df['is_visible'] = df['is_visible'].fillna(0)

grouped = df.groupby('image_name')
i = 0
valid_rows = []

for img_name, group in grouped:
    i += 1
    print(i)

    unique_joint_ids = set(group['joint_id'])
    visible_count = (group['is_visible'] == 1).sum()

    has_enough_visible = visible_count >= 14
    is_single_person = len(group['joint_id'].unique()) == len(group)
    has_complete_joints = set(range(16)).issubset(unique_joint_ids)

    if has_enough_visible and is_single_person and has_complete_joints:
        valid_rows.append(group)

if valid_rows:
    final_df = pd.concat(valid_rows)
    final_df.to_csv(output_path, index=False)
    print(f"Saved file: {output_path}")
else:
    print("No data meeting all conditions")
