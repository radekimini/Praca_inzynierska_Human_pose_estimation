import pandas as pd
import os
import cv2

images_folder = 'augmented_new_cropped_images_128'
cropped_csv = 'augmented_new_keypoints_128.csv'
output_folder_annotated_cropped = 'annotated_new_cropped_images_128'

os.makedirs(output_folder_annotated_cropped, exist_ok=True)

df_cropped = pd.read_csv(cropped_csv)

connections = [
    (0, 1), (1, 2), (2, 6), (6, 7),
    (5, 4), (4, 3), (3, 6),
    (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]

def annotate_and_save(df, output_folder, image_folder):
    grouped = df.groupby('image_name')

    for img_name, group in grouped:
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        points = {}
        for _, row in group.iterrows():
            x, y = int(row['x']), int(row['y'])
            joint_id = int(row['joint_id'])
            points[joint_id] = (x, y)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(
                image,
                str(joint_id),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        for p1, p2 in connections:
            if p1 in points and p2 in points:
                cv2.line(image, points[p1], points[p2], (0, 255, 0), 2)

        name, ext = os.path.splitext(img_name)
        output_path = os.path.join(output_folder, f"{name}_ad{ext}")
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")

annotate_and_save(df_cropped, output_folder_annotated_cropped, images_folder)
