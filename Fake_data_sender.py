import time
import numpy as np
import cv2
from shared import read_config, setup_logger

logger = setup_logger(__name__)

class Fake_data_sender:
    def __init__(self, path="calculated_points/projected_points.json", video_path="calculated_points/video.mp4"):
        try:
            self.target_duration = 0.05
            self.video_path = video_path

            self.points = read_config(path)
            fixed_data = []
            for frame in self.points:
                frame_fixed = frame.copy()
                while len(frame_fixed) < 16:
                    frame_fixed.append([np.nan, np.nan])
                fixed_data.append(frame_fixed)
            self.skeleton_calculated = np.array(fixed_data)
        except Exception as e:
            logger.error(f"Fake data sender cannot init: {e}")

    def fake_data_symulator(self, queue_joints, queue_visual):
        video = cv2.VideoCapture(self.video_path)

        if not video.isOpened():
            logger.error("Cannot open video file")
            return

        for index, points in enumerate(self.skeleton_calculated):
            try:
                start = time.perf_counter()

                # Ustaw pozycję i wczytaj konkretną klatkę
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = video.read()
                if not ret or frame is None:
                    logger.warning(f"Video frame {index} read failed")
                    break

                queue_joints.put(points)
                queue_visual.put((frame, points))

                elapsed = time.perf_counter() - start
                remaining = self.target_duration - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            except Exception as e:
                logger.error(f"Error queueing data in faker in frame {index}: {e}")




