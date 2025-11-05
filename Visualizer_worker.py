import cv2
import numpy as np
from shared import setup_logger, Joint

logger = setup_logger(__name__)


class Visualizer_worker:
    def __init__(self):
        self.skeleton_lines = [
            (Joint.R_FOOT, Joint.R_KNEE), (Joint.R_KNEE, Joint.R_HIP), (Joint.R_HIP, Joint.C_HIP),
            (Joint.L_HIP, Joint.C_HIP), (Joint.L_HIP, Joint.L_KNEE), (Joint.L_KNEE, Joint.L_FOOT),
            (Joint.C_HIP, Joint.C_SHOULDER), (Joint.C_SHOULDER, Joint.NECK), (Joint.NECK, Joint.HEAD),
            (Joint.C_SHOULDER, Joint.R_SHOULDER), (Joint.R_SHOULDER, Joint.R_ELBOW), (Joint.R_ELBOW, Joint.R_HAND),
            (Joint.C_SHOULDER, Joint.L_SHOULDER), (Joint.L_SHOULDER, Joint.L_ELBOW), (Joint.L_ELBOW, Joint.L_HAND)
        ]

    def draw(self, frame, points, angles):
        # Rysowanie punktów
        for i, pt in enumerate(points):
            if np.isnan(pt[0]) or np.isnan(pt[1]):
                continue
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Rysowanie połączeń
        for idx1, idx2 in self.skeleton_lines:
            p1 = points[idx1.value]
            p2 = points[idx2.value]
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Rysowanie kątów
        y_offset = 20
        for key, angle in angles.items():
            text = f"{key}: {round(angle, 1)}°"
            cv2.putText(frame, text, (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        cv2.imshow("Skeleton Visualization", frame)
        cv2.waitKey(1)

    def run(self, queue_visual, queue_angles):
        logger.info("Visualizer started")
        angles = {}

        while True:
            try:
                if not queue_visual.empty():
                    frame, points = queue_visual.get()
                    if not queue_angles.empty():
                        angles = queue_angles.get()
                    self.draw(frame, points, angles)
            except Exception as e:
                logger.error(f"Visualizer error: {e}")
