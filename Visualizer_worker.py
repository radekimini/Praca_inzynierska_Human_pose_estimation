import cv2
import math
import numpy as np
from shared import setup_logger, Joint

logger = setup_logger(__name__)


class Visualizer_worker:
    """
    Skalowalny wizualizator kątów szkieletu.
    Każdy kąt opisany jest przez konfigurację (a, b, c, label).
    """

    def __init__(self):

        self.angle_defs = [
            # łokcie
            dict(label="L_ELBOW",
                 a=Joint.L_HAND, b=Joint.L_ELBOW, c=Joint.L_SHOULDER,
                 color=(0, 150, 255), radius_scale=0.5),

            dict(label="R_ELBOW",
                 a=Joint.R_HAND, b=Joint.R_ELBOW, c=Joint.R_SHOULDER,
                 color=(0, 150, 255), radius_scale=0.5),

            # barki
            dict(label="L_ARM",
                 a=Joint.C_HIP, b=Joint.C_SHOULDER, c=Joint.L_HAND,
                 color=(60, 200, 60), radius_scale=0.4),

            dict(label="R_ARM",
                 a=Joint.C_HIP, b=Joint.C_SHOULDER, c=Joint.R_HAND,
                 color=(60, 200, 60), radius_scale=0.4),

            # nogi
            dict(label="LEGS",
                 a=Joint.L_FOOT, b=Joint.C_HIP, c=Joint.R_FOOT,
                 color=(200, 60, 200), radius_scale=0.6),
        ]


    def segment_angle_deg(self, center_pt, tip_pt):
        """
        Kąt wektora center -> tip w stopniach
        """
        if center_pt is None or tip_pt is None:
            return None
        dx = float(tip_pt[0]) - float(center_pt[0])
        dy = float(tip_pt[1]) - float(center_pt[1])
        return math.degrees(math.atan2(-dy, dx))

    def angle_diff_signed(self, a_deg, b_deg):
        """
        Najmniejszy podpisany obrót z a do b: (-180, 180]
        """
        if a_deg is None or b_deg is None:
            return None
        return (b_deg - a_deg + 180) % 360 - 180

    def _linspace_angles(self, start_deg, sweep_deg, n):
        if n <= 1:
            return [start_deg]
        return [start_deg + t * sweep_deg for t in np.linspace(0.0, 1.0, n)]

    def draw_angle_sector(self, frame, center, ang1, ang2,
                          radius=30, color=(0, 128, 255),
                          alpha=0.35, n_points=36):

        if center is None or ang1 is None or ang2 is None:
            return

        cx, cy = int(center[0]), int(center[1])
        sweep = self.angle_diff_signed(ang1, ang2)

        if sweep is None or abs(sweep) < 1e-3:
            cv2.circle(frame, (cx, cy), max(3, int(radius * 0.15)), color, -1)
            return

        angles = self._linspace_angles(ang1, sweep, n_points)
        pts = [
            (
                int(cx + radius * math.cos(math.radians(a))),
                int(cy - radius * math.sin(math.radians(a)))
            )
            for a in angles
        ]

        poly = np.array([(cx, cy)] + pts + [(cx, cy)], dtype=np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [poly], True, (0, 0, 0), 1, cv2.LINE_AA)


    def draw_angle(self, frame, points, angles, cfg):
        a = points[cfg["a"].value]
        b = points[cfg["b"].value]
        c = points[cfg["c"].value]

        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
            return

        ang1 = self.segment_angle_deg(b, a)
        ang2 = self.segment_angle_deg(b, c)

        d1 = math.hypot(a[0] - b[0], a[1] - b[1])
        d2 = math.hypot(c[0] - b[0], c[1] - b[1])
        radius = int(max(14, min(d1, d2) * cfg["radius_scale"]))

        self.draw_angle_sector(
            frame,
            b,
            ang1,
            ang2,
            radius=radius,
            color=cfg["color"]
        )

        angle_val = angles.get(cfg["label"]) if angles else None
        if angle_val is None:
            return

        sweep = self.angle_diff_signed(ang1, ang2)
        bisector = ang1 + (sweep / 2 if sweep else 0)

        tx = int(b[0] + radius * 1.25 * math.cos(math.radians(bisector)))
        ty = int(b[1] - radius * 1.25 * math.sin(math.radians(bisector)))

        txt = f"{round(angle_val, 1)}°"
        cv2.putText(frame, txt, (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


    def draw(self, frame, points, angles):

        # punkty szkieletu
        for pt in points:
            if not np.any(np.isnan(pt)):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

        # wszystkie kąty z konfiguracji
        for cfg in self.angle_defs:
            self.draw_angle(frame, points, angles, cfg)

        cv2.imshow("Skeleton Visualization", frame)
        cv2.waitKey(1)

    def run(self, queue_frames, queue_angles):
        logger.info("Visualizer started")
        angles = {}
        while True:
            try:
                packet = queue_frames.get()

                while not queue_angles.empty():
                    angles = queue_angles.get()

                self.draw(packet["frame"], packet["points"], angles)

            except Exception as e:
                logger.error(f"Visualizer error: {e}")


