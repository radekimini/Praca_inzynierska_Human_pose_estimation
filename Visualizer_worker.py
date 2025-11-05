import cv2
import math
import numpy as np
from shared import setup_logger, Joint

logger = setup_logger(__name__)


class Visualizer_worker:
    """
    Wizualizator rysujący tylko ramiona:
      - sektory przy łokciach: (HAND, ELBOW, SHOULDER)
      - sektory przy barkach: (C_HIP, C_SHOULDER, SHOULDER->HAND)
    Zakłada, że `points` to lista punktów [(x,y), ...] i `angles` to dict z kluczami:
      "L_ELBOW", "R_ELBOW", "L_ARM", "R_ARM"
    """

    def __init__(self):
        # trójki do rysowania jako sektory: (pA, pB, pC, label)
        # pB jest środkiem (węzłem), sektor rysujemy wokół pB między wektorem pB->pA i pB->pC
        self.elbow_triplets = [
            (Joint.L_HAND, Joint.L_ELBOW, Joint.L_SHOULDER, "L_ELBOW"),
            (Joint.R_HAND, Joint.R_ELBOW, Joint.R_SHOULDER, "R_ELBOW"),
        ]
        self.shoulder_triplets = [
            (Joint.C_HIP, Joint.C_SHOULDER, Joint.L_HAND, "L_ARM"),  # ciało vs lewy ramie
            (Joint.C_HIP, Joint.C_SHOULDER, Joint.R_HAND, "R_ARM"),  # ciało vs prawy ramie
        ]

    # --------------------- funkcje pomocnicze ---------------------
    def segment_angle_deg(self, center_pt, tip_pt):
        """
        Orientacja wektora center -> tip w stopniach (matematyczne: 0° = w prawo, + = CCW).
        Kompensacja: -dy bo w obrazach oś Y rośnie DO DOŁU.
        Zwraca -> (-180, 180] lub None jeśli brak danych.
        """
        if center_pt is None or tip_pt is None:
            return None
        dx = float(tip_pt[0]) - float(center_pt[0])
        dy = float(tip_pt[1]) - float(center_pt[1])
        angle_rad = math.atan2(-dy, dx)   # -dy: zamiana układu pikselowego na matematyczny
        angle_deg = math.degrees(angle_rad)
        if angle_deg <= -180:
            angle_deg += 360
        if angle_deg > 180:
            angle_deg -= 360
        return angle_deg

    def angle_diff_signed(self, a_deg, b_deg):
        """
        Najmniejszy podpisany obrót z a_deg do b_deg w stopniach -> (-180,180]
        """
        if a_deg is None or b_deg is None:
            return None
        return (b_deg - a_deg + 180) % 360 - 180

    def _linspace_angles(self, start_deg, sweep_deg, n):
        """Generuje n kątów od start do start+sweep (w stopniach)."""
        if n <= 1:
            return [start_deg]
        return [start_deg + t * sweep_deg for t in np.linspace(0.0, 1.0, n)]

    def draw_angle_sector(self, frame, center, ang1_deg, ang2_deg,
                          radius=30, color=(0, 128, 255), thickness=2,
                          filled=True, alpha=0.4, n_points=36):
        """
        Rysuje półprzezroczysty sektor (wypełniony) lub kontur przy center:
        - center: (x,y)
        - ang1_deg, ang2_deg: kąty matematyczne (-180..180)
        - radius: promień sektora w px
        - color: BGR
        - filled: jeśli True -> wypełniony półprzezroczystością (alpha)
        - n_points: aproksymacja łuku
        """
        if center is None or ang1_deg is None or ang2_deg is None:
            return
        cx, cy = int(round(center[0])), int(round(center[1]))
        sweep = self.angle_diff_signed(ang1_deg, ang2_deg)
        if sweep is None:
            return
        if abs(sweep) < 1e-3:
            # niemal prosty (brak wyraźnego zgięcia) -> mały kółek
            cv2.circle(frame, (cx, cy), max(2, int(radius*0.12)), color, -1)
            return

        angles = self._linspace_angles(ang1_deg, sweep, n_points)
        pts = []
        for a_deg in angles:
            a_rad = math.radians(a_deg)
            x = cx + radius * math.cos(a_rad)
            y = cy - radius * math.sin(a_rad)  # minus, bo y obrazu w dół
            pts.append((int(round(x)), int(round(y))))

        # polygon do wypełnienia: [center] + pts + [center]
        poly = np.array([(cx, cy)] + pts + [(cx, cy)], dtype=np.int32)

        if filled:
            # rysujemy na overlay, by uzyskać półprzezroczystość
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            # blend - nakładamy overlay z alfa
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # opcjonalnie kontur:
            cv2.polylines(frame, [poly], isClosed=True, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
        else:
            # rysuj jedynie kontur łuku i promienie
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.line(frame, (cx, cy), pts[0], color, thickness=thickness)
            cv2.line(frame, (cx, cy), pts[-1], color, thickness=thickness)

    # --------------------- główna funkcja rysująca ---------------------
    def draw(self, frame, points, angles):
        """
        Rysuje punkty, linie szkieletu (opcjonalnie) i sektory tylko dla ramion.
        - points: lista punktów [(x,y), ...] (mogą być np.nan)
        - angles: słownik z wartościami kątów (np. "L_ELBOW": 45.0)
        """
        # rysuj punkty (opcjonalne)
        for i, pt in enumerate(points):
            if np.isnan(pt[0]) or np.isnan(pt[1]):
                continue
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            # jeśli chcesz numer punktu:
            # cv2.putText(frame, str(i), (x+5,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        # --------- łokcie (HAND - ELBOW - SHOULDER) ---------
        for pA_idx, pB_idx, pC_idx, label in self.elbow_triplets:
            pA = points[pA_idx.value]; pB = points[pB_idx.value]; pC = points[pC_idx.value]
            if np.any(np.isnan(pA)) or np.any(np.isnan(pB)) or np.any(np.isnan(pC)):
                continue

            # orientacje segmentów (matematyczne)
            ang1 = self.segment_angle_deg(pB, pA)  # pB->pA (np. łokieć -> ręka)
            ang2 = self.segment_angle_deg(pB, pC)  # pB->pC (np. łokieć -> bark)

            # promień sektora: połowa krótszego z segmentów (min(dist1,dist2)*0.5), min 12 px
            dist1 = math.hypot(float(pA[0]) - float(pB[0]), float(pA[1]) - float(pB[1]))
            dist2 = math.hypot(float(pC[0]) - float(pB[0]), float(pC[1]) - float(pB[1]))
            radius = int(max(12, min(dist1, dist2) * 0.5))

            # wartość kąta: najpierw spróbuj pobrać z przekazanego słownika
            angle_value = angles.get(label) if angles else None

            # jeśli brak, policz lokalnie (zwrotnie: 0..180)
            if angle_value is None:
                v1 = (float(pA[0]) - float(pB[0]), float(pA[1]) - float(pB[1]))
                v2 = (float(pC[0]) - float(pB[0]), float(pC[1]) - float(pB[1]))
                la = math.hypot(v1[0], v1[1]); lb = math.hypot(v2[0], v2[1])
                if la == 0 or lb == 0:
                    angle_value = None
                else:
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    cos_theta = max(-1.0, min(1.0, dot / (la*lb)))
                    angle_value = math.degrees(math.acos(cos_theta))

            # rysuj sektor (wypełniony, półprzezroczysty)
            self.draw_angle_sector(frame, pB, ang1, ang2, radius=radius, color=(0, 150, 255), filled=True, alpha=0.35)

            # rysuj tekst kąta przy bisektorze sektora
            if angle_value is not None:
                sweep = self.angle_diff_signed(ang1, ang2)
                bisector = ang1 + (sweep / 2.0 if sweep is not None else 0.0)
                text_r = int(radius * 1.15)
                bx = int(round(pB[0] + text_r * math.cos(math.radians(bisector))))
                by = int(round(pB[1] - text_r * math.sin(math.radians(bisector))))
                txt = f"{round(angle_value,1)}°"
                # czarny outline dla czytelności
                cv2.putText(frame, txt, (bx+1, by+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # --------- barki (C_HIP - C_SHOULDER - SHOULDER->HAND) ---------
        # tutaj pA = C_HIP, pB = C_SHOULDER, pC = HAND (ale label 'L_ARM' oznacza rysowanie sektora przy barku)
        # Uwaga: center dla barku to punkt C_SHOULDER (w tej implementacji)
        for pA_idx, pB_idx, pC_idx, label in self.shoulder_triplets:
            pA = points[pA_idx.value]; pB = points[pB_idx.value]; pC = points[pC_idx.value]
            if np.any(np.isnan(pA)) or np.any(np.isnan(pB)) or np.any(np.isnan(pC)):
                continue

            # orientacje: pB->pA (tu: torso direction), pB->pC (tu: shoulder->hand)
            ang1 = self.segment_angle_deg(pB, pA)
            ang2 = self.segment_angle_deg(pB, pC)

            # promień sektora mniejszy, żeby nie zasłaniać łokcia
            dist1 = math.hypot(float(pA[0]) - float(pB[0]), float(pA[1]) - float(pB[1]))
            dist2 = math.hypot(float(pC[0]) - float(pB[0]), float(pC[1]) - float(pB[1]))
            radius = int(max(14, min(dist1, dist2) * 0.4))

            # spróbuj pobrać wartość kąta ze słownika
            angle_value = angles.get(label) if angles else None

            # jeśli brak, policz jak wyżej
            if angle_value is None:
                v1 = (float(pA[0]) - float(pB[0]), float(pA[1]) - float(pB[1]))
                v2 = (float(pC[0]) - float(pB[0]), float(pC[1]) - float(pB[1]))
                la = math.hypot(v1[0], v1[1]); lb = math.hypot(v2[0], v2[1])
                if la == 0 or lb == 0:
                    angle_value = None
                else:
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    cos_theta = max(-1.0, min(1.0, dot / (la*lb)))
                    angle_value = math.degrees(math.acos(cos_theta))

            # rysuj sektor przy barku (inny kolor)
            self.draw_angle_sector(frame, pB, ang1, ang2, radius=radius, color=(60, 200, 60), filled=True, alpha=0.30)

            # tekst
            if angle_value is not None:
                sweep = self.angle_diff_signed(ang1, ang2)
                bisector = ang1 + (sweep / 2.0 if sweep is not None else 0.0)
                text_r = int(radius * 1.2)
                bx = int(round(pB[0] + text_r * math.cos(math.radians(bisector))))
                by = int(round(pB[1] - text_r * math.sin(math.radians(bisector))))
                txt = f"{round(angle_value,1)}°"
                cv2.putText(frame, txt, (bx+1, by+1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # pokaż okno
        cv2.imshow("Skeleton Visualization", frame)
        cv2.waitKey(1)

    # run() - możesz użyć tej samej funkcji run co wcześniej
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
