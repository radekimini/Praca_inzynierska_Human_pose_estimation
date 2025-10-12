import cv2
import numpy as np
from pathlib import Path

# --- Ustawienia użytkownika ---
GRID_SIZE = (23, 16)   # (cols, rows)
VALIDATION_IMAGE = "frame_01444.png"  # ścieżka do obrazu walidacyjnego
OUT_PREFIX = "result_image_analisis_2_m"  # prefiks plików wynikowych

# macierz kamery
cfg = {
    'camera_matrix': np.array([
        [659.1335331,    0.,         309.26599601],
        [0.,         658.13176293, 225.12565591],
        [0.,           0.,           1.        ]
    ], dtype=np.float64),
    'dist_matrix': np.array([[ 1.20261687e-01, -7.89795463e-01,  2.13838781e-03, -2.47974967e-05, 1.03624635e+00]], dtype=np.float64).reshape(-1),
    'new_camera_matrix': np.array([
        [651.0860892,    0.,         309.02327403],
        [0.,         650.56864018, 225.98281995],
        [0.,           0.,           1.        ]
    ], dtype=np.float64),
    'roi': [3, 4, 632, 471]
}

# --- Funkcje pomocnicze ---

def load_image_gray(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def undistort_image(img, mtx, dist, new_mtx):
    und = cv2.undistort(img, mtx, dist, None, new_mtx)
    return und

def detect_corners(gray, grid_size):
    found, corners = cv2.findChessboardCorners(gray, grid_size, None)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    return True, corners2.reshape(-1,2)

def get_cell_corners(corners, grid_size, i, j):
    cols, rows = grid_size
    idx_tl = j*cols + i
    idx_tr = j*cols + (i+1)
    idx_bl = (j+1)*cols + i
    idx_br = (j+1)*cols + (i+1)
    tl = corners[idx_tl]; tr = corners[idx_tr]; bl = corners[idx_bl]; br = corners[idx_br]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def angle_between(v1, v2):
    dot = v1.dot(v2)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1==0 or n2==0: return 0.0
    cosang = np.clip(dot/(n1*n2), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def cell_error_metrics_and_sides(cell_quad):
    tl, tr, br, bl = cell_quad
    v_top = tr - tl; v_right = br - tr; v_bottom = bl - br; v_left = tl - bl

    angs = np.array([
        angle_between(v_left, v_top),
        angle_between(v_top, v_right),
        angle_between(v_right, v_bottom),
        angle_between(v_bottom, v_left)
    ])
    angle_dev_mean = np.mean(np.abs(angs - 90.0))

    top_len = np.linalg.norm(v_top)
    right_len = np.linalg.norm(v_right)
    bottom_len = np.linalg.norm(v_bottom)
    left_len = np.linalg.norm(v_left)

    opp_diff_1 = np.abs(top_len - bottom_len) / max(1e-6, 0.5*(top_len+bottom_len))
    opp_diff_2 = np.abs(left_len - right_len) / max(1e-6, 0.5*(left_len+right_len))

    area = 0.5 * np.abs(
        (tl[0]*tr[1] + tr[0]*br[1] + br[0]*bl[1] + bl[0]*tl[1]) -
        (tl[1]*tr[0] + tr[1]*br[0] + br[1]*bl[0] + bl[1]*tl[0])
    )
    avg_h = 0.5*(left_len+right_len); avg_w = 0.5*(top_len+bottom_len)
    approx_area = avg_h * avg_w if (avg_h>0 and avg_w>0) else area
    rel_area_diff = np.abs(area - approx_area) / max(1e-6, approx_area)

    e_angle = angle_dev_mean / 90.0
    e_sides = 0.5*(opp_diff_1 + opp_diff_2)
    e_area = rel_area_diff
    w_angle = 0.6; w_sides = 0.25; w_area = 0.15
    combined = w_angle*e_angle + w_sides*e_sides + w_area*e_area

    return {
        "combined": float(combined),
        "angle_deg": float(angle_dev_mean),
        "top_len": float(top_len),
        "bottom_len": float(bottom_len),
        "left_len": float(left_len),
        "right_len": float(right_len),
        "area": float(area),
        "area_rel_diff": float(rel_area_diff),
        "side_rel_diff": float(0.5*(opp_diff_1+opp_diff_2))
    }

# --- Interaktywna korekta ekstremów ---
class ClickCollector:
    def __init__(self, img, pre_points=None):
        self.img = img.copy()
        self.points = [] if pre_points is None else [tuple(map(int,p)) for p in pre_points]
        self.window = "TL, TR, BR, BL ('s' to save, 'r' to reset, ESC to cancel)"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x,y))

    def run(self):
        while True:
            vis = self.img.copy()
            for i, p in enumerate(self.points):
                cv2.circle(vis, p, 6, (0,255,0), -1)
                cv2.putText(vis, f"{i+1}", (p[0]+8,p[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            instruction = "Click TL,TR,BR,BL. 's' save, 'r' reset, ESC cancel"
            cv2.putText(vis, instruction, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow(self.window, vis)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:  # ESC
                cv2.destroyWindow(self.window)
                return None
            if k == ord('r'):
                self.points = []
            if k == ord('s'):
                if len(self.points) == 4:
                    cv2.destroyWindow(self.window)
                    return np.array(self.points, dtype=np.float32)
                else:
                    print("TL,TR,BR,BL")

def apply_corners_homography(corners, src_extremes, dst_extremes):
    H, _ = cv2.findHomography(src_extremes, dst_extremes)
    pts = corners.reshape(-1,1,2).astype(np.float32)
    pts_t = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    return pts_t

def analyze_validation_image_with_manual(img_path, cfg, grid_size):
    img_color, gray = load_image_gray(img_path)
    h, w = gray.shape
    mtx = cfg['camera_matrix']; dist = cfg['dist_matrix']; new_mtx = cfg.get('new_camera_matrix', mtx)

    und = undistort_image(img_color, mtx, dist, new_mtx)
    und_gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

    found_und, corners_und = detect_corners(und_gray, grid_size)
    if not found_und:
        found_orig, corners_orig = detect_corners(gray, grid_size)
        if not found_orig:
            raise RuntimeError("Nie znaleziono szachownicy na oryginale i na undistort.")
        pts = corners_orig.reshape(-1,1,2)
        pts_ud = cv2.undistortPoints(pts, mtx, dist, P=new_mtx)
        corners_und = pts_ud.reshape(-1,2)
        found_und = True

    cols, rows = grid_size

    # obliczanie domyślnych ekstremalne rogi (TL, TR, BR, BL)
    src_extremes = np.array([
        corners_und[0],                         # TL
        corners_und[cols-1],                    # TR
        corners_und[(rows-1)*cols + (cols-1)],  # BR
        corners_und[(rows-1)*cols + 0]          # BL
    ], dtype=np.float32)

    preview = und.copy()
    for i, p in enumerate(src_extremes):
        cv2.circle(preview, tuple(p.astype(int)), 8, (0,0,255), -1)
        cv2.putText(preview, f"E{i+1}", tuple((p+np.array([6,6])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("Detected extremes (press 'm' to manual adjust, any other key to continue)", preview)
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Detected extremes (press 'm' to manual adjust, any other key to continue)")

    if k == ord('m'):
        cc = ClickCollector(und, pre_points=src_extremes)
        dst_extremes = cc.run()
        if dst_extremes is not None:
            corners_und = apply_corners_homography(corners_und, src_extremes, dst_extremes)
            src_extremes = dst_extremes  # zaktualizuj
            print("Zastosowano ręczną korekcję ekstremów i przeliczono rogi przez homografię.")
        else:
            print("Manual correction cancelled - używam automatycznych ekstremów.")

    # metryki per komórka i tworzenie map
    error_map = np.zeros((h,w), dtype=np.float32)
    count_map = np.zeros((h,w), dtype=np.float32)
    per_cell_info = []

    for j in range(rows-1):
        for i in range(cols-1):
            quad = get_cell_corners(corners_und, grid_size, i, j)
            metrics = cell_error_metrics_and_sides(quad)
            per_cell_info.append(((i,j), quad, metrics))
            mask = np.zeros_like(error_map, dtype=np.uint8)
            pts = quad.reshape((-1,1,2)).astype(np.int32)
            cv2.fillConvexPoly(mask, pts, 1)
            error_map += mask * metrics['combined']
            count_map += mask.astype(np.float32)

    safe_count = np.where(count_map == 0, 1.0, count_map)
    avg_error_map = error_map / safe_count
    avg_error_map_blur = cv2.GaussianBlur(avg_error_map, (0,0), sigmaX=8, sigmaY=8)
    minv, maxv = float(np.min(avg_error_map_blur)), float(np.max(avg_error_map_blur))
    norm = (avg_error_map_blur - minv) / (maxv - minv + 1e-9)
    heat_8u = np.uint8(255 * norm)
    heat_color = cv2.applyColorMap(heat_8u, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(und, 0.6, heat_color, 0.4, 0)

    # raport graficzny
    detailed = und.copy()
    for (i_j, quad, metrics) in per_cell_info:
        (i,j) = i_j
        center = np.mean(quad, axis=0).astype(int)
        # formatowanie tekstu do 2 linii, z ograniczeniem długości
        line1 = f"T{metrics['top_len']:.1f} B{metrics['bottom_len']:.1f}"
        line2 = f"L{metrics['left_len']:.1f} R{metrics['right_len']:.1f}"
        line3 = f"A{metrics['angle_deg']:.1f}"
        # rysuj tło prostokątne dla czytelności
        cv2.rectangle(detailed, tuple(center+np.array([-50,-22])), tuple(center+np.array([50,28])), (0,0,0), -1)
        cv2.putText(detailed, line1, tuple(center+np.array([-46,-6])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(detailed, line2, tuple(center+np.array([-46,8])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(detailed, line3, tuple(center+np.array([-46,22])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    for idx, p in enumerate(src_extremes):
        p_int = tuple(p.astype(int))
        cv2.circle(detailed, p_int, 10, (0,0,255), -1)
        cv2.putText(detailed, f"E{idx+1}:{p_int}", (p_int[0]+12, p_int[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # zapis plików
    cv2.imwrite(f"{OUT_PREFIX}_undistorted.png", und)
    cv2.imwrite(f"{OUT_PREFIX}_heatmap.png", heat_color)
    cv2.imwrite(f"{OUT_PREFIX}_overlay.png", overlay)
    cv2.imwrite(f"{OUT_PREFIX}_detailed_report.png", detailed)

    print(f"Zapisano: {OUT_PREFIX}_undistorted.png, {OUT_PREFIX}_heatmap.png, {OUT_PREFIX}_overlay.png, {OUT_PREFIX}_detailed_report.png")
    print(f"Min/Max error (mapa): {minv:.6f} / {maxv:.6f}")

    # zwróć dane
    per_cell_info_sorted = sorted(per_cell_info, key=lambda x: x[2]['combined'], reverse=True)
    return {
        "undistorted": und,
        "heatmap_color": heat_color,
        "overlay": overlay,
        "detailed": detailed,
        "error_map": avg_error_map_blur,
        "per_cell_info": per_cell_info_sorted,
        "extremes": src_extremes
    }

# --- Uruchomienie ---
if __name__ == "__main__":
    if not Path(VALIDATION_IMAGE).exists():
        raise SystemExit(f"Brak pliku walidacyjnego pod ścieżką: {VALIDATION_IMAGE}. Podaj poprawną ścieżkę.")
    res = analyze_validation_image_with_manual(VALIDATION_IMAGE, cfg, GRID_SIZE)

    print("5 największych błędów (combined) w komórkach:")
    for (ij, quad, metrics) in res['per_cell_info'][:5]:
        print(f" cell {ij} -> combined={metrics['combined']:.4f}, angle_deg={metrics['angle_deg']:.2f}, top={metrics['top_len']:.1f}, bottom={metrics['bottom_len']:.1f}, left={metrics['left_len']:.1f}, right={metrics['right_len']:.1f}")
    print("Ekstremalne rogi (TL,TR,BR,BL):")
    for i,p in enumerate(res['extremes']):
        print(f" E{i+1}: {tuple(p.astype(int))}")
