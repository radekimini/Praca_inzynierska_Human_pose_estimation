import cv2
import numpy as np
from pathlib import Path

# --- Ustawienia użytkownika ---
GRID_SIZE = (23, 16)   # (columns, rows
VALIDATION_IMAGE = "frame_01444.png"  # ścieżka do obrazu walidacyjnego (szachownica)
OUT_PREFIX = "result_image_analisis"  # prefiks plików wynikowych

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
    h, w = img.shape[:2]
    und = cv2.undistort(img, mtx, dist, None, new_mtx)
    return und

def detect_corners(gray, grid_size):
    # OpenCV expects (cols, rows) = (nx, ny) in findChessboardCorners
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
    tl = corners[idx_tl]
    tr = corners[idx_tr]
    bl = corners[idx_bl]
    br = corners[idx_br]
    return np.array([tl, tr, br, bl], dtype=np.float32)  # kolejność: tl, tr, br, bl

def angle_between(v1, v2):
    # v1, v2 : 2d vectors
    dot = v1.dot(v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1==0 or n2==0:
        return 0.0
    cosang = np.clip(dot/(n1*n2), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return ang

def cell_error_metrics(cell_quad):
    tl, tr, br, bl = cell_quad
    # wektory boków
    v_top = tr - tl
    v_right = br - tr
    v_bottom = bl - br
    v_left = tl - bl

    # kąty (między bokami)
    ang_tl = angle_between(v_left, v_top)
    ang_tr = angle_between(v_top, v_right)
    ang_br = angle_between(v_right, v_bottom)
    ang_bl = angle_between(v_bottom, v_left)
    angle_devs = np.abs(np.array([ang_tl, ang_tr, ang_br, ang_bl]) - 90.0)

    # długości boków
    top_len = np.linalg.norm(v_top)
    right_len = np.linalg.norm(v_right)
    bottom_len = np.linalg.norm(v_bottom)
    left_len = np.linalg.norm(v_left)

    # różnice przeciwległych boków
    opp_diff_1 = np.abs(top_len - bottom_len) / max(1e-6, 0.5*(top_len+bottom_len))
    opp_diff_2 = np.abs(left_len - right_len) / max(1e-6, 0.5*(left_len+right_len))

    # pola (pola czworokąta przez podział na dwa trójkąty)
    area = 0.5 * np.abs(
        (tl[0]*tr[1] + tr[0]*br[1] + br[0]*bl[1] + bl[0]*tl[1]) -
        (tl[1]*tr[0] + tr[1]*br[0] + br[1]*bl[0] + bl[1]*tl[0])
    )

    # oczekiwane pole na podstawie średnich boków (estymacja)
    avg_h = 0.5*(left_len+right_len)
    avg_w = 0.5*(top_len+bottom_len)
    approx_area = avg_h * avg_w if (avg_h>0 and avg_w>0) else area

    # miary łączone -> skalujemy tak aby były w przybliżeniu porównywalne
    mean_angle_dev = np.mean(angle_devs)  # w stopniach
    rel_area_diff = np.abs(area - approx_area) / max(1e-6, approx_area)

    e_angle = mean_angle_dev / 90.0
    e_sides = 0.5*(opp_diff_1 + opp_diff_2)
    e_area = rel_area_diff

    # waga — możesz dopasować
    w_angle = 0.6
    w_sides = 0.25
    w_area = 0.15

    combined = w_angle*e_angle + w_sides*e_sides + w_area*e_area
    return {
        "combined": float(combined),
        "angle_deg": float(mean_angle_dev),
        "side_rel_diff": float(0.5*(opp_diff_1+opp_diff_2)),
        "area_rel_diff": float(rel_area_diff),
        "area": float(area)
    }

def rasterize_cell_to_map(error_map, cell_quad, error_value):
    # cell_quad: (4,2) float
    mask = np.zeros_like(error_map, dtype=np.uint8)
    pts = cell_quad.reshape((-1,1,2)).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 1)
    error_map += (mask * error_value)

# --- Główna procedura ---
def analyze_validation_image(img_path, cfg, grid_size):
    img_color, gray = load_image_gray(img_path)
    h, w = gray.shape
    mtx = cfg['camera_matrix']
    dist = cfg['dist_matrix']
    new_mtx = cfg.get('new_camera_matrix', None)
    if new_mtx is None:
        new_mtx = mtx

    und = undistort_image(img_color, mtx, dist, new_mtx)
    und_gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    found_und, corners_und = detect_corners(und_gray, grid_size)

    if not found_und:
        found_orig, corners_orig = detect_corners(gray, grid_size)
        if not found_orig:
            raise RuntimeError("Nie znaleziono szachownicy ani na obrazie oryginalnym ani na undistort. Sprawdź obraz walidacyjny.")
        pts = corners_orig.reshape(-1,1,2)
        pts_ud = cv2.undistortPoints(pts, mtx, dist, P=new_mtx)
        corners_und = pts_ud.reshape(-1,2)
        found_und = True

    cols, rows = grid_size
    error_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    per_cell_info = []

    for j in range(rows-1):
        for i in range(cols-1):
            quad = get_cell_corners(corners_und, grid_size, i, j)  # tl,tr,br,bl
            metrics = cell_error_metrics(quad)
            e = metrics['combined']
            per_cell_info.append(((i,j), quad, metrics))
            mask = np.zeros_like(error_map, dtype=np.uint8)
            pts = quad.reshape((-1,1,2)).astype(np.int32)
            cv2.fillConvexPoly(mask, pts, 1)
            error_map += mask * e
            count_map += mask.astype(np.float32)

    safe_count = np.where(count_map == 0, 1.0, count_map)
    avg_error_map = error_map / safe_count

    # gauss żeby uzyskać czytelniejszą heatmapę
    avg_error_map_blur = cv2.GaussianBlur(avg_error_map, (0,0), sigmaX=8, sigmaY=8)

    # normalizacja do 0..1
    minv, maxv = np.min(avg_error_map_blur), np.max(avg_error_map_blur)
    norm = (avg_error_map_blur - minv) / (maxv - minv + 1e-9)

    # tworzenie kolorowej mapy
    heat_8u = np.uint8(255 * norm)
    heat_color = cv2.applyColorMap(heat_8u, cv2.COLORMAP_JET)

    # overlay na undistorted
    overlay = cv2.addWeighted(und, 0.6, heat_color, 0.4, 0)

    per_cell_info_sorted = sorted(per_cell_info, key=lambda x: x[2]['combined'], reverse=True)

    for k, (ij, quad, metrics) in enumerate(per_cell_info_sorted[:8]):
        pts = quad.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255,255,255), thickness=1)
        center = np.mean(quad, axis=0).astype(int)
        cv2.putText(overlay, f"{metrics['combined']:.2f}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # zapis wyników
    cv2.imwrite(f"{OUT_PREFIX}_undistorted.png", und)
    cv2.imwrite(f"{OUT_PREFIX}_heatmap.png", heat_color)
    cv2.imwrite(f"{OUT_PREFIX}_overlay.png", overlay)

    print(f"Zapisano: {OUT_PREFIX}_undistorted.png, {OUT_PREFIX}_heatmap.png, {OUT_PREFIX}_overlay.png")
    print(f"Min/Max error (mapa): {minv:.6f} / {maxv:.6f}")

    # zwróć też per_cell_info dla możliwości dalszej analizy
    return {
        "undistorted": und,
        "heatmap_color": heat_color,
        "overlay": overlay,
        "error_map": avg_error_map_blur,
        "per_cell_info": per_cell_info_sorted
    }

# --- Uruchomienie ---
if __name__ == "__main__":
    img_path = VALIDATION_IMAGE
    if not Path(img_path).exists():
        raise SystemExit(f"Brak pliku walidacyjnego pod ścieżką: {img_path}. Podaj poprawną ścieżkę w VALIDATION_IMAGE.")
    res = analyze_validation_image(img_path, cfg, GRID_SIZE)

    # print("5 największych błędów w komórkach:")
    # for (ij, quad, metrics) in res['per_cell_info'][:5]:
    #     print(f" cell {ij} -> combined={metrics['combined']:.4f}, angle_deg={metrics['angle_deg']:.2f}, side_rel={metrics['side_rel_diff']:.3f}, area_rel={metrics['area_rel_diff']:.3f}")
