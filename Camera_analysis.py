import cv2
import numpy as np
from pathlib import Path

GRID_SIZE = (15, 10)
GRID_SIZE = (23, 16)
VALIDATION_IMAGE = "NAZWA_PLIKU_Z_SZACHOWNICA"

OUTPUT_DIR = "DOR_Z_WYNIKAMI"

# --- WSPÓŁCZYNNIKI DO HEATMAPY ---
W_ANGLE = 0.2  # kąty
W_SIDES = 0.65  # długości boków
W_AREA = 0.15  # pole


cfg = { # macierze kalibracji
    'camera_matrix': np.array([
        [661.65344583, 0., 319.11639216],
        [0., 662.0753099, 214.97976122],
        [0., 0., 1.]
    ], dtype=np.float64),
    'dist_matrix': np.array([[0.06560695, -0.27749729, -0.0015134, 0.00466734, -0.0823074]], dtype=np.float64).reshape(
        -1),
    'new_camera_matrix': np.array([
        [643.13205167, 0., 322.37088013],
        [0., 644.06947468, 212.72870091],
        [0., 0., 1.]
    ], dtype=np.float64),
    'roi': [10, 5, 622, 464]
}

def angle_between(v1, v2):
    cosang = np.clip(
        v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9),
        -1.0, 1.0
    )
    return np.degrees(np.arccos(cosang))


def cell_metrics(cell):
    tl, tr, br, bl = cell

    v_top = tr - tl
    v_right = br - tr
    v_bottom = bl - br
    v_left = tl - bl

    angles = np.array([
        angle_between(v_left, v_top),
        angle_between(v_top, v_right),
        angle_between(v_right, v_bottom),
        angle_between(v_bottom, v_left)
    ])

    angle_err = np.mean(np.abs(angles - 90.0))  # [deg]

    top = np.linalg.norm(v_top)
    bottom = np.linalg.norm(v_bottom)
    left = np.linalg.norm(v_left)
    right = np.linalg.norm(v_right)

    side_diff_px = 0.5 * (abs(top - bottom) + abs(left - right))  # [px]

    area = 0.5 * abs(
        tl[0]*tr[1] + tr[0]*br[1] + br[0]*bl[1] + bl[0]*tl[1]
        - tl[1]*tr[0] - tr[1]*br[0] - br[1]*bl[0] - bl[1]*tl[0]
    )

    approx_area = 0.25 * (top + bottom) * (left + right)
    area_diff_pct = abs(area - approx_area) / (approx_area + 1e-9) * 100.0  # %

    combined = (
        W_ANGLE * (angle_err / 90.0) +
        W_SIDES * (side_diff_px / 10.0) +
        W_AREA  * (area_diff_pct / 10.0)
    )

    return angle_err, side_diff_px, area_diff_pct, combined


def make_heatmap(img, data, title, unit, extra_text=None):
    # --- dynamiczny zakres legendy ---
    valid = data[np.isfinite(data)]
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))

    # 10% zapasu
    margin = 0.1 * (vmax - vmin + 1e-9)
    vmin -= margin
    vmax += margin

    # normalizacja
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)

    # --- legenda ---
    h, w = overlay.shape[:2]
    bar_x = w - 45
    bar_y1, bar_y2 = 40, h - 40

    for y in range(bar_y1, bar_y2):
        t = 1.0 - (y - bar_y1) / (bar_y2 - bar_y1)
        c = cv2.applyColorMap(np.uint8([[t * 255]]), cv2.COLORMAP_JET)[0, 0]
        cv2.line(overlay, (bar_x, y), (bar_x + 18, y), c.tolist(), 1)

    cv2.rectangle(overlay, (bar_x, bar_y1), (bar_x + 18, bar_y2), (255, 255, 255), 1)

    # opisy skali
    cv2.putText(overlay, f"{vmax:.2f} {unit}",
                (bar_x - 5, bar_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(overlay, f"{vmin:.2f} {unit}",
                (bar_x - 5, bar_y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # tytuł
    cv2.putText(overlay, title, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # wagi
    if extra_text:
        y = 55
        for line in extra_text:
            cv2.putText(overlay, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 18

    return overlay


def analyze():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(VALIDATION_IMAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    und = cv2.undistort(img, cfg['camera_matrix'],
                        cfg['dist_matrix'], None, cfg['new_camera_matrix'])
    und_gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(und_gray, GRID_SIZE)
    if not found:
        raise RuntimeError("Nie znaleziono szachownicy.")

    corners = corners.reshape(-1, 2)
    cols, rows = GRID_SIZE
    h, w = und_gray.shape

    angle_map = np.zeros((h, w), np.float32)
    side_map = np.zeros((h, w), np.float32)
    area_map = np.zeros((h, w), np.float32)
    comb_map = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)

    for j in range(rows - 1):
        for i in range(cols - 1):
            idx = j * cols + i
            quad = np.array([
                corners[idx],
                corners[idx + 1],
                corners[idx + cols + 1],
                corners[idx + cols]
            ])

            a, s, ar, c = cell_metrics(quad)

            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)

            angle_map += mask * a
            side_map += mask * s
            area_map += mask * ar
            comb_map += mask * c
            count += mask

    count[count == 0] = 1
    angle_map /= count
    side_map /= count
    area_map /= count
    comb_map /= count

    heat_angles = make_heatmap(
        und,
        angle_map,
        title="Średnia różnica kątów [°]",
        unit="°"
    )

    heat_sides = make_heatmap(
        und,
        side_map,
        title="Niezgodność długości boków [px]",
        unit="px"
    )

    heat_area = make_heatmap(
        und,
        area_map,
        title="Różnica pola komórek [%]",
        unit="%"
    )

    heat_combined = make_heatmap(
        und,
        comb_map,
        title="Łączny błąd kalibracji",
        unit="",
        extra_text=[
            "Wagi metryk:",
            f"kąty = {W_ANGLE}",
            f"boki = {W_SIDES}",
            f"pole = {W_AREA}"
        ]
    )

    cv2.imwrite(out / "heat_katy.png", heat_angles)
    cv2.imwrite(out / "heat_boki.png", heat_sides)
    cv2.imwrite(out / "heat_pole.png", heat_area)
    cv2.imwrite(out / f"heat_laczny{str(W_ANGLE)[1:]}{str(W_SIDES)[1:]}{str(W_AREA)[1:]}.png", heat_combined)
    print(f"Wyniki zapisane w folderze: {out.resolve()}")

if __name__ == "__main__":
    analyze()
