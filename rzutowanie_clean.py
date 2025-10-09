# rzutowanie_clean.py

import numpy as np
import pandas as pd
import cv2
import math

# === 1. Konfiguracja ===
csv_file = 'Radzio8_Casual.csv'
video_file = 'Take8_Calibrated10.mp4'
video_fps = 25
optitrack_fps = 120
frame_ratio = optitrack_fps / video_fps
start_video_frame = 1
total_output_frames = 1000

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    raise RuntimeError("Nie mogę otworzyć pliku wideo: " + video_file)

# === 2. Dane kamery ===
camPos = np.array([2.388, 1.25, 3.60], dtype=float)
lookDir = np.array([0.0, 0.0, 1.0], dtype=float)
upVec = np.array([0.0, 1.0, 0.0], dtype=float)

def normalize_vec(v):
    return v / (np.linalg.norm(v) + 1e-8)

camZ = normalize_vec(lookDir)
camX = normalize_vec(np.cross(upVec, camZ))
camY = normalize_vec(np.cross(camZ, camX))

R = np.vstack([camX, camY, camZ])
t = -R.dot(camPos.reshape(3,1)).flatten()

MV = np.eye(4)
MV[0:3,0:3] = R
MV[0:3,3] = t

fovY = np.deg2rad(90.0)
aspect = 4.0 / 3.0
zN = 2.7
zF = 3.29

tang = math.tan(fovY / 2.0)
S_y = 1.0 / tang
S_x = S_y / aspect

MP = np.zeros((4,4), dtype=float)
MP[0,0] = S_x
MP[1,1] = S_y
MP[2,2] = -(zF + zN) / (zF - zN)
MP[2,3] = -2.0 * zF * zN / (zF - zN)
MP[3,2] = -1.0

# === 3. Wczytaj dane z CSV ===
raw = pd.read_csv(csv_file, sep=',', skiprows=11).values.astype(float)

raw[np.isinf(raw)] = np.nan
raw[raw == 0] = np.nan

skeleton = np.zeros((raw.shape[0] - 6, 3*19), dtype=float)
skeleton[:,0:2] = raw[6:, 0:2]
skeleton[:, 2:5]   = raw[6:, 237:240]  #1
skeleton[:, 5:8]   = raw[6:, 224:227]  #2
skeleton[:, 8:11]  = raw[6:, 211:214]  #3
skeleton[:, 11:14] = raw[6:, 172:175]  #4
skeleton[:, 14:17] = raw[6:, 185:188]  #5
skeleton[:, 17:20] = raw[6:, 198:201]  #6
skeleton[:, 20:23] = (raw[6:,211:214] + raw[6:,172:175]) / 2.0  #7
skeleton[:, 23:26] = (((raw[6:,126:129] + raw[6:,129:132]) / 2.0) + ((raw[6:,77:80] + raw[6:,80:83]) / 2.0)) / 2.0  #8
skeleton[:, 26:29] = raw[6:,58:61]   #9
skeleton[:, 29:32] = raw[6:,61:64]   #10
skeleton[:, 32:35] = raw[6:,156:159] #11
skeleton[:, 35:38] = (raw[6:,139:142] + raw[6:,142:145]) / 2.0  #12
skeleton[:, 38:41] = (raw[6:,126:129] + raw[6:,129:132]) / 2.0  #13
skeleton[:, 41:44] = (raw[6:,77:80] + raw[6:,80:83]) / 2.0      #14
skeleton[:, 44:47] = (raw[6:,90:93] + raw[6:,93:96]) / 2.0      #15
skeleton[:, 47:50] = raw[6:,107:110]  #16

# Dodatkowe markery
# skeleton[:,50:53] = raw[6:,371:374]  # 17 (zakomentowane w oryginale)
# skeleton[:,53:56] = raw[6:,374:377]  # 18 (zakomentowane)
skeleton[:,56:59] = raw[6:,377:380]  # 19

N = skeleton.shape[0]

MW = [None]*19
pW_cartesian = [None]*19
p2D = [None]*19

for i in range(1,20):
    scale = 3.4
    theta = 0.0
    # translacje precyzyjne z oryginału
    translation = np.array([-4.65, -2.0, 0.0])
    if i == 1:
        translation = np.array([-4.65, -2.2, 0.0])
    if i == 8:
        translation = np.array([-4.65, -2.0, 0.0])
    if i == 11:
        translation = np.array([-4.75, -2.0, 0.0])
    if i == 15:
        translation = np.array([-4.65, -1.9, -0.1])
    if i == 16:
        translation = np.array([-4.55, -1.9, 0.0])

    theta_rad = math.radians(theta)
    S = np.eye(4)
    S[0,0] = scale; S[1,1] = scale; S[2,2] = scale

    Ry = np.eye(4)
    c = math.cos(theta_rad); s = math.sin(theta_rad)
    Ry[0,0] = c; Ry[0,2] = s
    Ry[2,0] = -s; Ry[2,2] = c

    T = np.eye(4)
    T[0:3,3] = translation

    MW[i-1] = T.dot(Ry).dot(S)

    p = skeleton[:, 3*(i-1):3*(i-1)+3]
    pL_h = np.hstack([p, np.ones((N,1))])
    pW = (MW[i-1].dot(pL_h.T)).T
    pW_cartesian[i-1] = pW[:,0:3]

    pC = (MV.dot(np.hstack([pW_cartesian[i-1], np.ones((N,1))]).T)).T
    pProj = (MP.dot(pC.T)).T
    with np.errstate(invalid='ignore', divide='ignore'):
        p2D[i-1] = pProj[:,0:2] / pProj[:,3:4]

# === 5. Rysowanie i zapis do wideo ===
skeletonLines = np.array([
    [1,2],[2,3],[3,7],[4,7],[4,5],[5,6],
    [7,8],[8,9],[9,10],
    [8,13],[13,12],[12,11],
    [8,14],[14,15],[15,16]
], dtype=int)

# ustawienia VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Błąd odczytu pierwszej klatki wideo")
h, w = first_frame.shape[:2]
out = cv2.VideoWriter('skeleton_overlay.mp4', fourcc, video_fps, (w,h))

# Jeśli chcesz by pętla zaczynała od frame 1 tak jak MATLAB, ustaw Pozycję
for vf in range(1, total_output_frames+1):
    frame_idx = int(round((vf - 1) * frame_ratio) + start_video_frame)
    if frame_idx > N:
        break

    cap.set(cv2.CAP_PROP_POS_FRAMES, vf-1)
    ret, frame = cap.read()
    if not ret:
        break
    img = frame.copy()

    points2D = np.full((19,2), np.nan)
    for i in range(1,20):
        p = skeleton[frame_idx-1, 3*(i-1):3*(i-1)+3]
        if np.any(np.isnan(p)):
            points2D[i-1,:] = [np.nan, np.nan]
            continue
        pL_h = np.hstack([p, 1.0])
        pW = MW[i-1].dot(pL_h)
        pC = MV.dot(pW)
        pProj = MP.dot(pC)
        if abs(pProj[3]) < 1e-8:
            points2D[i-1,:] = [np.nan, np.nan]
            continue
        pt2D = pProj[0:2] / pProj[3]
        img_width = w; img_height = h
        px = int((pt2D[0] + 1.0) / 2.0 * img_width)
        py = int((1.0 - (pt2D[1] + 1.0) / 2.0) * img_height)
        points2D[i-1,:] = [px, py]
        cv2.circle(img, (px,py), 6, (0,0,255), 2)
        cv2.putText(img, str(i), (px+5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)

    for e in range(skeletonLines.shape[0]):
        idx1 = skeletonLines[e,0]-1
        idx2 = skeletonLines[e,1]-1
        p1 = points2D[idx1]
        p2 = points2D[idx2]
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            continue
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), 2)

    cv2.putText(img, f'Klatka: {vf} (CSV {frame_idx})', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    out.write(img)
    # jeśli chcesz też podejrzeć na żywo, odkomentuj:
    # cv2.imshow('skeleton_overlay', img)
    # if cv2.waitKey(int(1000/video_fps)) & 0xFF == ord('q'):
    #     break

out.release()
cap.release()
cv2.destroyAllWindows()
print('✅ Wideo zapisane jako skeleton_overlay.mp4')
