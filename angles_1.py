import numpy as np
import pandas as pd
import cv2
import math

# === Konfiguracja ===
csv_file = 'Take 2025-05-13 02.53.csv'
video_file = '20250513_144441_scaled.avi'
video_fps = 25
optitrack_fps = 120
frame_ratio = optitrack_fps / video_fps
start_video_frame = 1
total_output_frames = 1000

# Wczytaj wideo
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    raise RuntimeError(f"Nie mogę otworzyć pliku wideo: {video_file}")

# === 2. Dane kamery ===
camPos = np.array([2.387222, 1.176419, 3.290892], dtype=float)
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
MP[2,2] = (zF + zN) / (zF - zN)
MP[2,3] = 2.0 * zF * zN / (zF - zN)
MP[3,2] = -1.0

# === 3. Wczytaj dane z CSV ===
raw = pd.read_csv(csv_file, sep=',', skiprows=11).values

raw = raw.astype(float)
raw[np.isinf(raw)] = np.nan
raw[raw == 0] = np.nan

# MATLAB: raw(7:end, ...) -> Python: raw[6:, ...]
skeleton = np.zeros((raw.shape[0] - 6, 3*19), dtype=float)

skeleton[:,0:2] = raw[6:, 0:2]
skeleton[:, 2:5]   = raw[6:, 237:240]  # 1
skeleton[:, 5:8]   = raw[6:, 224:227]  # 2
skeleton[:, 8:11]  = raw[6:, 211:214]  # 3
skeleton[:, 11:14] = raw[6:, 172:175]  # 4
skeleton[:, 14:17] = raw[6:, 185:188]  # 5
skeleton[:, 17:20] = raw[6:, 198:201]  # 6
skeleton[:, 20:23] = (raw[6:,211:214] + raw[6:,172:175]) / 2.0  # 7
skeleton[:, 23:26] = (((raw[6:,126:129] + raw[6:,129:132]) / 2.0) + ((raw[6:,77:80] + raw[6:,80:83]) / 2.0)) / 2.0  # 8
skeleton[:, 26:29] = raw[6:,58:61]   # 9
skeleton[:, 29:32] = raw[6:,61:64]   #10
skeleton[:, 32:35] = raw[6:,156:159] #11
skeleton[:, 35:38] = (raw[6:,139:142] + raw[6:,142:145]) / 2.0  #12
skeleton[:, 38:41] = (raw[6:,126:129] + raw[6:,129:132]) / 2.0  #13
skeleton[:, 41:44] = (raw[6:,77:80] + raw[6:,80:83]) / 2.0      #14
skeleton[:, 44:47] = (raw[6:,90:93] + raw[6:,93:96]) / 2.0      #15
skeleton[:, 47:50] = raw[6:,107:110]  #16

# Dodatkowy marker (19)
skeleton[:, 56:59] = raw[6:,377:380]  # 19

N = skeleton.shape[0]  # liczba klatek

MW = [None]*19
pW_cartesian = [None]*19
p2D = [None]*19

for i in range(1,20):  # i = 1..19
    scale = 3.4
    theta = 0.0  # stopnie

    # przypisz translacje
    if i in range(1,20):
        translation = np.array([-4.65, -2.2, 0.0])
    # poprawki jak w oryginalnym kodzie
    if i == 15:
        translation = np.array([-4.65, -1.9, -0.1])
    if i == 16:
        translation = np.array([-4.55, -1.9, 0.0])

    theta_rad = np.deg2rad(theta)
    S = np.eye(4)
    S[0:3,0:3] = scale * np.eye(3)

    Ry = np.eye(4)
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    Ry[[0,2],[0,2]] = c
    Ry[0,2] = s
    Ry[2,0] = -s

    T = np.eye(4)
    T[0:3,3] = translation

    MW[i-1] = T.dot(Ry).dot(S)

    # p = skeleton(:, (3*i):(3*i+2));
    start_col = 3*(i-1)
    p = skeleton[:, start_col:start_col+3]  # N x 3
    pL_h = np.hstack([p, np.ones((N,1))])   # N x 4
    pW = (MW[i-1].dot(pL_h.T)).T            # N x 4
    pW_cartesian[i-1] = pW[:,0:3]

    pC = (MV.dot(np.hstack([pW_cartesian[i-1], np.ones((N,1))]).T)).T
    pProj = (MP.dot(pC.T)).T
    # dzielenie przez W (czwarty komponent)
    with np.errstate(invalid='ignore', divide='ignore'):
        p2D[i-1] = pProj[:,0:2] / pProj[:,3:4]

# === 5. Rysowanie ===
skeletonLines = np.array([
    [1,2],[2,3],[3,7],[4,7],[4,5],[5,6],
    [7,8],[8,9],[9,10],
    [8,13],[13,12],[12,11],
    [8,14],[14,15],[15,16]
], dtype=int)  # 1-based indices

pairs_vert = np.array([[2,3],[4,5],[12,13],[14,15]], dtype=int)
joint_triplets = np.array([[1,2,3],[4,5,6],[11,12,13],[14,15,16]], dtype=int)

right_side_idx = [1,3]
left_side_idx  = [2,4]

# Pętla wyświetlająca
for vf in range(1, total_output_frames+1):
    frame_idx = int(round((vf-1) * frame_ratio) + start_video_frame)
    if frame_idx > N:
        break

    # odczytaj ramkę wideo (MATLAB read(video, vf) => tu iterujemy po kolei)
    # uwaga: VideoCapture numeruje od 0; ustawimy pozycję co do vf-1
    cap.set(cv2.CAP_PROP_POS_FRAMES, vf-1)
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()
    h, w = img.shape[:2]

    # Oś pionowa (lokalna "vert")
    p7  = pW_cartesian[6][frame_idx-1,:]
    p10 = pW_cartesian[9][frame_idx-1,:]
    v_vert = (p10 - p7)
    nrm = np.linalg.norm(v_vert)
    if nrm > 0:
        v_vert = v_vert / nrm
    else:
        v_vert = v_vert

    # znaki wg lokalnych układów w 12 i 15
    y11 = pW_cartesian[10][frame_idx-1,1]
    y12 = pW_cartesian[11][frame_idx-1,1]
    y15 = pW_cartesian[14][frame_idx-1,1]
    y16 = pW_cartesian[15][frame_idx-1,1]

    sign12 = 1
    if not (np.isnan(y11) or np.isnan(y12)):
        if y11 < y12:
            sign12 = -1

    sign15 = 1
    if not (np.isnan(y16) or np.isnan(y15)):
        if y16 < y15:
            sign15 = -1

    # Kąty segmentów vs vert
    angles_vert = np.zeros(4)
    for k in range(4):
        a = pairs_vert[k,0]
        b = pairs_vert[k,1]
        # convert 1-based -> 0-based
        u = pW_cartesian[b-1][frame_idx-1,:] - pW_cartesian[a-1][frame_idx-1,:]
        # jeśli NaN -> skip => zostanie 0
        if np.any(np.isnan(u)) or np.any(np.isnan(v_vert)):
            angles_vert[k] = np.nan
            continue
        ang = math.atan2(np.linalg.norm(np.cross(u, v_vert)), np.dot(u, v_vert))
        ang = min(ang, math.pi - ang)
        if a==12 and b==13:
            ang = sign12 * ang
        elif a==14 and b==15:
            ang = sign15 * ang
        angles_vert[k] = ang

    # Kąty stawów
    angles_joint = np.zeros(4)
    for k in range(4):
        tri = joint_triplets[k,:]
        u1 = pW_cartesian[tri[1]-1][frame_idx-1,:] - pW_cartesian[tri[0]-1][frame_idx-1,:]
        u2 = pW_cartesian[tri[2]-1][frame_idx-1,:] - pW_cartesian[tri[1]-1][frame_idx-1,:]
        if np.any(np.isnan(u1)) or np.any(np.isnan(u2)):
            angles_joint[k] = np.nan
            continue
        ang = math.atan2(np.linalg.norm(np.cross(u1,u2)), np.dot(u1,u2))
        ang = min(ang, math.pi - ang)
        if np.array_equal(tri, np.array([11,12,13])):
            ang = sign12 * ang
        elif np.array_equal(tri, np.array([14,15,16])):
            ang = sign15 * ang
        angles_joint[k] = ang

    # Kąt głowy
    head_vec = pW_cartesian[9][frame_idx-1,:] - pW_cartesian[8][frame_idx-1,:]
    if np.any(np.isnan(head_vec)) or np.any(np.isnan(v_vert)):
        head_angle = np.nan
    else:
        head_angle = math.atan2(np.linalg.norm(np.cross(head_vec, v_vert)), np.dot(head_vec, v_vert))
        head_angle = min(head_angle, math.pi - head_angle)

    # Napisy w prawym dolnym rogu
    baseX = w - 300
    baseY = h - 20
    dy = 18

    texts = []
    cols = []

    for k in range(4):
        txt1 = f"Seg {pairs_vert[k,0]}-{pairs_vert[k,1]} vs vert: {np.degrees(angles_vert[k]):.1f}°" if not np.isnan(angles_vert[k]) else f"Seg {pairs_vert[k,0]}-{pairs_vert[k,1]} vs vert: NaN"
        tri = joint_triplets[k,:]
        jdeg = np.degrees(angles_joint[k]) if not np.isnan(angles_joint[k]) else np.nan
        txt2 = f"Joint {tri[0]}-{tri[1]}-{tri[2]}: {jdeg:.1f}°" if not np.isnan(jdeg) else f"Joint {tri[0]}-{tri[1]}-{tri[2]}: NaN"

        if (k+1) in right_side_idx:
            col = (0, 0, 128)  # BGR
        else:
            col = (0, 102, 0)

        texts.append(txt1); cols.append(col)
        texts.append(txt2); cols.append(col)

    head_deg = np.degrees(head_angle) if not np.isnan(head_angle) else np.nan
    texts.append(f"Head angle: {head_deg:.1f}°" if not np.isnan(head_deg) else "Head angle: NaN")
    cols.append((153,0,153))

    # prostokąt tła półprzezroczysty
    rectW = 300
    rectH = dy * len(texts) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (baseX-5, baseY-rectH), (baseX-5+rectW, baseY), (255,255,255), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    # wypisywanie tekstów (użyj cv2.putText)
    for n, t in enumerate(texts):
        pos = (baseX, baseY - dy*n)
        col = cols[n]
        cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # Skeleton (bez zmian) - rysowanie punktów i linii
    points2D = np.full((19,2), np.nan)
    for i in range(1,20):
        p = skeleton[frame_idx-1, 3*(i-1):3*(i-1)+3]
        if np.any(np.isnan(p)):
            continue
        pL = np.hstack([p, 1.0])
        pW = MW[i-1].dot(pL)
        pC = MV.dot(pW)
        pProj = MP.dot(pC)
        if abs(pProj[3]) < 1e-8:
            continue
        pt = pProj[0:2] / pProj[3]
        # mapowanie z NDC [-1,1] do obrazka
        px = int((pt[0] + 1.0)/2.0 * w)
        py = int((1.0 - (pt[1] + 1.0)/2.0) * h)
        points2D[i-1,:] = [px, py]
        cv2.circle(img, (px,py), 6, (0,0,255), 2)
        cv2.putText(img, str(i), (px+5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)

    # rysowanie linii
    for e in range(skeletonLines.shape[0]):
        i1 = skeletonLines[e,0] - 1
        i2 = skeletonLines[e,1] - 1
        p1 = points2D[i1]
        p2 = points2D[i2]
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            continue
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), 2)

    cv2.imshow('angles_1 overlay', img)
    # Pauza zgodnie z fps
    if cv2.waitKey(int(1000/video_fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
