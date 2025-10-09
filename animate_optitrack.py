import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# === Konfiguracja / Wczytanie ===
filename = 'OptitrackAgataPassiveWAggressive3.csv'
frameRate = 100

raw = pd.read_csv(filename, header=None).values.astype(float)
raw[np.isinf(raw)] = np.nan
raw[raw == 0] = np.nan

size_skeleton = raw.shape
skeleton = np.zeros((size_skeleton[0]-6, 65), dtype=float)

skeleton[:,0:2] = raw[6:, 0:2]
skeleton[:,2:5]   = raw[6:, 237:240]  #1
skeleton[:,5:8]   = raw[6:, 224:227]  #2
skeleton[:,8:11]  = raw[6:, 211:214]  #3
skeleton[:,11:14] = raw[6:, 172:175]  #4
skeleton[:,14:17] = raw[6:, 185:188]  #5
skeleton[:,17:20] = raw[6:, 198:201]  #6
skeleton[:,20:23] = (raw[6:,211:214] + raw[6:,172:175]) / 2.0  #7
skeleton[:,23:26] = (((raw[6:,126:129] + raw[6:,129:132]) / 2.0) + ((raw[6:,77:80] + raw[6:,80:83]) / 2.0)) / 2.0  #8
skeleton[:,26:29] = raw[6:,58:61]   #9
skeleton[:,29:32] = raw[6:,61:64]   #10
skeleton[:,32:35] = raw[6:,156:159] #11
skeleton[:,35:38] = (raw[6:,139:142] + raw[6:,142:145]) / 2.0  #12
skeleton[:,38:41] = (raw[6:,126:129] + raw[6:,129:132]) / 2.0  #13
skeleton[:,41:44] = (raw[6:,77:80] + raw[6:,80:83]) / 2.0      #14
skeleton[:,44:47] = (raw[6:,90:93] + raw[6:,93:96]) / 2.0      #15
skeleton[:,47:50] = raw[6:,107:110]  #16
skeleton[:,50:53] = raw[6:,371:374]  #17
skeleton[:,53:56] = raw[6:,374:377]  #18
skeleton[:,56:59] = raw[6:,377:380]  #19
skeleton[:,59:62] = raw[6:,380:383]  #kamera (nr 20 w MATLABie)

# Dodaj punkt 21 (fixed) jako [0,0,0] (w oryginale wpisują do 51:53)
fixedPoint21 = np.tile(np.array([0.0,0.0,0.0]), (skeleton.shape[0],1))
skeleton[:,50:53] = fixedPoint21

nFrames = skeleton.shape[0]
dataOnly = skeleton[:, 2:]  # od trzeciej kolumny
nMarkers = dataOnly.shape[1] // 3

# przygotuj strukturę Data.markers [nMarkers x 3 x nFrames]
Data_markers = np.full((nMarkers, 3, nFrames), np.nan)
for f in range(nFrames):
    frameData = dataOnly[f,:]
    for m in range(nMarkers):
        idx = 3*m
        Data_markers[m, :, f] = frameData[idx:idx+3]

# Workspace vertices (kopiowane z MATLABa)
workspaceVertices = np.array([
    [0, 0, -0.5],
    [0, 0, 3.6],
    [4, 0, 3.6],
    [4, 2.5, 3.6],
    [0, 2.5, 3.6],
    [4, 0, -0.5],
    [0, 2.5, -0.5],
    [4, 2.5, -0.5],
    [0, 0, 1.53],
    [4, 0, 1.53],
    [0, 2.5, 1.53],
    [4, 2.5, 1.53],
])

workspaceEdges = np.array([
    [101,102],[102,103],[103,104],[104,105],[105,102],
    [107,108],[108,106],
    [101,106],[101,107],[103,106],[104,108],[105,107],
    [109,110],[110,112],[111,112],[111,109]
])

# Rysowanie (Matplotlib 3D)
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
plt.ion()
ax.set_box_aspect([1,1,1])

# Pierwsza klatka
M0 = Data_markers[:,:,0]
scatter = ax.scatter(M0[:,0], M0[:,1], M0[:,2], s=50)

# Linie szkieletu
skeletonLines = np.array([
    [1,2],[2,3],[3,7],[4,7],[4,5],[5,6],
    [7,8],[8,9],[9,10],
    [8,13],[13,12],[12,11],
    [8,14],[14,15],[15,16]
], dtype=int)

line_handles = []
for pair in skeletonLines:
    i1 = pair[0]-1
    i2 = pair[1]-1
    ln, = ax.plot([M0[i1,0], M0[i2,0]],
                  [M0[i1,1], M0[i2,1]],
                  [M0[i1,2], M0[i2,2]],
                  'g-', linewidth=2)
    line_handles.append(ln)

# Etykiety markerów
label_texts = []
for m in range(nMarkers):
    t = ax.text(M0[m,0], M0[m,1], M0[m,2], str(m+1), fontsize=8, verticalalignment='bottom')
    label_texts.append(t)

# Wierzchołki workspace
ax.scatter(workspaceVertices[:,0], workspaceVertices[:,1], workspaceVertices[:,2], s=60, c=(0,0.4,0))

# Etykiety workspace 101-112
for i, v in enumerate(workspaceVertices):
    ax.text(v[0], v[1], v[2], str(100+i+1), color=(0,0.4,0), fontsize=8)

# Linie workspace
for edge in workspaceEdges:
    idx1 = edge[0]-101
    idx2 = edge[1]-101
    v1 = workspaceVertices[idx1]
    v2 = workspaceVertices[idx2]
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color=(0,0.4,0), linewidth=2)

# Promienie z punktu kamery (nr 20 w MATLAB => cameraIdx=20 -> tu marker index 19)
cameraIdx = 19
horizontalFOV = math.radians(82)
verticalFOV = math.radians(52)
rayLength = 5.5
leftRayDir = np.array([-math.sin(horizontalFOV/2), 0, -math.cos(horizontalFOV/2)])
rightRayDir = np.array([ math.sin(horizontalFOV/2), 0, -math.cos(horizontalFOV/2)])
topRayDir = np.array([0, math.sin(verticalFOV/2), -math.cos(verticalFOV/2)])
bottomRayDir = np.array([0, -math.sin(verticalFOV/2), -math.cos(verticalFOV/2)])
violet = (0.5, 0, 0.5)

cameraPoint = M0[cameraIdx,:]
ray_lines = []
for d in [leftRayDir, rightRayDir, topRayDir, bottomRayDir]:
    ln, = ax.plot([cameraPoint[0], cameraPoint[0]+rayLength*d[0]],
                  [cameraPoint[1], cameraPoint[1]+rayLength*d[1]],
                  [cameraPoint[2], cameraPoint[2]+rayLength*d[2]],
                  color=violet, linewidth=2)
    ray_lines.append(ln)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Animacja optitrack')

# Ustawienia osi (dopasuj do danych)
ax.auto_scale_xyz([-1,5], [-1,3], [-1,4])

# Animacja
for f in range(nFrames):
    M = Data_markers[:,:,f]
    # update scatter
    scatter._offsets3d = (M[:,0], M[:,1], M[:,2])
    # update labels
    for m in range(nMarkers):
        label_texts[m].set_position((M[m,0], M[m,1]))
        label_texts[m].set_3d_properties(M[m,2], zdir='z')
    # update skeleton lines
    for idx, pair in enumerate(skeletonLines):
        i1 = pair[0]-1
        i2 = pair[1]-1
        x = [M[i1,0], M[i2,0]]
        y = [M[i1,1], M[i2,1]]
        z = [M[i1,2], M[i2,2]]
        line_handles[idx].set_data(x, y)
        line_handles[idx].set_3d_properties(z, zdir='z')
    # update rays (kamera)
    cameraPoint = M[cameraIdx,:]
    dirs = [leftRayDir, rightRayDir, topRayDir, bottomRayDir]
    for ri in range(4):
        d = dirs[ri]
        xs = [cameraPoint[0], cameraPoint[0] + rayLength*d[0]]
        ys = [cameraPoint[1], cameraPoint[1] + rayLength*d[1]]
        zs = [cameraPoint[2], cameraPoint[2] + rayLength*d[2]]
        ray_lines[ri].set_data(xs, ys)
        ray_lines[ri].set_3d_properties(zs, zdir='z')

    ax.set_title(f'Klatka {f+1} / {nFrames}')
    plt.draw()
    plt.pause(0.01)  # kontrola prędkości animacji

plt.ioff()
plt.show()
